import os
import sys
import time
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import numba

# -----------------------------------------------------------------------------
# Fast Numba Dynamic Time Warping (DTW)
# -----------------------------------------------------------------------------
@numba.njit(fastmath=True)
def numba_dtw_l1(x, y):
    T1, D = x.shape
    T2, _ = y.shape
    
    dp = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            cost = 0.0
            for d in range(D):
                cost += abs(x[i-1, d] - y[j-1, d])
            
            m = dp[i-1, j]
            if dp[i, j-1] < m:
                m = dp[i, j-1]
            if dp[i-1, j-1] < m:
                m = dp[i-1, j-1]
            
            dp[i, j] = cost + m
            
    return dp[T1, T2] / (T1 + T2)


def count_parameters(module):
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def main():
    parser = argparse.ArgumentParser(description="MambaFlow-TTS Benchmark Engine")
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of model architecture")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--val_csv", type=str, default="/home/monesh/ljspeech/LJSpeech-1.1/val.csv")
    parser.add_argument("--stats_path", type=str, default="/home/monesh/ljspeech/LJSpeech-1.1/stats.py")
    parser.add_argument("--output", type=str, required=True, help="Path to save output JSON")
    parser.add_argument("--n_timesteps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.667)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.model_id}] Starting benchmark on device: {device}", flush=True)

    # 1. Clean import path & environment
    abs_model_dir = os.path.abspath(args.model_dir)
    sys.path.insert(0, abs_model_dir)
    sys.path.insert(0, os.path.abspath("."))

    # Import modules from the isolated model directory
    from TTSDataModule import TTSMODEL
    from TTSDatasetModule import load_normalization_stats
    from preprocessing.text import text_to_sequence

    # 2. Instantiate and Load Checkpoint strictly
    lightning_model = TTSMODEL()
    ckpt_data = torch.load(args.ckpt, map_location=device)
    sd = ckpt_data.get("state_dict", ckpt_data)
    clean_sd = {k.replace('model.', ''): v for k, v in sd.items()}

    # Strict load guarantees 100% parameter accuracy
    lightning_model.model.load_state_dict(clean_sd, strict=True)
    model = lightning_model.model.to(device).eval()

    # Parameter Analysis
    total_params = count_parameters(model)
    encoder_params = count_parameters(getattr(model, "encoder", None))
    duration_params = count_parameters(getattr(model, "duration", None))
    decoder_params = count_parameters(getattr(model, "decoder", None))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt_size_mb = os.path.getsize(args.ckpt) / (1024 * 1024)

    param_stats = {
        "model_id": args.model_id,
        "total_params": total_params,
        "encoder_params": encoder_params,
        "duration_params": duration_params,
        "decoder_params": decoder_params,
        "trainable_params": trainable_params,
        "ckpt_size_mb": round(ckpt_size_mb, 2),
    }
    print(f"[{args.model_id}] Parameters: Total={total_params:,}, Encoder={encoder_params:,}, "
          f"Duration={duration_params:,}, Decoder={decoder_params:,}, Ckpt={ckpt_size_mb:.2f}MB", flush=True)

    # 3. Load Normalization Statistics directly from stats.py or stats.npy
    if args.stats_path.endswith(".py") and os.path.exists(args.stats_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ljspeech_stats", args.stats_path)
        stats_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stats_mod)
        stats_mean_np = np.array(stats_mod.MEL_MEAN, dtype=np.float32)
        stats_std_np = np.array(stats_mod.MEL_STD, dtype=np.float32)
    else:
        stats_mean, stats_std = load_normalization_stats(args.stats_path, dim=100)
        stats_mean_np = stats_mean.astype(np.float32)
        stats_std_np = stats_std.astype(np.float32)

    # 4. Load Dataset
    df = pd.read_csv(args.val_csv)
    if args.max_samples is not None:
        df = df.iloc[:args.max_samples]
    total_samples = len(df)
    print(f"[{args.model_id}] Processing {total_samples} validation samples...", flush=True)

    # Warmup GPU
    dummy_seq, _ = text_to_sequence("Benchmark warmup run", ["english_cleaners2"])
    dummy_x = torch.tensor(dummy_seq, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        _ = model(x=dummy_x, n_timesteps=5)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Metrics Accumulator
    metrics_list = []
    start_all_time = time.perf_counter()

    for idx, row in df.iterrows():
        text = ""
        for col in ['normalized_text', 'normalized', 'text', 'transcript', 'txt']:
            if col in row and pd.notna(row[col]):
                text = str(row[col])
                break

        mel_path = ""
        for col in ['mel_path', 'mel_spectrogram_path', 'mel']:
            if col in row and pd.notna(row[col]):
                mel_path = str(row[col])
                break

        if not os.path.exists(mel_path):
            continue

        # Load Ground Truth Mel
        mel_raw = np.squeeze(np.load(mel_path)).astype(np.float32)
        if mel_raw.ndim == 2 and mel_raw.shape[0] == 100 and mel_raw.shape[1] != 100:
            mel_raw = mel_raw.T  # [T_gt, 100]

        mel_norm = (mel_raw - stats_mean_np) / (stats_std_np + 1e-8)
        gt_tensor = torch.tensor(mel_norm, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T_gt, 100]
        T_gt = gt_tensor.size(1)

        # Convert text to phoneme sequence
        seq, _ = text_to_sequence(text, ["english_cleaners2"])
        x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            # Encoder & to_latent
            x_in = model.emb_dropout(model.txt_emb(x))
            enc_out = model.encoder(x_in, mask=None)
            mu_text_base = model.to_latent(enc_out)

            # -------------------------------------------------------------
            # 1. Aligned Reconstruction (Oracle MAS Alignment)
            # -------------------------------------------------------------
            alignment, _, _ = model.duration(
                mu_text_base,
                latent=gt_tensor,
                mask=None,
                audio_mask=None,
                mas_prior=mu_text_base
            )
            durations_aligned = alignment.sum(dim=-1).long()

            latent_expanded_aligned = torch.repeat_interleave(mu_text_base[0], durations_aligned[0], dim=0).unsqueeze(0)
            min_T = min(gt_tensor.size(1), latent_expanded_aligned.size(1))
            gt_aligned = gt_tensor[:, :min_T, :]
            mu_aligned = latent_expanded_aligned[:, :min_T, :]

            pred_aligned_norm = model.decoder(
                mu=mu_aligned,
                mask=None,
                target=None,
                n_timesteps=args.n_timesteps,
                temperature=args.temperature,
                solver=args.solver,
                t_end=1.0
            )

            # -------------------------------------------------------------
            # 2. Free-Running Synthesis & Duration Prediction
            # -------------------------------------------------------------
            dur_free, _, _ = model.duration(mu_text_base, latent=None, mask=None)
            durations_free = dur_free.clamp(min=1)

            latent_expanded_free = torch.repeat_interleave(mu_text_base[0], durations_free[0], dim=0).unsqueeze(0)

            # Time free-running inference for RTF measurement
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            pred_free_norm = model.decoder(
                mu=latent_expanded_free,
                mask=None,
                target=None,
                n_timesteps=args.n_timesteps,
                temperature=args.temperature,
                solver=args.solver,
                t_end=1.0
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            infer_time = t_end - t_start

        # -----------------------------------------------------------------
        # Compute Numerical Metrics
        # -----------------------------------------------------------------
        # Aligned Mel Reconstruction Metrics
        diff = pred_aligned_norm - gt_aligned
        mae_norm = torch.mean(torch.abs(diff)).item()
        mse_norm = torch.mean(diff ** 2).item()
        rmse_norm = np.sqrt(mse_norm)

        # Raw (Denormalized) Scale Metrics
        stats_std_t = torch.tensor(stats_std_np, device=device).unsqueeze(0).unsqueeze(0)
        diff_raw = diff * stats_std_t
        mae_raw = torch.mean(torch.abs(diff_raw)).item()
        mse_raw = torch.mean(diff_raw ** 2).item()
        rmse_raw = np.sqrt(mse_raw)

        # Sub-Band Frequency Decomposition (Channels: Low 0-33, Mid 33-66, High 66-100)
        mae_low = torch.mean(torch.abs(diff_raw[:, :, :33])).item()
        mae_mid = torch.mean(torch.abs(diff_raw[:, :, 33:66])).item()
        mae_high = torch.mean(torch.abs(diff_raw[:, :, 66:])).item()

        # Directional & Envelope Alignment Metrics
        cos_sim = F.cosine_similarity(pred_aligned_norm, gt_aligned, dim=-1).mean().item()

        # Pearson Correlation Coefficient (r)
        p_c = pred_aligned_norm - pred_aligned_norm.mean(dim=-1, keepdim=True)
        g_c = gt_aligned - gt_aligned.mean(dim=-1, keepdim=True)
        denom = torch.sqrt((p_c ** 2).sum(dim=-1) * (g_c ** 2).sum(dim=-1)) + 1e-8
        pearson_r = (p_c * g_c).sum(dim=-1).div(denom).mean().item()

        # Spectral Convergence (SC)
        sc = (torch.norm(gt_aligned - pred_aligned_norm, p='fro') / (torch.norm(gt_aligned, p='fro') + 1e-8)).item()

        # Duration & Alignment Metrics
        T_pred = pred_free_norm.size(1)
        dur_err_frames = abs(T_pred - T_gt)
        dur_err_sec = dur_err_frames * 256.0 / 24000.0
        audio_dur_sec = T_pred * 256.0 / 24000.0

        rtf = infer_time / audio_dur_sec if audio_dur_sec > 0 else 0.0

        # Fast DTW Distance on Free-Running vs Ground Truth
        pred_free_np = pred_free_norm[0].cpu().numpy().astype(np.float32)
        gt_full_np = gt_tensor[0].cpu().numpy().astype(np.float32)
        dtw_dist_norm = float(numba_dtw_l1(pred_free_np, gt_full_np))

        sample_record = {
            "sample_idx": idx,
            "T_gt": int(T_gt),
            "T_pred": int(T_pred),
            "dur_err_frames": int(dur_err_frames),
            "dur_err_sec": float(dur_err_sec),
            "audio_dur_sec": float(audio_dur_sec),
            "infer_time_sec": float(infer_time),
            "rtf": float(rtf),
            "mae_norm": float(mae_norm),
            "rmse_norm": float(rmse_norm),
            "mae_raw": float(mae_raw),
            "rmse_raw": float(rmse_raw),
            "mae_low": float(mae_low),
            "mae_mid": float(mae_mid),
            "mae_high": float(mae_high),
            "cos_sim": float(cos_sim),
            "pearson_r": float(pearson_r),
            "spectral_conv": float(sc),
            "dtw_dist_norm": float(dtw_dist_norm),
        }
        metrics_list.append(sample_record)

        if (len(metrics_list) % 100 == 0) or (len(metrics_list) == total_samples):
            elapsed = time.perf_counter() - start_all_time
            rate = len(metrics_list) / elapsed
            print(f"[{args.model_id}] {len(metrics_list)}/{total_samples} samples ({rate:.1f} samp/s) | "
                  f"MAE_raw: {np.mean([m['mae_raw'] for m in metrics_list]):.3f} | "
                  f"CosSim: {np.mean([m['cos_sim'] for m in metrics_list]):.4f} | "
                  f"SC: {np.mean([m['spectral_conv'] for m in metrics_list]):.4f} | "
                  f"RTF: {np.mean([m['rtf'] for m in metrics_list]):.4f}", flush=True)

    # -------------------------------------------------------------------------
    # Aggregate Summary Statistics
    # -------------------------------------------------------------------------
    def agg(key):
        vals = [m[key] for m in metrics_list]
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
        }

    summary = {
        "params": param_stats,
        "n_samples": len(metrics_list),
        "total_benchmark_time_sec": time.perf_counter() - start_all_time,
        "metrics": {
            "mae_norm": agg("mae_norm"),
            "rmse_norm": agg("rmse_norm"),
            "mae_raw": agg("mae_raw"),
            "rmse_raw": agg("rmse_raw"),
            "mae_low": agg("mae_low"),
            "mae_mid": agg("mae_mid"),
            "mae_high": agg("mae_high"),
            "cos_sim": agg("cos_sim"),
            "pearson_r": agg("pearson_r"),
            "spectral_conv": agg("spectral_conv"),
            "dur_err_frames": agg("dur_err_frames"),
            "dur_err_sec": agg("dur_err_sec"),
            "dtw_dist_norm": agg("dtw_dist_norm"),
            "rtf": agg("rtf"),
        },
        "samples": metrics_list
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{args.model_id}] BENCHMARK COMPLETE! Summary saved to {args.output}\n", flush=True)


if __name__ == "__main__":
    main()
