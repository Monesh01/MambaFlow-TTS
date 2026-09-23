import os
import sys
import json
import time
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

# Quality metrics
try:
    from pesq import pesq
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pesq"])
    from pesq import pesq

try:
    import torchaudio
    import torchaudio.functional as TAF
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchaudio"])
    import torchaudio
    import torchaudio.functional as TAF

from bigvgan import BigVGAN, AttrDict

# Local codebase imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from TTSDataModule import TTSDataModule, TTSMODEL
from TTSDatasetModule import load_normalization_stats
from preprocessing.text import text_to_sequence


def compute_numerical_stability(p: torch.Tensor, t: torch.Tensor):
    """
    Computes numerical stability and distribution health metrics between
    predicted mel (p) and target mel (t).
    """
    nan_count = int(torch.isnan(p).sum().item())
    inf_count = int(torch.isinf(p).sum().item())
    is_finite = bool(nan_count == 0 and inf_count == 0)

    if not is_finite:
        return {
            "is_finite": False,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "stability_score": 0.0,
            "variance_ratio": 0.0,
            "extreme_outliers": -1,
            "mae_norm": float("inf"),
            "mse_norm": float("inf"),
            "rmse_norm": float("inf"),
            "cosine_similarity": 0.0,
            "pearson_correlation": 0.0,
            "p_min": float("nan"),
            "p_max": float("nan"),
            "p_mean": float("nan"),
            "p_std": float("nan"),
        }

    p_min = float(p.min().item())
    p_max = float(p.max().item())
    p_mean = float(p.mean().item())
    p_std = float(p.std().item())
    t_mean = float(t.mean().item())
    t_std = float(t.std().item())

    # Variance ratio: pred_std / target_std (ideal = 1.0)
    variance_ratio = float(p_std / (t_std + 1e-8))

    # Extreme outlier count (|p| > 10.0 in z-normalized space)
    extreme_outliers = int((torch.abs(p) > 10.0).sum().item())

    # Stability score [0 to 100%]
    stability_score = 100.0
    if variance_ratio < 0.2 or variance_ratio > 3.0:
        stability_score -= 40.0
    elif variance_ratio < 0.6 or variance_ratio > 1.4:
        stability_score -= 15.0

    if extreme_outliers > 0:
        stability_score -= min(30.0, extreme_outliers * 0.5)
    stability_score = max(0.0, stability_score)

    # MAE, MSE, RMSE on aligned length
    min_t = min(p.size(1), t.size(1))
    p_aligned = p[:, :min_t, :]
    t_aligned = t[:, :min_t, :]

    mae = float(F.l1_loss(p_aligned, t_aligned).item())
    mse = float(F.mse_loss(p_aligned, t_aligned).item())
    rmse = float(np.sqrt(mse))

    # Cosine similarity and Pearson correlation
    p_flat = p_aligned.detach().cpu().numpy().flatten()
    t_flat = t_aligned.detach().cpu().numpy().flatten()

    norm_p = np.linalg.norm(p_flat)
    norm_t = np.linalg.norm(t_flat)
    cos_sim = float(np.dot(p_flat, t_flat) / (norm_p * norm_t + 1e-8))

    std_prod = np.std(p_flat) * np.std(t_flat)
    if std_prod > 1e-8:
        pearson = float(np.corrcoef(p_flat, t_flat)[0, 1])
    else:
        pearson = 0.0

    return {
        "is_finite": True,
        "nan_count": 0,
        "inf_count": 0,
        "stability_score": stability_score,
        "variance_ratio": variance_ratio,
        "extreme_outliers": extreme_outliers,
        "mae_norm": mae,
        "mse_norm": mse,
        "rmse_norm": rmse,
        "cosine_similarity": cos_sim,
        "pearson_correlation": pearson,
        "p_min": p_min,
        "p_max": p_max,
        "p_mean": p_mean,
        "p_std": p_std,
    }


def main():
    torch.set_float32_matmul_precision("high")

    # 1. Dataset & Paths setup
    train_csv_path = "/home/monesh/ljspeech/LJSpeech-1.1/train.csv"
    stats_path = "/home/monesh/ljspeech/LJSpeech-1.1/stats.py"
    subset_csv_path = "train_bigvgan_subset10.csv"
    ckpt_dir = "subset_checkpoints_bigvgan"
    output_audio_dir = "subset_generated_audio_bigvgan"
    bigvgan_config_path = "/home/monesh/bigvgan_model/config.json"
    bigvgan_ckpt_path = "/home/monesh/bigvgan_model/bigvgan_generator.pt"

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_audio_dir, exist_ok=True)

    print(f"[+] Creating 10-sample subset from {train_csv_path}...")
    df = pd.read_csv(train_csv_path)
    df_subset = df.head(10)
    df_subset.to_csv(subset_csv_path, index=False)
    print(f"[+] Saved {len(df_subset)} samples to {subset_csv_path}")

    # 2. Setup DataModule
    datamodule = TTSDataModule(
        train_file=subset_csv_path,
        val_file=subset_csv_path,
        batch_size=2,
        num_workers=2,
        prefetch_data=False,
        stats_path=stats_path,
    )

    # 3. Setup Model & Trainer
    max_epochs = 700
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best_model_{epoch:03d}_{train_loss:.4f}",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_last=False
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        precision="bf16-mixed",
        callbacks=[checkpoint_callback, ModelSummary(max_depth=-1)],
        enable_progress_bar=True,
        log_every_n_steps=1,
        val_check_interval=1.0,
    )

    # Initialize Model
    model = TTSMODEL(lr=5e-4, max_epochs=max_epochs)

    # 4. Train
    print(f"\n[+] Starting training on {len(df_subset)} samples for {max_epochs} epochs...")
    start_time = time.time()
    trainer.fit(model, datamodule=datamodule)
    elapsed = time.time() - start_time
    print(f"[+] Training completed in {elapsed:.1f} seconds.")

    best_ckpt = checkpoint_callback.best_model_path
    if not best_ckpt or not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(ckpt_dir, "best_model_posenc.ckpt")
        trainer.save_checkpoint(best_ckpt)
    else:
        canonical_ckpt = os.path.join(ckpt_dir, "best_model_posenc.ckpt")
        import shutil
        shutil.copyfile(best_ckpt, canonical_ckpt)
        print(f"[+] Canonical checkpoint saved to: {canonical_ckpt}")

    print(f"\n[+] Best checkpoint saved to: {best_ckpt}")

    final_loss = trainer.callback_metrics.get("train_loss", "N/A")
    final_cfm = trainer.callback_metrics.get("train_loss_cfm", "N/A")
    final_dur = trainer.callback_metrics.get("train_loss_dur", "N/A")
    print(f"\n" + "="*50)
    print("=== FINAL TRAINING LOSSES ===")
    print("="*50)
    print(f"  Total Train Loss:    {final_loss}")
    print(f"  CFM (Flow) Loss:     {final_cfm}")
    print(f"  Duration Loss:       {final_dur}")
    print("="*50 + "\n")

    # 5. Load Trained Model for Inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Loading trained model from {best_ckpt} on device {device}...")
    eval_model = TTSMODEL.load_from_checkpoint(best_ckpt).model
    eval_model.eval()
    eval_model.to(device)

    # 6. Load BigVGAN Vocoder
    print(f"[+] Loading BigVGAN vocoder from {bigvgan_config_path} and {bigvgan_ckpt_path}...")
    with open(bigvgan_config_path) as f:
        h = AttrDict(json.load(f))
    bigvgan_vocoder = BigVGAN(h)
    bigvgan_state = torch.load(bigvgan_ckpt_path, map_location="cpu")
    bigvgan_vocoder.load_state_dict(bigvgan_state["generator"] if "generator" in bigvgan_state else bigvgan_state)
    bigvgan_vocoder.remove_weight_norm()
    bigvgan_vocoder.eval()
    bigvgan_vocoder.to(device)
    print("[+] BigVGAN vocoder successfully loaded!")

    # Load Normalization Stats
    stats_mean, stats_std = load_normalization_stats(stats_path, dim=100)
    mean_t = torch.tensor(stats_mean, device=device, dtype=torch.float32)
    std_t = torch.tensor(stats_std, device=device, dtype=torch.float32)

    def denorm(mel):
        return mel * std_t + mean_t

    # 7. Generation, Numerical Testing, and Audio Synthesis
    temperature = 0.66
    length_scale = 1.0

    print(f"\n[+] Generating Audio & Computing Numerical Metrics (T={temperature}, scale={length_scale})...")
    sample_metrics = []

    for idx, row in df_subset.iterrows():
        audio_id = row.get("audio_id", f"sample_{idx+1}")
        text = str(row.get("normalized_text", row.get("text")))
        mel_path = row["mel_path"]

        # 1. Load Ground Truth Mel
        raw_mel = np.squeeze(np.load(mel_path))
        if raw_mel.ndim == 2 and raw_mel.shape[0] == 100 and raw_mel.shape[1] != 100:
            raw_mel = raw_mel.T # ensure shape is (T, 100)
        norm_mel = (raw_mel.astype(np.float32) - stats_mean) / (stats_std + 1e-8)
        gt_mel_tensor = torch.tensor(norm_mel, dtype=torch.float32).unsqueeze(0).to(device)

        # 2. Text input
        ipa_seq = text_to_sequence(text, ["english_cleaners2"])[0]
        text_tensor = torch.tensor(ipa_seq, dtype=torch.long, device=device).unsqueeze(0)
        text_mask = torch.ones((1, text_tensor.size(1)), dtype=torch.bool, device=device)

        # 3. Model Inference (Predicted Mel)
        with torch.no_grad():
            pred_mel_norm, _ = eval_model(
                x=text_tensor,
                target_latent=None,
                mask=text_mask,
                n_timesteps=30,
                temperature=temperature,
                length_scale=length_scale,
                solver="euler"
            )

        # 4. Numerical Stability & Mel Metrics
        stab_metrics = compute_numerical_stability(pred_mel_norm, gt_mel_tensor)

        # 5. Denormalize Mel for BigVGAN: shape needs to be (1, 100, T)
        pred_mel_denorm = denorm(pred_mel_norm).transpose(1, 2)
        gt_mel_denorm = denorm(gt_mel_tensor).transpose(1, 2)

        # 6. Vocode with BigVGAN
        with torch.no_grad():
            pred_wav = bigvgan_vocoder(pred_mel_denorm).squeeze().cpu().numpy()
            gt_wav = bigvgan_vocoder(gt_mel_denorm).squeeze().cpu().numpy()

        # Clip waveform to [-1, 1]
        pred_wav = np.clip(pred_wav, -1.0, 1.0)
        gt_wav = np.clip(gt_wav, -1.0, 1.0)

        # 7. Save Generated and GT Waveforms using scipy.io.wavfile.write
        pred_audio_path = os.path.join(output_audio_dir, f"{idx+1:02d}_{audio_id}_pred.wav")
        gt_audio_path = os.path.join(output_audio_dir, f"{idx+1:02d}_{audio_id}_gt.wav")
        wavfile.write(pred_audio_path, 24000, (pred_wav * 32767).astype(np.int16))
        wavfile.write(gt_audio_path, 24000, (gt_wav * 32767).astype(np.int16))

        # 8. Audio Quality Tests (PESQ, STFT distance, Audio Cosine, Audio Pearson)
        pred_wav_t = torch.tensor(pred_wav).unsqueeze(0)
        gt_wav_t = torch.tensor(gt_wav).unsqueeze(0)

        pred_16k = TAF.resample(pred_wav_t, 24000, 16000).squeeze().numpy()
        gt_16k = TAF.resample(gt_wav_t, 24000, 16000).squeeze().numpy()

        try:
            pesq_val = float(pesq(16000, gt_16k, pred_16k, 'wb'))
        except Exception:
            pesq_val = 0.0

        # STFT distance
        window = torch.hann_window(1024)
        stft_gt = torch.stft(gt_wav_t, n_fft=1024, hop_length=256, window=window, return_complex=True)
        stft_pred = torch.stft(pred_wav_t, n_fft=1024, hop_length=256, window=window, return_complex=True)
        mag_gt = torch.abs(stft_gt)
        mag_pred = torch.abs(stft_pred)
        min_frames = min(mag_gt.size(-1), mag_pred.size(-1))
        stft_dist = float(torch.mean(torch.abs(mag_gt[..., :min_frames] - mag_pred[..., :min_frames])).item())

        # Waveform correlation
        min_samples = min(len(pred_wav), len(gt_wav))
        gw_clip = gt_wav[:min_samples]
        pw_clip = pred_wav[:min_samples]
        audio_cos = float(np.dot(gw_clip, pw_clip) / (np.linalg.norm(gw_clip) * np.linalg.norm(pw_clip) + 1e-8))
        if np.std(gw_clip) > 1e-8 and np.std(pw_clip) > 1e-8:
            audio_pearson = float(np.corrcoef(gw_clip, pw_clip)[0, 1])
        else:
            audio_pearson = 0.0

        # 9. MAS Alignment comparison (Ceiling benchmark with GT durations)
        try:
            with torch.no_grad():
                _, mu_mas, _, _, _ = eval_model(
                    x=text_tensor,
                    target_latent=gt_mel_tensor,
                    mask=text_mask,
                )
                mel_mas = eval_model.decoder(
                    mu=mu_mas,
                    mask=None,
                    target=None,
                    n_timesteps=30,
                    temperature=temperature,
                    solver="euler",
                )
                mas_mel_denorm = denorm(mel_mas).transpose(1, 2)
                mas_wav = bigvgan_vocoder(mas_mel_denorm).squeeze().cpu().numpy()
                mas_wav = np.clip(mas_wav, -1.0, 1.0)
                mas_audio_path = os.path.join(output_audio_dir, f"{idx+1:02d}_{audio_id}_mas.wav")
                wavfile.write(mas_audio_path, 24000, (mas_wav * 32767).astype(np.int16))

                mas_wav_t = torch.tensor(mas_wav).unsqueeze(0)
                mas_16k = TAF.resample(mas_wav_t, 24000, 16000).squeeze().numpy()
                mas_pesq = float(pesq(16000, gt_16k, mas_16k, 'wb'))
        except Exception as e:
            mas_pesq = 0.0
            mas_audio_path = None

        sample_res = {
            "index": idx + 1,
            "audio_id": audio_id,
            "pred_audio_path": pred_audio_path,
            "gt_audio_path": gt_audio_path,
            "mas_audio_path": mas_audio_path,
            "mel_frames_pred": int(pred_mel_norm.size(1)),
            "mel_frames_gt": int(gt_mel_tensor.size(1)),
            "duration_ratio": float(pred_mel_norm.size(1) / gt_mel_tensor.size(1)),
            "stability_score": stab_metrics["stability_score"],
            "variance_ratio": stab_metrics["variance_ratio"],
            "extreme_outliers": stab_metrics["extreme_outliers"],
            "mae_norm": stab_metrics["mae_norm"],
            "mse_norm": stab_metrics["mse_norm"],
            "mel_cosine": stab_metrics["cosine_similarity"],
            "mel_pearson": stab_metrics["pearson_correlation"],
            "audio_pesq": pesq_val,
            "audio_stft_dist": stft_dist,
            "audio_cosine": audio_cos,
            "audio_pearson": audio_pearson,
            "mas_pesq": mas_pesq,
        }
        sample_metrics.append(sample_res)

        print(
            f"Sample {idx+1:02d} [{audio_id}]: "
            f"Stab={sample_res['stability_score']:.1f}% | "
            f"VarRatio={sample_res['variance_ratio']:.2f} | "
            f"MelMAE={sample_res['mae_norm']:.3f} | "
            f"MelPearson={sample_res['mel_pearson']:.3f} | "
            f"PESQ={pesq_val:.2f} (MAS={mas_pesq:.2f}) | "
            f"STFT_Dist={stft_dist:.3f} | "
            f"Saved: {pred_audio_path}"
        )

    # 10. Compute Aggregates
    summary = {
        "avg_stability_score": float(np.mean([s["stability_score"] for s in sample_metrics])),
        "avg_variance_ratio": float(np.mean([s["variance_ratio"] for s in sample_metrics])),
        "avg_mel_mae": float(np.mean([s["mae_norm"] for s in sample_metrics])),
        "avg_mel_mse": float(np.mean([s["mse_norm"] for s in sample_metrics])),
        "avg_mel_cosine": float(np.mean([s["mel_cosine"] for s in sample_metrics])),
        "avg_mel_pearson": float(np.mean([s["mel_pearson"] for s in sample_metrics])),
        "avg_audio_pesq": float(np.mean([s["audio_pesq"] for s in sample_metrics])),
        "avg_mas_pesq": float(np.mean([s["mas_pesq"] for s in sample_metrics])),
        "avg_audio_stft_dist": float(np.mean([s["audio_stft_dist"] for s in sample_metrics])),
        "avg_audio_cosine": float(np.mean([s["audio_cosine"] for s in sample_metrics])),
        "avg_audio_pearson": float(np.mean([s["audio_pearson"] for s in sample_metrics])),
        "avg_duration_ratio": float(np.mean([s["duration_ratio"] for s in sample_metrics])),
    }

    print("\n" + "="*55)
    print("=== SUMMARY OF NUMERICAL & QUALITY RESULTS (BIGVGAN) ===")
    print("="*55)
    print(f"  Numerical Stability Score: {summary['avg_stability_score']:.1f}%")
    print(f"  Variance Ratio (pred/tgt): {summary['avg_variance_ratio']:.3f}")
    print(f"  Mel MAE (Normalized):      {summary['avg_mel_mae']:.4f}")
    print(f"  Mel MSE (Normalized):      {summary['avg_mel_mse']:.4f}")
    print(f"  Mel Cosine Similarity:     {summary['avg_mel_cosine']:.4f}")
    print(f"  Mel Pearson Correlation:   {summary['avg_mel_pearson']:.4f}")
    print(f"  Autonomous Audio PESQ:     {summary['avg_audio_pesq']:.3f}")
    print(f"  MAS Audio PESQ (Ceiling):  {summary['avg_mas_pesq']:.3f}")
    print(f"  Audio STFT Distance:       {summary['avg_audio_stft_dist']:.4f}")
    print(f"  Audio Cosine Similarity:   {summary['avg_audio_cosine']:.4f}")
    print(f"  Audio Pearson Correlation: {summary['avg_audio_pearson']:.4f}")
    print(f"  Avg Duration Ratio:        {summary['avg_duration_ratio']:.2f}")
    print("="*55)

    # Save to JSON
    results_payload = {
        "best_checkpoint": best_ckpt,
        "canonical_checkpoint": os.path.join(ckpt_dir, "best_model_posenc.ckpt"),
        "final_loss": str(final_loss),
        "final_cfm_loss": str(final_cfm),
        "final_dur_loss": str(final_dur),
        "summary": summary,
        "samples": sample_metrics,
    }
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_bigvgan_subset.json")
    with open(json_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[+] Full metrics saved to {json_path}")
    print(f"[+] All audio files saved in {output_audio_dir}/\n")


if __name__ == "__main__":
    main()
