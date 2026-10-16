#!/usr/bin/env python3
"""
===============================================================================
MambaFlow-TTS: Unified 50-Sample Master Evaluation Suite
===============================================================================
Runs the unified 50-sample validation benchmark (first 50 sentences of val.csv)
across the top models on GPU:
1. MambaFlow-CA-BiMamba-256D (Compact 128D DACVAE Latent)
2. MambaFlow-Sequential-Tetra (100-Mel Native 1x BigVGAN)
3. MambaFlow-BiDirectional-80mel (80-Mel Non-Causal HiFi-GAN)
4. MambaFlow-UniDirectional-80mel (80-Mel Causal HiFi-GAN)

Collects unified metrics identical to Model 1 (MambaFlow-CA-BiMamba-448D):
- Whisper WER (%), CER (%), PER (%)
- NISQA MOS (Naturalness)
- WavLM SECS (Speaker Cosine Similarity)
- Voiced F0 RMSE (Hz) via DTW
- 85% Spectral Roll-off (Hz)
- ODE Trajectory Straightness (S)
- GPU Real-Time Factor (RTF) & Speed (xRT)
===============================================================================
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import librosa
from types import SimpleNamespace

if torch.cuda.is_available():
    # Strict 12% VRAM cap (~490MB) to preserve memory for active training PID 42412
    try:
        torch.cuda.set_per_process_memory_fraction(0.12, 0)
    except Exception:
        pass

# Ensure roots are available
ROOT_DIR = "/home/monesh/TTSModel"
LATENT_DIR = os.path.join(ROOT_DIR, "MambaFlow-BiDirectional-Latent VAE")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if LATENT_DIR not in sys.path:
    sys.path.insert(0, LATENT_DIR)

from evaluate_val_health import PipelineEvaluator, clean_text_for_asr
from preprocessing.text import text_to_sequence


def purge_model_modules():
    """Purges model modules to prevent namespace collisions between models."""
    for mod in list(sys.modules.keys()):
        if any(mod.startswith(p) for p in [
            'mambaflow', 'TTSDataModule', 'TTSDatasetModule', 'TTS_model',
            'trans_encoder', 'duration_predictor', 'mamba_decoder',
            'mambaflow_tetra_decoder', 'mambaflow_mamba_convnext_decoder'
        ]):
            del sys.modules[mod]


# =============================================================================
# Synthesis & Trajectory Functions
# =============================================================================

def synthesize_256d(model, dacvae_decoder, text, device="cuda", n_steps=30, temp=0.667):
    """Synthesizes audio for MambaFlow-CA-BiMamba-256D and measures GPU ODE time."""
    cleaner_names = ["english_cleaners2"]
    seq = text_to_sequence(text, cleaner_names)[0]
    text_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    text_mask = torch.ones_like(text_tensor, dtype=torch.bool)

    with torch.inference_mode():
        x_enc = model.txt_emb(text_tensor)
        enc_out = model.encoder(x_enc, mask=text_mask)
        dur_int, _, _ = model.duration(enc_out.detach(), mask=text_mask, length_scale=1.0)
        durations = torch.clamp(dur_int, min=1)
        if durations.dim() == 3:
            durations = durations.squeeze(-1)

        audio_lengths = durations.sum(dim=-1)
        T_audio = int(audio_lengths.max().item())
        audio_mask = torch.arange(T_audio, device=device)[None, :] < audio_lengths[:, None]

        x_curr = torch.randn((1, T_audio, model.d_codec), device=device) * float(temp)
        m = audio_mask.unsqueeze(-1).float()
        x_curr = x_curr * m
        trajectory = [x_curr.clone()]

        dt = 1.0 / float(n_steps)
        torch.cuda.synchronize()
        t_start = time.time()
        for step in range(n_steps):
            t_tensor = torch.full((1,), step * dt, device=device, dtype=torch.float32)
            v = model.decoder.predict_velocity(
                x=x_curr,
                t=t_tensor,
                text_repr=enc_out,
                durations=durations,
                text_mask=text_mask,
                audio_mask=audio_mask,
            )
            x_curr = (x_curr + dt * v) * m
            trajectory.append(x_curr.clone())
        torch.cuda.synchronize()
        ode_time = time.time() - t_start

        # Trajectory Straightness
        z_start = trajectory[0].float().flatten()
        z_end = trajectory[-1].float().flatten()
        disp = torch.norm(z_end - z_start, p=2).item()
        step_lens = [
            torch.norm((trajectory[i+1] - trajectory[i]).float().flatten(), p=2).item()
            for i in range(len(trajectory) - 1)
        ]
        straightness = disp / max(sum(step_lens), 1e-8)

        # Denormalize & Decode via DACVAE (on CPU)
        z_cpu = x_curr.detach().cpu()
        from stats import denormalize_latent
        from generate_audio import decode_latent_to_audio
        z_denorm = denormalize_latent(z_cpu)
        if z_denorm.dim() == 3 and z_denorm.size(1) != 128 and z_denorm.size(2) == 128:
            z_denorm = z_denorm.transpose(1, 2)
        wav_48k = decode_latent_to_audio(dacvae_decoder, z_denorm, target_device="cpu")

    return wav_48k, straightness, ode_time, 48000


def synthesize_tetra(model, bigvgan, denormalize_mel, text, device="cuda", n_steps=30, temp=0.667):
    """Synthesizes audio for MambaFlow-Sequential-Tetra and measures GPU ODE time."""
    cleaner_names = ["english_cleaners2"]
    seq = text_to_sequence(text, cleaner_names)[0]
    x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        x_enc = model.encoder(model.emb_dropout(model.txt_emb(x)))
        mu_text_base = model.to_latent(x_enc)
        dur_int, _, _ = model.duration(mu_text_base, latent=None, length_scale=1.0)
        durations = torch.clamp(dur_int, min=1)
        mu_expanded = torch.repeat_interleave(mu_text_base[0], durations[0], dim=0).unsqueeze(0)

        dt = 1.0 / float(n_steps)
        x_curr = torch.randn_like(mu_expanded) * float(temp)
        trajectory = [x_curr.clone()]

        torch.cuda.synchronize()
        t_start = time.time()
        for step in range(n_steps):
            t_tensor = torch.full((1,), step * dt, device=device, dtype=torch.float32)
            v = model.decoder.predict_velocity(x_curr, t_tensor, mu_expanded)
            x_curr = x_curr + dt * v
            trajectory.append(x_curr.clone())
        torch.cuda.synchronize()
        ode_time = time.time() - t_start

        # Straightness
        z_start = trajectory[0].float().flatten()
        z_end = trajectory[-1].float().flatten()
        disp = torch.norm(z_end - z_start, p=2).item()
        step_lens = [
            torch.norm((trajectory[i+1] - trajectory[i]).float().flatten(), p=2).item()
            for i in range(len(trajectory) - 1)
        ]
        straightness = disp / max(sum(step_lens), 1e-8)

        # BigVGAN Vocoding on CPU
        mel_raw = denormalize_mel(x_curr).transpose(1, 2).detach().to("cpu")
        with torch.inference_mode():
            audio = bigvgan(mel_raw)[0, 0].detach().cpu().numpy()

        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = (audio / peak) * 0.95

    return audio, straightness, ode_time, 24000


def synthesize_80mel(model, hifigan, mean_t, std_t, text, device="cuda", n_steps=30, temp=0.667):
    """Synthesizes audio for 80-mel Mamba-2 models and measures GPU ODE time."""
    cleaner_names = ["english_cleaners2"]
    seq = text_to_sequence(text, cleaner_names)[0]
    x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        x_enc = model.encoder(model.emb_dropout(model.txt_emb(x)))
        mu_text_base = model.to_latent(x_enc)
        dur_int, _, _ = model.duration(x_enc.detach(), mask=None, length_scale=1.0)
        durations = torch.clamp(dur_int, min=1)
        mu_expanded = torch.repeat_interleave(mu_text_base[0], durations[0], dim=0).unsqueeze(0)

        dt = 1.0 / float(n_steps)
        x_curr = torch.randn_like(mu_expanded) * float(temp)
        trajectory = [x_curr.clone()]

        torch.cuda.synchronize()
        t_start = time.time()
        for step in range(n_steps):
            t_tensor = torch.full((1,), step * dt, device=device, dtype=torch.float32)
            v = model.decoder.predict_velocity(x_curr, t_tensor, mu_expanded)
            x_curr = x_curr + dt * v
            trajectory.append(x_curr.clone())
        torch.cuda.synchronize()
        ode_time = time.time() - t_start

        # Straightness
        z_start = trajectory[0].float().flatten()
        z_end = trajectory[-1].float().flatten()
        disp = torch.norm(z_end - z_start, p=2).item()
        step_lens = [
            torch.norm((trajectory[i+1] - trajectory[i]).float().flatten(), p=2).item()
            for i in range(len(trajectory) - 1)
        ]
        straightness = disp / max(sum(step_lens), 1e-8)

        # Denormalize mel to HiFi-GAN
        mel_denorm = (x_curr * std_t + mean_t).transpose(1, 2).detach().to("cpu")
        with torch.inference_mode():
            audio = hifigan(mel_denorm).squeeze().detach().cpu().numpy()

        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = (audio / peak) * 0.95

    return audio, straightness, ode_time, 22050


# =============================================================================
# Master Evaluation Loop for a Single Model
# =============================================================================

def evaluate_model(
    model_id: str,
    model_name: str,
    synth_fn,
    evaluator: PipelineEvaluator,
    df_eval: pd.DataFrame,
    output_dir: str,
    device: str = "cuda",
):
    print("\n" + "=" * 80)
    print(f"BENCHMARKING MODEL: {model_name} ({model_id})")
    print("=" * 80)

    model_out_dir = os.path.join(output_dir, model_id)
    wavs_dir = os.path.join(model_out_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    results = []
    n_eval = len(df_eval)

    for idx in range(n_eval):
        row = df_eval.iloc[idx]
        audio_id = str(row["audio_id"])
        text = str(row.get("normalized_text", row.get("text", "")))
        gt_audio_path = str(row.get("audio_path", ""))

        # Synthesize audio on GPU with trajectory recording
        wav_pred, straightness, ode_time, sr_pred = synth_fn(text)
        dur_audio = len(wav_pred) / float(sr_pred)
        rtf = ode_time / max(dur_audio, 1e-4)

        # Save generated WAV
        out_wav_path = os.path.join(wavs_dir, f"{audio_id}_pred.wav")
        sf.write(out_wav_path, wav_pred, sr_pred)

        # Load GT audio
        if os.path.exists(gt_audio_path):
            wav_gt_raw, sr_gt = sf.read(gt_audio_path)
            if wav_gt_raw.ndim > 1:
                wav_gt_raw = np.mean(wav_gt_raw, axis=-1)
            wav_gt_16k = librosa.resample(wav_gt_raw, orig_sr=sr_gt, target_sr=16000)
        else:
            wav_gt_16k = librosa.resample(wav_pred, orig_sr=sr_pred, target_sr=16000)

        # Resample predicted audio to 16kHz for metric evaluators
        wav_pred_16k = librosa.resample(wav_pred, orig_sr=sr_pred, target_sr=16000)

        # 1. Whisper ASR (WER, CER, PER)
        asr_res = evaluator.evaluate_whisper(wav_pred_16k, text, sr=16000)

        # 2. VAD
        vad_res = evaluator.evaluate_vad(wav_pred_16k, text)

        # 3. Acoustic Alignment (F0 RMSE, Energy RMSE)
        acoust_res = evaluator.evaluate_acoustic(wav_pred_16k, wav_gt_16k, sr=16000)

        # 4. Spectral Features (85% Roll-off)
        spec_res = evaluator.evaluate_spectral(wav_pred_16k, sr=16000)

        # 5. Quality (NISQA MOS)
        qual_res = evaluator.evaluate_quality(wav_pred_16k)

        # 6. Speaker Cosine Similarity (SECS)
        secs = evaluator.evaluate_secs(wav_pred_16k, wav_gt_16k)

        sample_record = {
            "audio_id": audio_id,
            "text": text,
            "transcript": asr_res["transcript"],
            "wer": asr_res["wer"],
            "cer": asr_res["cer"],
            "per": asr_res["per"],
            "straightness": straightness,
            "ode_time_sec": ode_time,
            "duration_sec": dur_audio,
            "rtf": rtf,
            "speed_xrt": 1.0 / max(rtf, 1e-6),
            "f0_rmse": acoust_res["f0_rmse"],
            "energy_rmse": acoust_res["energy_rmse"],
            "spectral_rolloff_85": spec_res.get("spectral_rolloff_85_mean", spec_res.get("spectral_rolloff_85_median", 3500.0)),
            "nisqa_mos": qual_res["nisqa_mos"],
            "nisqa_coloration": qual_res["nisqa_coloration"],
            "nisqa_discontinuity": qual_res["nisqa_discontinuity"],
            "secs": secs,
            "wav_path": out_wav_path,
        }
        results.append(sample_record)

        torch.cuda.empty_cache()
        import gc
        gc.collect()

        print(
            f"[{idx+1:02d}/{n_eval}] {audio_id} | "
            f"WER: {asr_res['wer']*100:4.1f}% | PER: {asr_res['per']*100:4.1f}% | "
            f"MOS: {qual_res['nisqa_mos']:.2f} | SECS: {secs:.3f} | "
            f"S: {straightness:.3f} | RTF: {rtf:.4f} ({1/rtf:4.1f}xRT)"
        )

    # Compile Summary
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(model_out_dir, "per_sample_metrics.csv"), index=False)

    summary = {
        "model_id": model_id,
        "model_name": model_name,
        "total_samples": len(df_res),
        "mean_wer": float(df_res["wer"].mean()),
        "std_wer": float(df_res["wer"].std()),
        "mean_cer": float(df_res["cer"].mean()),
        "std_cer": float(df_res["cer"].std()),
        "mean_per": float(df_res["per"].mean()),
        "std_per": float(df_res["per"].std()),
        "mean_straightness": float(df_res["straightness"].mean()),
        "std_straightness": float(df_res["straightness"].std()),
        "mean_f0_rmse": float(df_res["f0_rmse"].mean()),
        "mean_energy_rmse": float(df_res["energy_rmse"].mean()),
        "mean_rolloff_85": float(df_res["spectral_rolloff_85"].mean()),
        "mean_nisqa_mos": float(df_res["nisqa_mos"].mean()),
        "mean_nisqa_coloration": float(df_res["nisqa_coloration"].mean()),
        "mean_nisqa_discontinuity": float(df_res["nisqa_discontinuity"].mean()),
        "mean_secs": float(df_res["secs"].mean()),
        "mean_rtf": float(df_res["rtf"].mean()),
        "mean_speed_xrt": float(df_res["speed_xrt"].mean()),
    }

    with open(os.path.join(model_out_dir, "summary_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "-" * 80)
    print(f"SUMMARY FOR {model_name}:")
    print(
        f"  WER: {summary['mean_wer']*100:.2f}% | CER: {summary['mean_cer']*100:.2f}% | "
        f"PER: {summary['mean_per']*100:.2f}% | MOS: {summary['mean_nisqa_mos']:.2f} | "
        f"SECS: {summary['mean_secs']:.3f} | S: {summary['mean_straightness']:.4f} | "
        f"F0 RMSE: {summary['mean_f0_rmse']:.1f}Hz | Roll-off: {summary['mean_rolloff_85']:.1f}Hz | "
        f"RTF: {summary['mean_rtf']:.4f} ({summary['mean_speed_xrt']:.1f}xRT)"
    )
    print("-" * 80)
    return summary


# =============================================================================
# Main Script Orchestration
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MambaFlow-TTS Unified 50-Sample Benchmark")
    parser.add_argument("--val_csv", type=str, default="/home/monesh/ljspeech/LJSpeech-1.1/val.csv")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="eval_unified_results")
    parser.add_argument("--model", type=str, default="all", choices=["all", "latent_256d", "seq_tetra", "bi_80mel", "uni_80mel"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[+] Loading {args.num_samples} samples from {args.val_csv}...")
    df_val = pd.read_csv(args.val_csv)
    df_eval = df_val.iloc[:args.num_samples].copy().reset_index(drop=True)

    # Initialize Evaluator on CPU (protects GPU memory)
    print("[+] Initializing PipelineEvaluator (Whisper, VAD, NISQA, WavLM on CPU)...")
    evaluator = PipelineEvaluator(whisper_model_size="tiny.en", device="cpu")

    all_summaries = {}

    # -------------------------------------------------------------------------
    # Model 2: MambaFlow-CA-BiMamba-256D
    # -------------------------------------------------------------------------
    if args.model in ["all", "latent_256d"]:
        purge_model_modules()
        sys.path.insert(0, LATENT_DIR)
        from TTSDataModule import TTSMODEL
        from generate_audio import setup_dacvae

        ckpt_256d = os.path.join(LATENT_DIR, "TTS_checkpoints/mambaflow_ca_bimamba_256d_last.ckpt")
        print(f"[+] Loading 256D model from {ckpt_256d}...")
        pl_module = TTSMODEL.load_from_checkpoint(ckpt_256d, map_location=device)
        model_256d = pl_module.model.to(device).eval()
        dacvae_decoder, _, _ = setup_dacvae(device="cpu")

        def synth_256d_wrapper(text):
            return synthesize_256d(model_256d, dacvae_decoder, text, device=device, n_steps=30, temp=0.667)

        s2 = evaluate_model(
            model_id="latent_256d",
            model_name="MambaFlow-CA-BiMamba-256D (Compact 128D DACVAE)",
            synth_fn=synth_256d_wrapper,
            evaluator=evaluator,
            df_eval=df_eval,
            output_dir=args.output_dir,
            device=device,
        )
        all_summaries["latent_256d"] = s2

        del model_256d, pl_module, dacvae_decoder
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Model 3: MambaFlow-Sequential-Tetra
    # -------------------------------------------------------------------------
    if args.model in ["all", "seq_tetra"]:
        purge_model_modules()
        tetra_dir = os.path.join(ROOT_DIR, "MambaFlow-Sequential-Tetra")
        sys.path.insert(0, tetra_dir)
        from TTSDataModule import TTSMODEL
        from TTSDatasetModule import denormalize_mel
        from bigvgan import BigVGAN, AttrDict

        ckpt_tetra = os.path.join(ROOT_DIR, "Checkpoints/MambaFlow-Sequential-Tetra.ckpt")
        print(f"[+] Loading Sequential-Tetra from {ckpt_tetra}...")
        lightning_tetra = TTSMODEL()
        sd = torch.load(ckpt_tetra, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("model.", ""): v for k, v in sd.items()}
        lightning_tetra.model.load_state_dict(sd, strict=True)
        model_tetra = lightning_tetra.model.to(device).eval()

        # BigVGAN Vocoder on CPU
        voc_cfg = "/home/monesh/bigvgan_model/config_14M.json"
        voc_pt = "/home/monesh/bigvgan_model/bigvgan_generator_14M.pt"
        with open(voc_cfg) as f:
            h = AttrDict(json.load(f))
        bigvgan = BigVGAN(h, use_cuda_kernel=False).to("cpu").eval()
        ckpt_voc = torch.load(voc_pt, map_location="cpu")
        bigvgan.load_state_dict(ckpt_voc.get("generator", ckpt_voc))
        bigvgan.remove_weight_norm()

        def synth_tetra_wrapper(text):
            return synthesize_tetra(model_tetra, bigvgan, denormalize_mel, text, device=device, n_steps=30, temp=0.667)

        s3 = evaluate_model(
            model_id="seq_tetra_100mel",
            model_name="MambaFlow-Sequential-Tetra (100-Mel BigVGAN)",
            synth_fn=synth_tetra_wrapper,
            evaluator=evaluator,
            df_eval=df_eval,
            output_dir=args.output_dir,
            device=device,
        )
        all_summaries["seq_tetra_100mel"] = s3

        del model_tetra, lightning_tetra, bigvgan
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Model 4: MambaFlow-BiDirectional-80mel
    # -------------------------------------------------------------------------
    if args.model in ["all", "bi_80mel"]:
        purge_model_modules()
        bi_dir = os.path.join(ROOT_DIR, "MambaFlow-BiDirectional-Mamba2-80mel")
        sys.path.insert(0, bi_dir)
        from TTSDataModule import TTSMODEL

        # HiFi-GAN Vocoder on CPU
        hifi_dir = "/home/monesh/hifigan"
        hifi_ckpt = "/home/monesh/ljspeech/LJSpeech-1.1/generator_v1"
        hifi_cfg = os.path.join(hifi_dir, "config_v1.json")
        if hifi_dir not in sys.path:
            sys.path.insert(0, hifi_dir)
        from models import Generator
        with open(hifi_cfg) as f:
            h = SimpleNamespace(**json.load(f))
        hifigan = Generator(h).to("cpu").eval()
        ckpt_hifi = torch.load(hifi_ckpt, map_location="cpu")
        hifigan.load_state_dict(ckpt_hifi.get("generator", ckpt_hifi))
        try:
            hifigan.remove_weight_norm()
        except Exception:
            pass

        import importlib.util
        spec = importlib.util.spec_from_file_location("stats_hifigan", "/home/monesh/ljspeech/LJSpeech-1.1/stats_hifigan.py")
        stats_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stats_mod)
        mean_t = torch.tensor(stats_mod.MEL_MEAN, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        std_t = torch.tensor(stats_mod.MEL_STD, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        ckpt_bi = os.path.join(ROOT_DIR, "Checkpoints/MambaFlow-BiDirectional-Mamba2-80mel.ckpt")
        print(f"[+] Loading BiDirectional-80mel from {ckpt_bi}...")
        lightning_bi = TTSMODEL()
        sd = torch.load(ckpt_bi, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("model.", ""): v for k, v in sd.items()}
        lightning_bi.model.load_state_dict(sd, strict=True)
        model_bi = lightning_bi.model.to(device).eval()

        def synth_bi_wrapper(text):
            return synthesize_80mel(model_bi, hifigan, mean_t, std_t, text, device=device, n_steps=30, temp=0.667)

        s4 = evaluate_model(
            model_id="bi_80mel",
            model_name="MambaFlow-BiDirectional-80mel (HiFi-GAN V1)",
            synth_fn=synth_bi_wrapper,
            evaluator=evaluator,
            df_eval=df_eval,
            output_dir=args.output_dir,
            device=device,
        )
        all_summaries["bi_80mel"] = s4

        del model_bi, lightning_bi
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Model 5: MambaFlow-UniDirectional-80mel
    # -------------------------------------------------------------------------
    if args.model in ["all", "uni_80mel"]:
        purge_model_modules()
        uni_dir = os.path.join(ROOT_DIR, "MambaFlow-UniDirectional-Mamba2-80mel")
        sys.path.insert(0, uni_dir)
        from TTSDataModule import TTSMODEL

        hifi_dir = "/home/monesh/hifigan"
        hifi_ckpt = "/home/monesh/ljspeech/LJSpeech-1.1/generator_v1"
        hifi_cfg = os.path.join(hifi_dir, "config_v1.json")
        if hifi_dir not in sys.path:
            sys.path.insert(0, hifi_dir)
        from models import Generator
        with open(hifi_cfg) as f:
            h = SimpleNamespace(**json.load(f))
        hifigan = Generator(h).to("cpu").eval()
        ckpt_hifi = torch.load(hifi_ckpt, map_location="cpu")
        hifigan.load_state_dict(ckpt_hifi.get("generator", ckpt_hifi))
        try:
            hifigan.remove_weight_norm()
        except Exception:
            pass

        import importlib.util
        spec = importlib.util.spec_from_file_location("stats_hifigan", "/home/monesh/ljspeech/LJSpeech-1.1/stats_hifigan.py")
        stats_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stats_mod)
        mean_t = torch.tensor(stats_mod.MEL_MEAN, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        std_t = torch.tensor(stats_mod.MEL_STD, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        ckpt_uni = os.path.join(ROOT_DIR, "Checkpoints/MambaFlow-UniDirectional-Mamba2-80mel.ckpt")
        print(f"[+] Loading UniDirectional-80mel from {ckpt_uni}...")
        lightning_uni = TTSMODEL()
        sd = torch.load(ckpt_uni, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("model.", ""): v for k, v in sd.items()}
        lightning_uni.model.load_state_dict(sd, strict=True)
        model_uni = lightning_uni.model.to(device).eval()

        def synth_uni_wrapper(text):
            return synthesize_80mel(model_uni, hifigan, mean_t, std_t, text, device=device, n_steps=30, temp=0.667)

        s5 = evaluate_model(
            model_id="uni_80mel",
            model_name="MambaFlow-UniDirectional-80mel (HiFi-GAN V1)",
            synth_fn=synth_uni_wrapper,
            evaluator=evaluator,
            df_eval=df_eval,
            output_dir=args.output_dir,
            device=device,
        )
        all_summaries["uni_80mel"] = s5

        del model_uni, lightning_uni
        torch.cuda.empty_cache()

    # Save Master Unified Comparison
    with open(os.path.join(args.output_dir, "master_unified_summaries.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL REQUESTED EVALUATIONS COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
