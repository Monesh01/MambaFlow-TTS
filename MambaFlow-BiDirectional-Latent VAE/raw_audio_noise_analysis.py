"""
Raw Audio Noise & Generalization Analysis (Self-Contained)
===========================================================
Loads the MambaFlow model, generates both GT audio (from latent) and model-predicted
audio (via Euler integration), then compares them RAW — no mastering, no EQ, no
loudness normalization.

Metrics:
  - RMS Energy, Peak Amplitude, Crest Factor
  - Noise Floor, SNR estimate
  - Spectral Centroid, Bandwidth, Flatness
  - Log-Mel Spectral Distance
  - Energy Envelope Correlation
  - Whisper ASR + WER/CER
  - Cross-sample variance preservation

Output: plain markdown report (no LaTeX)
"""

import os
import sys
import time
import json
import ast
import re
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import scipy.signal
import torch
import torchaudio.transforms as T

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import inspect

def enable_mamba_cpu_fallback():
    try:
        import causal_conv1d
        import causal_conv1d.causal_conv1d_interface as cci
        import mamba_ssm.ops.selective_scan_interface as ssi
        import mamba_ssm.ops.triton.layernorm_gated as lng
        import mamba_ssm.ops.triton.ssd_combined as ssd
        import mamba_ssm.modules.mamba2 as m2
        causal_conv1d.causal_conv1d_fn = cci.causal_conv1d_ref
        cci.causal_conv1d_fn = cci.causal_conv1d_ref
        ssd.causal_conv1d_fn = cci.causal_conv1d_ref
        ssi.selective_scan_fn = ssi.selective_scan_ref
        lng.rmsnorm_fn = lng.rms_norm_ref
        ssd.rmsnorm_fn = lng.rms_norm_ref
        ref_keys = set(inspect.signature(ssd.mamba_split_conv1d_scan_ref).parameters.keys())
        def cpu_scan(*args, **kwargs):
            filtered = {k: v for k, v in kwargs.items() if k in ref_keys}
            return ssd.mamba_split_conv1d_scan_ref(*args, **filtered)
        m2.mamba_split_conv1d_scan_combined = cpu_scan
    except Exception:
        pass

if not torch.cuda.is_available():
    enable_mamba_cpu_fallback()

from TTSDataModule import TTSMODEL
from preprocessing.text import text_to_sequence
from stats import denormalize_latent, normalize_latent
from generate_audio import setup_dacvae, decode_latent_to_audio

# ─── Audio Analysis Utilities ─────────────────────────────────────────────────

def db(x):
    return 20.0 * np.log10(max(x, 1e-10))

def rms_energy(wav):
    return db(np.sqrt(np.mean(wav ** 2)))

def peak_amplitude(wav):
    return db(np.max(np.abs(wav)))

def noise_floor_estimate(wav, sr, frame_ms=25, percentile=5):
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len // 2
    n_frames = max(1, (len(wav) - frame_len) // hop + 1)
    frame_rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = wav[start : start + frame_len]
        frame_rms[i] = np.sqrt(np.mean(frame ** 2))
    threshold_idx = max(1, int(n_frames * percentile / 100))
    sorted_rms = np.sort(frame_rms)
    noise_rms = np.mean(sorted_rms[:threshold_idx])
    return db(noise_rms), noise_rms

def snr_estimate(wav, sr, frame_ms=25, percentile=5):
    signal_rms_val = np.sqrt(np.mean(wav ** 2))
    _, noise_rms_val = noise_floor_estimate(wav, sr, frame_ms, percentile)
    if noise_rms_val < 1e-10:
        return 999.0
    return 20.0 * np.log10(signal_rms_val / noise_rms_val)

def spectral_centroid_hz(wav, sr):
    sc = librosa.feature.spectral_centroid(y=wav, sr=sr, n_fft=1024, hop_length=256)
    return float(np.mean(sc))

def spectral_bandwidth_hz(wav, sr):
    sb = librosa.feature.spectral_bandwidth(y=wav, sr=sr, n_fft=1024, hop_length=256)
    return float(np.mean(sb))

def spectral_flatness_mean(wav):
    sf_val = librosa.feature.spectral_flatness(y=wav, n_fft=1024, hop_length=256)
    return float(np.mean(sf_val))

def crest_factor_db(wav):
    peak = np.max(np.abs(wav))
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < 1e-10:
        return 0.0
    return 20.0 * np.log10(peak / rms)

def log_mel_spectral_distance(wav1, wav2, sr, n_mels=80):
    mel1 = librosa.feature.melspectrogram(y=wav1, sr=sr, n_fft=1024, hop_length=256, n_mels=n_mels)
    mel2 = librosa.feature.melspectrogram(y=wav2, sr=sr, n_fft=1024, hop_length=256, n_mels=n_mels)
    log1 = np.log(np.maximum(mel1, 1e-5))
    log2 = np.log(np.maximum(mel2, 1e-5))
    min_t = min(log1.shape[1], log2.shape[1])
    diff = log1[:, :min_t] - log2[:, :min_t]
    return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff ** 2)))

def energy_envelope_correlation(wav1, wav2, sr, hop_ms=10):
    hop = int(sr * hop_ms / 1000)
    def envelope(w):
        n_frames = max(1, len(w) // hop)
        env = np.zeros(n_frames)
        for i in range(n_frames):
            frame = w[i * hop : (i + 1) * hop]
            env[i] = np.sqrt(np.mean(frame ** 2))
        return env
    env1 = envelope(wav1)
    env2 = envelope(wav2)
    min_len = min(len(env1), len(env2))
    env1, env2 = env1[:min_len], env2[:min_len]
    if np.std(env1) < 1e-8 or np.std(env2) < 1e-8:
        return 0.0
    return float(np.corrcoef(env1, env2)[0, 1])


# ─── Whisper ASR ──────────────────────────────────────────────────────────────

def calculate_wer(reference, hypothesis):
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    r_len, h_len = len(ref_words), len(hyp_words)
    d = np.zeros((r_len + 1, h_len + 1), dtype=int)
    for i in range(r_len + 1):
        d[i, 0] = i
    for j in range(h_len + 1):
        d[0, j] = j
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + 1)
    return float(d[r_len, h_len] / r_len)

def calculate_cer(reference, hypothesis):
    r = list(reference.lower().strip())
    h = list(hypothesis.lower().strip())
    if not r:
        return 0.0 if not h else 1.0
    r_len, h_len = len(r), len(h)
    d = np.zeros((r_len + 1, h_len + 1), dtype=int)
    for i in range(r_len + 1):
        d[i, 0] = i
    for j in range(h_len + 1):
        d[0, j] = j
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if r[i - 1] == h[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + 1)
    return float(d[r_len, h_len] / r_len)


# ─── Model Checkpoint Resolution ─────────────────────────────────────────────

def find_best_checkpoint(ckpt_dir="TTS_checkpoints"):
    if not os.path.exists(ckpt_dir):
        return None, None
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None, None
    best_ckpt, min_loss = None, float("inf")
    for c in ckpts:
        match = re.search(r"loss_([0-9]+\.[0-9]+)", os.path.basename(c))
        if match:
            loss = float(match.group(1))
            if loss < min_loss:
                min_loss = loss
                best_ckpt = c
    if best_ckpt is None:
        last = os.path.join(ckpt_dir, "mambaflow_ca_bimamba_256d_last.ckpt")
        if os.path.exists(last):
            return last, None
        ckpts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return ckpts[0], None
    return best_ckpt, min_loss


# ─── Generate GT + Pred Raw Audio for a Sample ───────────────────────────────

def generate_raw_audio_pair(model, dacvae_decoder, sample_row, sample_index,
                            device="cuda", dacvae_device="cpu", n_steps=30,
                            temperature=1.0, sr=48000):
    """
    Generate raw GT audio (from latent decode) and raw predicted audio (from
    Euler integration + latent decode). Returns raw waveforms — NO mastering.
    """
    audio_id = sample_row["audio_id"]
    text = str(sample_row.get("normalized_text", sample_row.get("text", "")))
    latent_path = sample_row["latent"]
    dur_list = ast.literal_eval(sample_row["durations"]) if isinstance(sample_row["durations"], str) else sample_row["durations"]

    print(f"\n{'='*80}")
    print(f"  [Sample {sample_index}] Audio ID: {audio_id}")
    print(f"  Text: \"{text}\"")
    print(f"  Frames: {sum(dur_list)} | Phonemes: {len(dur_list)}")
    print(f"{'='*80}")

    # Load Ground Truth Latent & Decode GT Audio
    gt_raw = np.squeeze(np.load(latent_path))
    if gt_raw.ndim == 2:
        if gt_raw.shape[0] != 128 and gt_raw.shape[1] == 128:
            gt_raw = gt_raw.T
    T_gt = gt_raw.shape[1]

    gt_tensor = torch.tensor(gt_raw, dtype=torch.float32).unsqueeze(0).to(device)
    gt_norm = normalize_latent(gt_tensor).transpose(1, 2)  # [1, T, 128]

    # Decode GT audio (raw, no processing)
    print(f"  [+] Decoding Ground Truth audio ({T_gt} frames)...")
    gt_wav = decode_latent_to_audio(dacvae_decoder, gt_tensor, target_device=dacvae_device)

    # Euler integration to get predicted latent
    cleaner_names = ["english_cleaners2"]
    seq = text_to_sequence(text, cleaner_names)[0]
    text_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    text_mask = torch.ones_like(text_tensor, dtype=torch.bool)
    durations = torch.tensor(dur_list, dtype=torch.long, device=device).unsqueeze(0)

    dur_sum = int(durations.sum().item())
    if dur_sum != T_gt:
        diff = T_gt - dur_sum
        durations[0, -1] = max(1, durations[0, -1] + diff)

    audio_lengths = durations.sum(dim=-1)
    audio_mask = torch.arange(T_gt, device=device)[None, :] < audio_lengths[:, None]

    dt = 1.0 / n_steps
    torch.manual_seed(100 + sample_index)
    x_0 = torch.randn_like(gt_norm) * float(temperature)
    x_curr = x_0.clone()

    print(f"  [+] Running {n_steps}-step Euler integration...")
    with torch.inference_mode():
        x_enc = model.txt_emb(text_tensor)
        enc_out = model.encoder(x_enc, mask=text_mask)

        for step in range(n_steps):
            t_val = step * dt
            t_tensor = torch.full((1,), t_val, device=device, dtype=torch.float32)
            v = model.decoder.predict_velocity(
                x=x_curr, t=t_tensor, text_repr=enc_out,
                durations=durations, text_mask=text_mask, audio_mask=audio_mask,
            )
            x_curr = x_curr + dt * v
            if audio_mask is not None:
                x_curr = x_curr * audio_mask.unsqueeze(-1).float()

    # Decode predicted latent to raw audio (NO mastering)
    x_1 = x_curr
    pred_latent_denorm = denormalize_latent(x_1.float())
    pred_latent_128_t = pred_latent_denorm.transpose(1, 2)
    print(f"  [+] Decoding Predicted audio...")
    pred_wav = decode_latent_to_audio(dacvae_decoder, pred_latent_128_t, target_device=dacvae_device)

    return gt_wav, pred_wav, audio_id, text


# ─── Analysis ─────────────────────────────────────────────────────────────────

def analyze_sample(gt_wav, pred_wav, sr, audio_id, text, whisper_results=None):
    """Compute all metrics for a single GT/Pred pair."""
    r = {"audio_id": audio_id, "text": text}
    r["duration_sec_gt"] = round(len(gt_wav) / sr, 3)
    r["duration_sec_pred"] = round(len(pred_wav) / sr, 3)

    # GT metrics
    r["gt_rms_db"] = round(rms_energy(gt_wav), 2)
    r["gt_peak_dbfs"] = round(peak_amplitude(gt_wav), 2)
    r["gt_noise_floor_db"], _ = noise_floor_estimate(gt_wav, sr)
    r["gt_noise_floor_db"] = round(r["gt_noise_floor_db"], 2)
    r["gt_snr_db"] = round(snr_estimate(gt_wav, sr), 2)
    r["gt_spectral_centroid_hz"] = round(spectral_centroid_hz(gt_wav, sr), 1)
    r["gt_spectral_bandwidth_hz"] = round(spectral_bandwidth_hz(gt_wav, sr), 1)
    r["gt_spectral_flatness"] = round(spectral_flatness_mean(gt_wav), 5)
    r["gt_crest_factor_db"] = round(crest_factor_db(gt_wav), 2)

    # Pred metrics
    r["pred_rms_db"] = round(rms_energy(pred_wav), 2)
    r["pred_peak_dbfs"] = round(peak_amplitude(pred_wav), 2)
    r["pred_noise_floor_db"], _ = noise_floor_estimate(pred_wav, sr)
    r["pred_noise_floor_db"] = round(r["pred_noise_floor_db"], 2)
    r["pred_snr_db"] = round(snr_estimate(pred_wav, sr), 2)
    r["pred_spectral_centroid_hz"] = round(spectral_centroid_hz(pred_wav, sr), 1)
    r["pred_spectral_bandwidth_hz"] = round(spectral_bandwidth_hz(pred_wav, sr), 1)
    r["pred_spectral_flatness"] = round(spectral_flatness_mean(pred_wav), 5)
    r["pred_crest_factor_db"] = round(crest_factor_db(pred_wav), 2)

    # Comparative
    lmsd_l1, lmsd_l2 = log_mel_spectral_distance(gt_wav, pred_wav, sr)
    r["lmsd_l1"] = round(lmsd_l1, 3)
    r["lmsd_l2"] = round(lmsd_l2, 3)
    r["energy_envelope_corr"] = round(energy_envelope_correlation(gt_wav, pred_wav, sr), 4)

    # Deltas
    r["noise_floor_delta_db"] = round(r["pred_noise_floor_db"] - r["gt_noise_floor_db"], 2)
    r["snr_delta_db"] = round(r["pred_snr_db"] - r["gt_snr_db"], 2)
    r["spectral_flatness_delta"] = round(r["pred_spectral_flatness"] - r["gt_spectral_flatness"], 5)
    r["rms_delta_db"] = round(r["pred_rms_db"] - r["gt_rms_db"], 2)

    # Whisper results (filled in later)
    if whisper_results:
        r.update(whisper_results)
    else:
        r["whisper_gt_transcript"] = ""
        r["whisper_pred_transcript"] = ""
        r["wer_vs_ref_pct"] = 0.0
        r["cer_vs_ref_pct"] = 0.0

    return r


# ─── Report Generation ───────────────────────────────────────────────────────

def generate_report(all_results, report_path):
    L = []

    L.append("# MambaFlow TTS — Raw Audio Noise & Generalization Report")
    L.append("")
    L.append("## Purpose")
    L.append("")
    L.append("Comparing **raw, unmodified** ground-truth input audio vs raw model-generated audio.")
    L.append("No mastering, no EQ, no loudness normalization applied. Pure model output.")
    L.append("Goal: assess noise levels, generalization quality, and variance capture.")
    L.append("")
    L.append("---")
    L.append("")

    # Per-sample
    L.append("## Per-Sample Detailed Analysis")
    L.append("")

    for r in all_results:
        L.append(f"### Sample: {r['audio_id']}")
        L.append(f"**Text:** \"{r['text']}\"")
        L.append(f"**Duration:** GT={r['duration_sec_gt']}s | Pred={r['duration_sec_pred']}s")
        L.append("")

        L.append("#### Energy & Peak")
        L.append("")
        L.append("| Metric | Ground Truth | Model Predicted | Delta |")
        L.append("|--------|-------------|-----------------|-------|")
        L.append(f"| RMS Energy (dB) | {r['gt_rms_db']} | {r['pred_rms_db']} | {r['rms_delta_db']:+.2f} |")
        L.append(f"| Peak Amplitude (dBFS) | {r['gt_peak_dbfs']} | {r['pred_peak_dbfs']} | {r['pred_peak_dbfs'] - r['gt_peak_dbfs']:+.2f} |")
        L.append(f"| Crest Factor (dB) | {r['gt_crest_factor_db']} | {r['pred_crest_factor_db']} | {r['pred_crest_factor_db'] - r['gt_crest_factor_db']:+.2f} |")
        L.append("")

        L.append("#### Noise Assessment")
        L.append("")
        L.append("| Metric | Ground Truth | Model Predicted | Delta |")
        L.append("|--------|-------------|-----------------|-------|")
        L.append(f"| Noise Floor (dB) | {r['gt_noise_floor_db']} | {r['pred_noise_floor_db']} | {r['noise_floor_delta_db']:+.2f} |")
        L.append(f"| SNR Estimate (dB) | {r['gt_snr_db']} | {r['pred_snr_db']} | {r['snr_delta_db']:+.2f} |")
        L.append(f"| Spectral Flatness | {r['gt_spectral_flatness']} | {r['pred_spectral_flatness']} | {r['spectral_flatness_delta']:+.5f} |")
        L.append("")

        nf_delta = r["noise_floor_delta_db"]
        if nf_delta > 5:
            verdict = "SIGNIFICANT noise — noise floor raised substantially"
        elif nf_delta > 2:
            verdict = "MODERATE noise — noise floor elevated, audible hiss likely"
        elif nf_delta > 0:
            verdict = "MILD noise increase — slight elevation but may not dominate"
        else:
            verdict = "No additional noise — model output is as clean or cleaner than GT"

        sf_delta = r["spectral_flatness_delta"]
        if sf_delta > 0.05:
            verdict += "; spectrum is notably flatter (more noise-like)"
        elif sf_delta > 0.01:
            verdict += "; spectrum slightly flatter than GT"

        L.append(f"> **Noise Verdict:** {verdict}")
        L.append("")

        L.append("#### Spectral Characteristics")
        L.append("")
        L.append("| Metric | Ground Truth | Model Predicted |")
        L.append("|--------|-------------|-----------------|")
        L.append(f"| Spectral Centroid (Hz) | {r['gt_spectral_centroid_hz']} | {r['pred_spectral_centroid_hz']} |")
        L.append(f"| Spectral Bandwidth (Hz) | {r['gt_spectral_bandwidth_hz']} | {r['pred_spectral_bandwidth_hz']} |")
        L.append(f"| Log-Mel Spectral Distance (L1) | — | {r['lmsd_l1']} |")
        L.append(f"| Log-Mel Spectral Distance (L2) | — | {r['lmsd_l2']} |")
        L.append(f"| Energy Envelope Correlation | — | {r['energy_envelope_corr']} |")
        L.append("")

        L.append("#### Whisper ASR (Word Error Rate)")
        L.append("")
        L.append(f"- **Reference text:** \"{r['text']}\"")
        L.append(f"- **Whisper on GT audio:** \"{r['whisper_gt_transcript']}\"")
        L.append(f"- **Whisper on Pred audio:** \"{r['whisper_pred_transcript']}\"")
        L.append(f"- **WER (vs reference):** {r['wer_vs_ref_pct']}%")
        L.append(f"- **CER (vs reference):** {r['cer_vs_ref_pct']}%")
        L.append("")
        L.append("---")
        L.append("")

    # Summary table
    L.append("## Summary Comparison Table")
    L.append("")
    L.append("| Sample | GT RMS (dB) | Pred RMS (dB) | GT Peak (dBFS) | Pred Peak (dBFS) | GT SNR (dB) | Pred SNR (dB) | SNR Delta | NF Delta | Flatness Delta | LMSD L1 | Env Corr | WER% |")
    L.append("|--------|------------|--------------|---------------|-----------------|------------|--------------|-----------|----------|----------------|---------|----------|------|")
    for r in all_results:
        L.append(
            f"| {r['audio_id']} "
            f"| {r['gt_rms_db']} "
            f"| {r['pred_rms_db']} "
            f"| {r['gt_peak_dbfs']} "
            f"| {r['pred_peak_dbfs']} "
            f"| {r['gt_snr_db']} "
            f"| {r['pred_snr_db']} "
            f"| {r['snr_delta_db']:+.1f} "
            f"| {r['noise_floor_delta_db']:+.1f} "
            f"| {r['spectral_flatness_delta']:+.5f} "
            f"| {r['lmsd_l1']} "
            f"| {r['energy_envelope_corr']} "
            f"| {r['wer_vs_ref_pct']} |"
        )
    L.append("")

    # Aggregates
    L.append("## Aggregate Statistics")
    L.append("")

    avg = lambda key: np.mean([r[key] for r in all_results])
    std_fn = lambda key: np.std([r[key] for r in all_results])

    avg_snr_d = avg("snr_delta_db")
    avg_nf_d = avg("noise_floor_delta_db")
    avg_sf_d = avg("spectral_flatness_delta")
    avg_lmsd = avg("lmsd_l1")
    avg_env = avg("energy_envelope_corr")
    avg_wer = avg("wer_vs_ref_pct")
    avg_gt_snr = avg("gt_snr_db")
    avg_pred_snr = avg("pred_snr_db")
    avg_gt_crest = avg("gt_crest_factor_db")
    avg_pred_crest = avg("pred_crest_factor_db")
    std_snr_d = std_fn("snr_delta_db")

    L.append("| Metric | Value |")
    L.append("|--------|-------|")
    L.append(f"| Average GT SNR (dB) | {avg_gt_snr:.2f} |")
    L.append(f"| Average Pred SNR (dB) | {avg_pred_snr:.2f} |")
    L.append(f"| Average SNR Delta (Pred - GT) | {avg_snr_d:+.2f} dB |")
    L.append(f"| SNR Delta Std Dev | {std_snr_d:.2f} dB |")
    L.append(f"| Average Noise Floor Delta | {avg_nf_d:+.2f} dB |")
    L.append(f"| Average Spectral Flatness Delta | {avg_sf_d:+.5f} |")
    L.append(f"| Average GT Crest Factor (dB) | {avg_gt_crest:.2f} |")
    L.append(f"| Average Pred Crest Factor (dB) | {avg_pred_crest:.2f} |")
    L.append(f"| Average Log-Mel Spectral Dist (L1) | {avg_lmsd:.3f} |")
    L.append(f"| Average Energy Envelope Correlation | {avg_env:.4f} |")
    L.append(f"| Average WER (%) | {avg_wer:.1f} |")
    L.append("")

    # Variance analysis
    L.append("## Generalization & Variance Analysis")
    L.append("")
    L.append("### Does the model preserve dynamic range across samples?")
    L.append("")
    L.append("| Metric | GT Std Dev | Pred Std Dev | Ratio (Pred/GT) |")
    L.append("|--------|-----------|-------------|-----------------|")

    gt_rms_std = std_fn("gt_rms_db")
    pred_rms_std = std_fn("pred_rms_db")
    rms_ratio = pred_rms_std / gt_rms_std if gt_rms_std > 0.001 else 0.0
    L.append(f"| RMS Energy (dB) | {gt_rms_std:.3f} | {pred_rms_std:.3f} | {rms_ratio:.3f} |")

    gt_snr_std = std_fn("gt_snr_db")
    pred_snr_std = std_fn("pred_snr_db")
    snr_ratio = pred_snr_std / gt_snr_std if gt_snr_std > 0.001 else 0.0
    L.append(f"| SNR (dB) | {gt_snr_std:.3f} | {pred_snr_std:.3f} | {snr_ratio:.3f} |")

    gt_crest_std = std_fn("gt_crest_factor_db")
    pred_crest_std = std_fn("pred_crest_factor_db")
    crest_ratio = pred_crest_std / gt_crest_std if gt_crest_std > 0.001 else 0.0
    L.append(f"| Crest Factor (dB) | {gt_crest_std:.3f} | {pred_crest_std:.3f} | {crest_ratio:.3f} |")
    L.append("")

    L.append("> **Interpretation:** Ratio close to 1.0 = model preserves natural variance.")
    L.append("> Ratio << 1.0 = variance collapse (all outputs sound similar).")
    L.append("> Ratio >> 1.0 = model amplifies differences beyond GT.")
    L.append("")

    # Overall verdict
    L.append("## Overall Verdict")
    L.append("")

    if avg_snr_d < -10:
        noise_level, noise_desc = "SEVERE", "The model introduces heavy noise. SNR drops dramatically. The predicted audio has a significantly elevated noise floor. This points to the flow matching not converging — the velocity field is not driving the latent close enough to the ground truth."
    elif avg_snr_d < -5:
        noise_level, noise_desc = "SIGNIFICANT", "The model introduces noticeable noise. The predicted audio has a measurably higher noise floor than GT. The SNR degradation is consistent across samples, suggesting incomplete denoising during Euler integration — the trajectory stops short of the clean signal manifold."
    elif avg_snr_d < -2:
        noise_level, noise_desc = "MODERATE", "Moderate noise in model output. SNR is somewhat lower than GT, spectral flatness slightly higher. Suggests partial convergence — the flow trajectory moves toward the target but retains residual noise."
    elif avg_snr_d < 0:
        noise_level, noise_desc = "MILD", "Only mild noise degradation compared to GT. Model is reasonably close to input quality with minor spectral differences."
    else:
        noise_level, noise_desc = "CLEAN", "Model output is as clean as or cleaner than the ground truth. No additional noise introduced."

    L.append(f"### Noise Level: {noise_level}")
    L.append("")
    L.append(noise_desc)
    L.append("")

    if rms_ratio < 0.3:
        gen_v = "VARIANCE COLLAPSE — produces outputs with similar energy regardless of input. Generalization failure."
    elif rms_ratio < 0.7:
        gen_v = "Partially captures variance but compresses dynamic range. Outputs less differentiated than GT."
    elif rms_ratio < 1.3:
        gen_v = "Preserves variance well. Predicted outputs maintain similar spread of energy levels as GT. Good generalization."
    else:
        gen_v = "Amplifies variance beyond GT levels. May indicate instability or over-sensitivity."

    L.append("### Generalization Assessment")
    L.append("")
    L.append(gen_v)
    L.append("")

    L.append("### Flow Matching Convergence")
    L.append("")
    if avg_env > 0.7:
        L.append("Energy envelope correlation is high — temporal dynamics well-preserved. The model captures when speech energy rises and falls.")
    elif avg_env > 0.4:
        L.append("Energy envelope correlation is moderate. Captures some temporal dynamics but misses others.")
    else:
        L.append("Energy envelope correlation is low. Temporal energy pattern does not closely follow GT, suggesting the flow trajectory has not converged.")
    L.append("")

    L.append("### Intelligibility (Whisper WER)")
    L.append("")
    if avg_wer < 10:
        L.append(f"Average WER is {avg_wer:.1f}% — excellent intelligibility.")
    elif avg_wer < 20:
        L.append(f"Average WER is {avg_wer:.1f}% — good intelligibility with some errors.")
    elif avg_wer < 35:
        L.append(f"Average WER is {avg_wer:.1f}% — moderate intelligibility. Some words garbled.")
    else:
        L.append(f"Average WER is {avg_wer:.1f}% — poor intelligibility. Noise/distortion significantly impacts clarity.")
    L.append("")

    L.append("### Key Findings")
    L.append("")
    L.append(f"1. **Noise floor is {abs(avg_nf_d):.1f} dB {'higher' if avg_nf_d > 0 else 'lower'}** in model output vs ground truth on average.")
    L.append(f"2. **SNR drops by {abs(avg_snr_d):.1f} dB** from GT ({avg_gt_snr:.1f} dB) to Pred ({avg_pred_snr:.1f} dB).")
    L.append(f"3. **Spectral flatness change: {avg_sf_d:+.5f}** — {'more noise-like spectrum' if avg_sf_d > 0 else 'predictions slightly more tonal'}.")
    L.append(f"4. **LMSD L1 = {avg_lmsd:.3f}** — {'high spectral divergence' if avg_lmsd > 1.5 else 'moderate' if avg_lmsd > 1.0 else 'good spectral match'}.")
    L.append(f"5. **Energy envelope correlation = {avg_env:.4f}** — {'well-preserved' if avg_env > 0.6 else 'partially preserved' if avg_env > 0.3 else 'poorly captured'}.")
    L.append(f"6. **Crest factor: GT={avg_gt_crest:.1f} dB, Pred={avg_pred_crest:.1f} dB** — {'compressed dynamics' if avg_pred_crest < avg_gt_crest - 2 else 'similar dynamics' if abs(avg_pred_crest - avg_gt_crest) < 2 else 'expanded dynamics'}.")
    L.append(f"7. **Variance ratio (RMS): {rms_ratio:.3f}** — {'good preservation' if 0.7 <= rms_ratio <= 1.3 else 'variance issues'}.")
    L.append("")

    text = "\n".join(L)
    with open(report_path, "w") as f:
        f.write(text)
    return text


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  RAW AUDIO NOISE & GENERALIZATION ANALYSIS")
    print("  (Self-contained: loads model, generates audio, analyzes)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dacvae_device = "cpu"
    val_csv = "/home/monesh/ljspeech/LJSpeech-1.1/val.csv"
    ckpt_dir = os.path.join(CURRENT_DIR, "TTS_checkpoints")
    n_steps = 30
    temperature = 1.0
    sr = 48000

    # 1. Resolve checkpoint
    best_ckpt, best_loss = find_best_checkpoint(ckpt_dir)
    if best_ckpt is None:
        print("[ERROR] No checkpoint found!")
        sys.exit(1)
    loss_str = f"(val_loss={best_loss:.4f})" if best_loss is not None else ""
    print(f"[+] Checkpoint: {best_ckpt} {loss_str}")

    # 2. Load model
    print(f"[+] Loading MambaFlow model on {device}...")
    pl_module = TTSMODEL.load_from_checkpoint(best_ckpt, map_location=device)
    model = pl_module.model.to(device).eval()

    # 3. Load DACVAE
    print(f"[+] Loading DACVAE decoder on {dacvae_device}...")
    dacvae_decoder, sr, _ = setup_dacvae(device=dacvae_device)

    # 4. Load Whisper ASR
    print("[+] Loading Whisper ASR (tiny.en)...")
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").eval().to("cpu")
    resampler_16k = T.Resample(48000, 16000)

    # 5. Load last 5 validation samples
    df = pd.read_csv(val_csv)
    last_5 = df.tail(5).reset_index(drop=True)

    all_results = []
    output_dir = os.path.join(CURRENT_DIR, "raw_noise_analysis_output")
    os.makedirs(output_dir, exist_ok=True)

    for i, (_, row) in enumerate(last_5.iterrows()):
        sample_idx = i + 1

        # Generate raw audio pair
        gt_wav, pred_wav, audio_id, text = generate_raw_audio_pair(
            model=model, dacvae_decoder=dacvae_decoder, sample_row=row,
            sample_index=sample_idx, device=device, dacvae_device=dacvae_device,
            n_steps=n_steps, temperature=temperature, sr=sr,
        )

        # Save raw wavs for reference
        sf.write(os.path.join(output_dir, f"s{sample_idx}_{audio_id}_gt_raw.wav"), gt_wav, sr)
        sf.write(os.path.join(output_dir, f"s{sample_idx}_{audio_id}_pred_raw.wav"), pred_wav, sr)

        # Whisper ASR on raw audio (resample to 16kHz)
        print(f"  [+] Running Whisper ASR...")
        gt_16k = resampler_16k(torch.tensor(gt_wav).float().unsqueeze(0)).squeeze().numpy()
        pred_16k = resampler_16k(torch.tensor(pred_wav).float().unsqueeze(0)).squeeze().numpy()

        with torch.no_grad():
            gt_feats = whisper_proc(gt_16k, sampling_rate=16000, return_tensors="pt").input_features
            gt_ids = whisper_model.generate(gt_feats)
            whisper_gt = whisper_proc.batch_decode(gt_ids, skip_special_tokens=True)[0].strip()

            pred_feats = whisper_proc(pred_16k, sampling_rate=16000, return_tensors="pt").input_features
            pred_ids = whisper_model.generate(pred_feats)
            whisper_pred = whisper_proc.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

        wer = calculate_wer(text, whisper_pred)
        cer = calculate_cer(text, whisper_pred)

        whisper_results = {
            "whisper_gt_transcript": whisper_gt,
            "whisper_pred_transcript": whisper_pred,
            "wer_vs_ref_pct": round(wer * 100, 1),
            "cer_vs_ref_pct": round(cer * 100, 1),
        }

        # Analyze
        print(f"  [+] Computing audio metrics...")
        result = analyze_sample(gt_wav, pred_wav, sr, audio_id, text, whisper_results)

        print(f"    GT:   RMS={result['gt_rms_db']} dB | Peak={result['gt_peak_dbfs']} dBFS | SNR={result['gt_snr_db']} dB | Flatness={result['gt_spectral_flatness']}")
        print(f"    Pred: RMS={result['pred_rms_db']} dB | Peak={result['pred_peak_dbfs']} dBFS | SNR={result['pred_snr_db']} dB | Flatness={result['pred_spectral_flatness']}")
        print(f"    Delta: SNR={result['snr_delta_db']:+.1f} dB | NF={result['noise_floor_delta_db']:+.1f} dB | LMSD={result['lmsd_l1']:.3f} | EnvCorr={result['energy_envelope_corr']:.4f}")
        print(f"    Whisper GT: \"{whisper_gt}\"")
        print(f"    Whisper Pred: \"{whisper_pred}\"")
        print(f"    WER={result['wer_vs_ref_pct']}% | CER={result['cer_vs_ref_pct']}%")

        all_results.append(result)

    # Generate report
    report_path = os.path.join(CURRENT_DIR, "raw_audio_noise_report.md")
    report = generate_report(all_results, report_path)

    # Save JSON
    json_path = os.path.join(CURRENT_DIR, "raw_audio_noise_analysis.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"  REPORT: {report_path}")
    print(f"  JSON:   {json_path}")
    print(f"  WAVs:   {output_dir}/")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
