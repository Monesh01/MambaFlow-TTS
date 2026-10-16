import os
import sys
import time
import json
import ast
import re
import glob
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal
import pyloudnorm as pyln
from pesq import pesq
from pystoi import stoi
import librosa
import torch
import torchaudio.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import WhisperProcessor, WhisperForConditionalGeneration

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from TTSDataModule import TTSMODEL
from preprocessing.text import text_to_sequence
from stats import denormalize_latent, normalize_latent
from generate_audio import setup_dacvae, decode_latent_to_audio, synthesize_text


def find_best_checkpoint(ckpt_dir="TTS_checkpoints"):
    """
    Auto-detects the best checkpoint based on lowest validation loss in checkpoint filenames,
    prioritizing epoch checkpoints matching the current architecture.
    """
    if not os.path.exists(ckpt_dir):
        return None, None

    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None, None

    best_ckpt = None
    min_loss = float("inf")
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


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Computes Word Error Rate (WER) using Levenshtein distance."""
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


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Computes Character Error Rate (CER) using Levenshtein distance."""
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


class ProductionAudioMasteringPipeline:
    """
    Production-grade audio mastering and post-processing tools:
    1. Infrasonic DC / sub-rumble high-pass filtering (60 Hz 4th-order Butterworth).
    2. EBU R128 loudness normalization to target LUFS (-23.0 LUFS standard) with true peak ceiling (-1.0 dBFS).
    3. Boundary anti-pop cosine taper (15ms).
    """
    def __init__(self, target_lufs=-23.0, true_peak_limit_db=-1.0, highpass_cutoff_hz=60.0):
        self.target_lufs = target_lufs
        self.peak_limit_linear = 10.0 ** (true_peak_limit_db / 20.0)
        self.highpass_cutoff = highpass_cutoff_hz

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        wav = np.asarray(audio, dtype=np.float32).copy()
        if len(wav) < int(sr * 0.05):
            return wav

        # 1. Remove DC offset and sub-audible infrasonic rumble (<60 Hz)
        sos = scipy.signal.butter(4, self.highpass_cutoff, btype="highpass", fs=sr, output="sos")
        wav = scipy.signal.sosfiltfilt(sos, wav)

        # 2. EBU R128 Loudness Normalization
        try:
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(wav)
            if np.isfinite(loudness) and loudness > -70.0:
                wav_norm = pyln.normalize.loudness(wav, loudness, self.target_lufs)
                peak = np.max(np.abs(wav_norm))
                # Peak limiting if needed to prevent clipping
                if peak > self.peak_limit_linear:
                    wav = wav_norm * (self.peak_limit_linear / peak)
                else:
                    wav = wav_norm
            else:
                peak = np.max(np.abs(wav))
                if peak > 1e-5:
                    wav = wav * (self.peak_limit_linear / max(peak, self.peak_limit_linear))
        except Exception:
            peak = np.max(np.abs(wav))
            if peak > 1e-5:
                wav = wav * (self.peak_limit_linear / max(peak, self.peak_limit_linear))

        # 3. Smooth boundary fade-in and fade-out (15ms)
        fade_len = int(sr * 0.015)
        if len(wav) > 2 * fade_len:
            fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len)))
            fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
            wav[:fade_len] *= fade_in
            wav[-fade_len:] *= fade_out

        return np.clip(wav, -1.0, 1.0)


def compute_log_mel_spectrogram_distance(synth_wav, ref_wav, sr=48000, n_mels=80):
    """Computes L1 and L2 Log-Mel Spectral Distance (LMSD)."""
    mel_synth = librosa.feature.melspectrogram(y=synth_wav, sr=sr, n_fft=1024, hop_length=256, n_mels=n_mels)
    mel_ref = librosa.feature.melspectrogram(y=ref_wav, sr=sr, n_fft=1024, hop_length=256, n_mels=n_mels)
    log_synth = np.log(np.maximum(mel_synth, 1e-5))
    log_ref = np.log(np.maximum(mel_ref, 1e-5))
    min_t = min(log_synth.shape[1], log_ref.shape[1])
    diff = log_synth[:, :min_t] - log_ref[:, :min_t]
    l1 = float(np.mean(np.abs(diff)))
    l2 = float(np.sqrt(np.mean(diff ** 2)))
    return l1, l2


def evaluate_production_sample(
    model,
    dacvae_decoder,
    whisper_model,
    whisper_proc,
    resampler_16k,
    mastering_pipeline,
    sample_row,
    sample_index,
    device="cuda",
    dacvae_device="cpu",
    n_steps=30,
    temperature=1.0,
    output_dir="production_evaluation_results",
    sr=48000,
):
    audio_id = sample_row["audio_id"]
    text = str(sample_row.get("normalized_text", sample_row.get("text", "")))
    latent_path = sample_row["latent"]
    dur_list = ast.literal_eval(sample_row["durations"]) if isinstance(sample_row["durations"], str) else sample_row["durations"]

    print(f"\n{'='*85}")
    print(f"  [Production Sample {sample_index}/5] Audio ID: {audio_id}")
    print(f"  Text: \"{text}\"")
    print(f"  Frames: {sum(dur_list)} | Phonemes: {len(dur_list)}")
    print(f"{'='*85}")

    # 1. Load Ground Truth Latent & Decode Ground Truth Audio
    gt_raw = np.squeeze(np.load(latent_path))
    if gt_raw.ndim == 2:
        if gt_raw.shape[0] != 128 and gt_raw.shape[1] == 128:
            gt_raw = gt_raw.T
    T_gt = gt_raw.shape[1]

    gt_tensor = torch.tensor(gt_raw, dtype=torch.float32).unsqueeze(0).to(device)
    gt_norm = normalize_latent(gt_tensor).transpose(1, 2)  # [1, T, 128]

    # Decode Raw GT Audio
    gt_wav_raw = decode_latent_to_audio(dacvae_decoder, gt_tensor, target_device=dacvae_device)
    gt_wav_mastered = mastering_pipeline.process(gt_wav_raw, sr=sr)

    gt_raw_path = os.path.join(output_dir, f"prod_s{sample_index}_{audio_id}_gt_raw.wav")
    gt_mastered_path = os.path.join(output_dir, f"prod_s{sample_index}_{audio_id}_gt_mastered.wav")
    sf.write(gt_raw_path, gt_wav_raw, sr)
    sf.write(gt_mastered_path, gt_wav_mastered, sr)

    # 2. Prepare Conditioning & Run Euler Integration with Velocity Tracking
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

    trajectory_states = [x_curr.clone()]
    velocity_vectors = []
    cos_sim_to_dest = []
    dist_to_dest = []
    alphas = []
    betas = []

    gt_flat = gt_norm.float().flatten()
    x0_flat = x_0.float().flatten()
    A_basis = torch.stack([gt_flat, x0_flat], dim=-1)

    sol_0 = torch.linalg.lstsq(A_basis, x_curr.float().flatten()).solution
    alphas.append(sol_0[0].item())
    betas.append(sol_0[1].item())
    dist_to_dest.append(torch.norm(x_curr.float().flatten() - gt_flat).item())

    t_infer_start = time.time()
    with torch.inference_mode():
        x_enc = model.txt_emb(text_tensor)
        enc_out = model.encoder(x_enc, mask=text_mask)

        for step in range(n_steps):
            t_val = step * dt
            t_tensor = torch.full((1,), t_val, device=device, dtype=torch.float32)

            v = model.decoder.predict_velocity(
                x=x_curr,
                t=t_tensor,
                text_repr=enc_out,
                durations=durations,
                text_mask=text_mask,
                audio_mask=audio_mask,
            )
            velocity_vectors.append(v.clone())

            dest_dir = gt_norm - x_curr
            v_flat = v.float().flatten()
            dest_flat = dest_dir.float().flatten()
            v_norm = torch.norm(v_flat).item()
            dest_norm = torch.norm(dest_flat).item()

            if v_norm > 1e-7 and dest_norm > 1e-7:
                cos_dest = torch.dot(v_flat, dest_flat).item() / (v_norm * dest_norm)
                cos_dest = float(np.clip(cos_dest, -1.0, 1.0))
            else:
                cos_dest = 0.0

            cos_sim_to_dest.append(cos_dest)

            # Euler step
            x_curr = x_curr + dt * v
            if audio_mask is not None:
                x_curr = x_curr * audio_mask.unsqueeze(-1).float()

            trajectory_states.append(x_curr.clone())
            x_flat = x_curr.float().flatten()
            sol = torch.linalg.lstsq(A_basis, x_flat).solution
            alphas.append(sol[0].item())
            betas.append(sol[1].item())
            dist_to_dest.append(torch.norm(x_flat - gt_flat).item())

    total_infer_time = time.time() - t_infer_start

    # 3. Decode Predicted Audio & Apply Production Audio Mastering
    x_1 = trajectory_states[-1]
    pred_latent_denorm = denormalize_latent(x_1.float())
    pred_latent_128_t = pred_latent_denorm.transpose(1, 2)
    pred_wav_raw = decode_latent_to_audio(dacvae_decoder, pred_latent_128_t, target_device=dacvae_device)

    # Master through production audio pipeline
    pred_wav_mastered = mastering_pipeline.process(pred_wav_raw, sr=sr)

    pred_raw_path = os.path.join(output_dir, f"prod_s{sample_index}_{audio_id}_pred_raw.wav")
    pred_mastered_path = os.path.join(output_dir, f"prod_s{sample_index}_{audio_id}_pred_mastered.wav")
    sf.write(pred_raw_path, pred_wav_raw, sr)
    sf.write(pred_mastered_path, pred_wav_mastered, sr)

    # Also synthesize full TTS with predicted duration
    tts_wav_raw, tts_frames, _ = synthesize_text(
        model=model,
        text=text,
        dacvae_decoder=dacvae_decoder,
        device=device,
        timesteps=n_steps,
        temperature=temperature,
        solver="euler",
    )
    tts_wav_mastered = mastering_pipeline.process(tts_wav_raw, sr=sr)
    tts_mastered_path = os.path.join(output_dir, f"prod_s{sample_index}_{audio_id}_tts_mastered.wav")
    sf.write(tts_mastered_path, tts_wav_mastered, sr)

    # 4. Production Metrics: Loudness, RTF, PESQ, STOI, LMSD
    audio_dur = len(pred_wav_mastered) / sr
    rtf = total_infer_time / audio_dur if audio_dur > 0 else 0.0
    throughput_khz = (len(pred_wav_mastered) / 1000.0) / total_infer_time if total_infer_time > 0 else 0.0

    meter = pyln.Meter(sr)
    raw_loudness = meter.integrated_loudness(pred_wav_raw)
    mastered_loudness = meter.integrated_loudness(pred_wav_mastered)
    gt_loudness = meter.integrated_loudness(gt_wav_mastered)

    # Resample to 16kHz for PESQ and STOI
    pred_16k = scipy.signal.resample_poly(pred_wav_mastered, 16000, sr)
    gt_16k = scipy.signal.resample_poly(gt_wav_mastered, 16000, sr)
    min_len = min(len(pred_16k), len(gt_16k))

    try:
        pesq_score = float(pesq(16000, gt_16k[:min_len], pred_16k[:min_len], "wb"))
    except Exception as e:
        pesq_score = 1.0

    try:
        stoi_score = float(stoi(gt_16k[:min_len], pred_16k[:min_len], 16000, extended=False))
    except Exception as e:
        stoi_score = 0.0

    lmsd_l1, lmsd_l2 = compute_log_mel_spectrogram_distance(pred_wav_mastered, gt_wav_mastered, sr=sr)

    # 5. Whisper WSR / ASR Transcription and Error Rates
    gt_16k_tensor = resampler_16k(torch.tensor(gt_wav_mastered).float().unsqueeze(0)).squeeze().numpy()
    pred_16k_tensor = resampler_16k(torch.tensor(pred_wav_mastered).float().unsqueeze(0)).squeeze().numpy()

    with torch.no_grad():
        gt_feats = whisper_proc(gt_16k_tensor, sampling_rate=16000, return_tensors="pt").input_features
        gt_ids = whisper_model.generate(gt_feats)
        whisper_gt = whisper_proc.batch_decode(gt_ids, skip_special_tokens=True)[0].strip()

        pred_feats = whisper_proc(pred_16k_tensor, sampling_rate=16000, return_tensors="pt").input_features
        pred_ids = whisper_model.generate(pred_feats)
        whisper_pred = whisper_proc.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

    wer_vs_ref = calculate_wer(text, whisper_pred)
    cer_vs_ref = calculate_cer(text, whisper_pred)
    wer_vs_gt = calculate_wer(whisper_gt, whisper_pred)

    print(f"  [WSR Reference]:   \"{text}\"")
    print(f"  [WSR GroundTruth]: \"{whisper_gt}\"")
    print(f"  [WSR ModelPred]:   \"{whisper_pred}\"")
    print(f"  [WSR Metrics]:     WER(vs Ref)={wer_vs_ref*100:.1f}% | CER={cer_vs_ref*100:.1f}% | WER(vs GT Audio)={wer_vs_gt*100:.1f}%")
    print(f"  [Audio Quality]:   PESQ={pesq_score:.3f} | STOI={stoi_score:.3f} | LMSD(L1)={lmsd_l1:.3f}")
    print(f"  [Loudness]:        Raw={raw_loudness:.1f} LUFS -> Mastered={mastered_loudness:.1f} LUFS (GT: {gt_loudness:.1f} LUFS)")
    print(f"  [Performance]:     RTF={rtf:.3f} ({throughput_khz:.1f} kHz/s) | Latency={total_infer_time:.2f}s")

    # 6. PCA Projection
    traj_mat = torch.stack([s.float().flatten() for s in trajectory_states] + [gt_flat], dim=0).cpu().numpy()
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(traj_mat)
    pca_steps = pca_coords[:-1]
    pca_gt = pca_coords[-1]

    sample_metrics = {
        "sample_index": sample_index,
        "audio_id": audio_id,
        "text": text,
        "whisper_gt": whisper_gt,
        "whisper_pred": whisper_pred,
        "wer_vs_ref_pct": round(wer_vs_ref * 100.0, 1),
        "cer_vs_ref_pct": round(cer_vs_ref * 100.0, 1),
        "wer_vs_gt_pct": round(wer_vs_gt * 100.0, 1),
        "pesq_wb": round(pesq_score, 3),
        "stoi": round(stoi_score, 3),
        "lmsd_l1": round(lmsd_l1, 3),
        "lmsd_l2": round(lmsd_l2, 3),
        "raw_loudness_lufs": round(float(raw_loudness), 1),
        "mastered_loudness_lufs": round(float(mastered_loudness), 1),
        "audio_duration_sec": round(audio_dur, 2),
        "inference_latency_sec": round(total_infer_time, 2),
        "rtf": round(rtf, 3),
        "throughput_khz_sec": round(throughput_khz, 1),
        "target_alpha_final": round(alphas[-1], 3),
        "noise_beta_final": round(betas[-1], 3),
        "final_l2_dist": round(dist_to_dest[-1], 2),
        "mean_velocity_align": round(float(np.mean(cos_sim_to_dest)), 3),
        "alphas": alphas,
        "betas": betas,
        "cos_sim_to_dest": cos_sim_to_dest,
        "dist_to_dest": dist_to_dest,
        "pca_steps_x": pca_steps[:, 0].tolist(),
        "pca_steps_y": pca_steps[:, 1].tolist(),
        "pca_gt_x": float(pca_gt[0]),
        "pca_gt_y": float(pca_gt[1]),
        "pca_var": pca.explained_variance_ratio_.tolist(),
        "pred_raw_path": pred_raw_path,
        "pred_mastered_path": pred_mastered_path,
        "gt_mastered_path": gt_mastered_path,
        "tts_mastered_path": tts_mastered_path,
    }

    return sample_metrics, gt_wav_mastered, pred_wav_mastered, pred_wav_raw


def plot_production_overview_dashboard(all_metrics, sample_audio_pairs, output_dir="production_evaluation_results", sr=48000):
    """
    Plots a multi-panel production grade dashboard:
    1. Phase-plane trajectories with velocity arrows
    2. Velocity heading alignment across 30 Euler steps
    3. Production WSR & Audio Quality metrics table
    4. Spectrogram & Waveform comparison (Ground Truth vs Mastered Output vs Raw Output)
    """
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle(
        "MAMBAFLOW-TTS: PRODUCTION-GRADE TTS EVALUATION & WSR BENCHMARK\n"
        "Model: 256D BiMamba-2 + Cross-Attention | Checkpoint: Best Validation Loss (Epoch 82, Loss 0.9754) | 48 kHz DACAE",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Row 1: 5 Phase Plane Trajectory plots showing velocity arrows, Start, End, Destination
    for i, m in enumerate(all_metrics):
        ax = plt.subplot2grid((3, 5), (0, i))
        alphas = np.array(m["alphas"])
        betas = np.array(m["betas"])

        ax.plot([0.0, 1.0], [1.0, 0.0], "k--", alpha=0.35, linewidth=1.5, label="Ideal Flow Path")
        ax.plot(alphas, betas, color=colors[i], linewidth=2.2, label="Model Trajectory")

        # Velocity direction arrows
        for step in range(0, len(alphas) - 1, 4):
            ax.annotate(
                "",
                xy=(alphas[step + 1], betas[step + 1]),
                xytext=(alphas[step], betas[step]),
                arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.5, shrinkA=0, shrinkB=0),
            )

        ax.scatter([alphas[0]], [betas[0]], color="red", s=70, marker="X", zorder=5, label="Start (t=0)")
        ax.scatter([1.0], [0.0], color="green", s=180, marker="*", zorder=5, label="Dest Target (x_gt)")
        ax.scatter([alphas[-1]], [betas[-1]], color="orange", s=80, marker="o", edgecolors="black", zorder=5, label="End (t=1)")
        ax.plot([alphas[-1], 1.0], [betas[-1], 0.0], "r:", alpha=0.7, linewidth=1.5)

        ax.set_title(f"S{m['sample_index']}: {m['audio_id']}\nEnd: ({alphas[-1]:.2f}, {betas[-1]:.2f})", fontsize=10, fontweight="bold")
        ax.set_xlabel(r"Target Proj $\alpha$ (Goal=1)", fontsize=9)
        ax.set_ylabel(r"Noise Proj $\beta$ (Goal=0)", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Row 2, Col 0: Velocity Heading Alignment
    ax_vel = plt.subplot2grid((3, 5), (1, 0))
    steps_arr = np.arange(30)
    for i, m in enumerate(all_metrics):
        ax_vel.plot(steps_arr, m["cos_sim_to_dest"], color=colors[i], marker="o", markersize=3, label=f"S{m['sample_index']}")
    ax_vel.axhline(1.0, color="green", linestyle=":", label="Direct Destination (+1.0)")
    ax_vel.axhline(0.0, color="gray", linestyle=":", label="Perpendicular (0.0)")
    ax_vel.set_title("Velocity Direction Heading\ncos(v_t, x_gt - x_t)", fontsize=10, fontweight="bold")
    ax_vel.set_xlabel("Euler Step k (0 -> 29)", fontsize=9)
    ax_vel.set_ylabel("Cosine Similarity", fontsize=9)
    ax_vel.set_ylim(-0.2, 1.05)
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend(fontsize=7, loc="lower right")

    # Row 2, Col 1: Distance to Destination
    ax_dist = plt.subplot2grid((3, 5), (1, 1))
    t_axis = np.linspace(0.0, 1.0, 31)
    for i, m in enumerate(all_metrics):
        ax_dist.plot(t_axis, m["dist_to_dest"], color=colors[i], marker="s", markersize=3, label=f"S{m['sample_index']}")
    ax_dist.set_title("Distance to Destination Target\n||x_t - x_gt||_2", fontsize=10, fontweight="bold")
    ax_dist.set_xlabel("Flow Timestep t", fontsize=9)
    ax_dist.set_ylabel("Euclidean Distance", fontsize=9)
    ax_dist.grid(True, alpha=0.3)
    ax_dist.legend(fontsize=7, loc="upper right")

    # Row 2, Col 2: Production Benchmark Bar Chart (PESQ and STOI)
    ax_bar = plt.subplot2grid((3, 5), (1, 2))
    s_labels = [f"S{m['sample_index']}" for m in all_metrics]
    pesq_vals = [m["pesq_wb"] for m in all_metrics]
    stoi_vals = [m["stoi"] for m in all_metrics]
    x_pos = np.arange(len(s_labels))
    width = 0.35
    ax_bar.bar(x_pos - width/2, pesq_vals, width, label="PESQ (WB, 1-4.5)", color="#2ca02c", alpha=0.85)
    ax_bar.bar(x_pos + width/2, stoi_vals, width, label="STOI (0-1.0)", color="#1f77b4", alpha=0.85)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(s_labels)
    ax_bar.set_title("Objective Speech Quality\nPESQ & STOI Metrics", fontsize=10, fontweight="bold")
    ax_bar.set_ylabel("Score")
    ax_bar.grid(True, alpha=0.3)
    ax_bar.legend(fontsize=8)

    # Row 2, Col 3: WSR Accuracy (100 - WER%)
    ax_wsr = plt.subplot2grid((3, 5), (1, 3))
    wer_vals = [m["wer_vs_ref_pct"] for m in all_metrics]
    cer_vals = [m["cer_vs_ref_pct"] for m in all_metrics]
    ax_wsr.bar(x_pos - width/2, wer_vals, width, label="WER % (vs Ref)", color="#d62728", alpha=0.85)
    ax_wsr.bar(x_pos + width/2, cer_vals, width, label="CER % (vs Ref)", color="#ff7f0e", alpha=0.85)
    ax_wsr.set_xticks(x_pos)
    ax_wsr.set_xticklabels(s_labels)
    ax_wsr.set_title("Whisper ASR Error Rates\nWER % and CER %", fontsize=10, fontweight="bold")
    ax_wsr.set_ylabel("Error Rate (%)")
    ax_wsr.grid(True, alpha=0.3)
    ax_wsr.legend(fontsize=8)

    # Row 2, Col 4: Loudness Mastered vs Raw
    ax_lufs = plt.subplot2grid((3, 5), (1, 4))
    raw_lufs = [m["raw_loudness_lufs"] for m in all_metrics]
    mast_lufs = [m["mastered_loudness_lufs"] for m in all_metrics]
    ax_lufs.bar(x_pos - width/2, raw_lufs, width, label="Raw Output", color="gray", alpha=0.7)
    ax_lufs.bar(x_pos + width/2, mast_lufs, width, label="Mastered (EBU R128)", color="#9467bd", alpha=0.85)
    ax_lufs.axhline(-23.0, color="purple", linestyle="--", label="Target (-23 LUFS)")
    ax_lufs.set_xticks(x_pos)
    ax_lufs.set_xticklabels(s_labels)
    ax_lufs.set_title("Loudness Normalization\nRaw vs Mastered (LUFS)", fontsize=10, fontweight="bold")
    ax_lufs.set_ylabel("Integrated Loudness (LUFS)")
    ax_lufs.grid(True, alpha=0.3)
    ax_lufs.legend(fontsize=8)

    # Row 3: Acoustic Waveform & Mel Spectrogram Display for Sample 4 (short) and Sample 1 (long)
    s4_gt, s4_pred_mast, s4_pred_raw = sample_audio_pairs[3]  # Sample 4
    s1_gt, s1_pred_mast, s1_pred_raw = sample_audio_pairs[0]  # Sample 1

    # Col 0..1: Waveforms (Sample 4)
    ax_wav = plt.subplot2grid((3, 5), (2, 0), colspan=2)
    t4 = np.arange(len(s4_pred_mast)) / sr
    t4_gt = np.arange(len(s4_gt)) / sr
    ax_wav.plot(t4_gt, s4_gt, color="green", alpha=0.6, label="Ground Truth (Mastered)", linewidth=1.0)
    ax_wav.plot(t4, s4_pred_mast, color="blue", alpha=0.7, label="Model (Production Mastered)", linewidth=1.0)
    ax_wav.set_title(f"Waveform Comparison: Sample 4 ({all_metrics[3]['audio_id']}) - \"{all_metrics[3]['text']}\"", fontsize=10, fontweight="bold")
    ax_wav.set_xlabel("Time (seconds)", fontsize=9)
    ax_wav.set_ylabel("Amplitude", fontsize=9)
    ax_wav.set_ylim(-1.05, 1.05)
    ax_wav.grid(True, alpha=0.3)
    ax_wav.legend(fontsize=8, loc="upper right")

    # Col 2..4: Mel Spectrograms (Sample 4 Ground Truth vs Model Mastered)
    ax_spec_gt = plt.subplot2grid((3, 5), (2, 2))
    mel_gt = librosa.feature.melspectrogram(y=s4_gt, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    log_mel_gt = np.log(np.maximum(mel_gt, 1e-5))
    im1 = ax_spec_gt.imshow(log_mel_gt, aspect="auto", origin="lower", cmap="viridis")
    ax_spec_gt.set_title("Ground Truth Log-Mel Spec", fontsize=10, fontweight="bold")
    ax_spec_gt.set_ylabel("Mel Frequency Bands", fontsize=9)
    ax_spec_gt.set_xlabel("Frames", fontsize=9)

    ax_spec_pred = plt.subplot2grid((3, 5), (2, 3))
    mel_pred = librosa.feature.melspectrogram(y=s4_pred_mast, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    log_mel_pred = np.log(np.maximum(mel_pred, 1e-5))
    im2 = ax_spec_pred.imshow(log_mel_pred, aspect="auto", origin="lower", cmap="viridis")
    ax_spec_pred.set_title("Production Model Mastered Spec", fontsize=10, fontweight="bold")
    ax_spec_pred.set_xlabel("Frames", fontsize=9)

    ax_spec_diff = plt.subplot2grid((3, 5), (2, 4))
    min_f = min(log_mel_gt.shape[1], log_mel_pred.shape[1])
    diff_spec = np.abs(log_mel_gt[:, :min_f] - log_mel_pred[:, :min_f])
    im3 = ax_spec_diff.imshow(diff_spec, aspect="auto", origin="lower", cmap="inferno")
    ax_spec_diff.set_title(f"Spectral Error |GT - Pred|\nL1={all_metrics[3]['lmsd_l1']:.2f} dB", fontsize=10, fontweight="bold")
    ax_spec_diff.set_xlabel("Frames", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    dash_file = os.path.join(output_dir, "production_tts_dashboard.png")
    plt.savefig(dash_file, dpi=180)
    plt.close()
    print(f"[+] Saved Production Overview Dashboard: {dash_file}")
    return dash_file


def main():
    parser = argparse.ArgumentParser(description="MambaFlow Production-Grade TTS Evaluation Pipeline")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (default: auto-detect best loss checkpoint)")
    parser.add_argument("--val_csv", type=str, default="/home/monesh/ljspeech/LJSpeech-1.1/val.csv")
    parser.add_argument("--steps", type=int, default=30, help="Euler integration steps (default: 30)")
    parser.add_argument("--temp", type=float, default=1.0, help="Noise temperature (default: 1.0)")
    parser.add_argument("--target_lufs", type=float, default=-23.0, help="EBU R128 loudness target (default: -23.0)")
    parser.add_argument("--output_dir", type=str, default="production_evaluation_results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dacvae_device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Resolve Best Checkpoint
    best_ckpt, best_loss = find_best_checkpoint("TTS_checkpoints")
    ckpt_path = args.ckpt if args.ckpt is not None else best_ckpt
    loss_str = f"(val_loss={best_loss:.4f})" if best_loss is not None else ""
    print(f"[+] Selected Best Checkpoint: {ckpt_path} {loss_str}")

    # 2. Load Model & Modules
    print(f"[+] Loading MambaFlow model on {args.device}...")
    pl_module = TTSMODEL.load_from_checkpoint(ckpt_path, map_location=args.device)
    model = pl_module.model.to(args.device).eval()

    print(f"[+] Loading DACVAE decoder on {args.dacvae_device}...")
    dacvae_decoder, sr, _ = setup_dacvae(device=args.dacvae_device)

    print(f"[+] Loading Whisper ASR (tiny.en) for WSR evaluation...")
    whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").eval().to("cpu")
    resampler_16k = T.Resample(48000, 16000)

    # 3. Initialize Production Mastering Pipeline
    mastering_pipeline = ProductionAudioMasteringPipeline(
        target_lufs=args.target_lufs,
        true_peak_limit_db=-1.0,
        highpass_cutoff_hz=60.0,
    )

    # 4. Load Last 5 Validation Samples
    df = pd.read_csv(args.val_csv)
    last_5 = df.tail(5).reset_index(drop=True)

    all_metrics = []
    sample_audio_pairs = []

    for i, (_, row) in enumerate(last_5.iterrows()):
        m, gt_mast, pred_mast, pred_raw = evaluate_production_sample(
            model=model,
            dacvae_decoder=dacvae_decoder,
            whisper_model=whisper_model,
            whisper_proc=whisper_proc,
            resampler_16k=resampler_16k,
            mastering_pipeline=mastering_pipeline,
            sample_row=row,
            sample_index=i + 1,
            device=args.device,
            dacvae_device=args.dacvae_device,
            n_steps=args.steps,
            temperature=args.temp,
            output_dir=args.output_dir,
            sr=sr,
        )
        all_metrics.append(m)
        sample_audio_pairs.append((gt_mast, pred_mast, pred_raw))

    # 5. Generate Production Dashboard
    plot_production_overview_dashboard(all_metrics, sample_audio_pairs, output_dir=args.output_dir, sr=sr)

    # 6. Save JSON Summary
    summary_path = os.path.join(args.output_dir, "production_benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # 7. Print Comprehensive Production Benchmark Table
    print("\n" + "=" * 125)
    print("  PRODUCTION-GRADE TTS SYSTEM BENCHMARK (BEST CHECKPOINT: EPOCH 82, LOSS 0.9754)")
    print("=" * 125)
    print(f"{'#':<2} | {'Audio ID':<11} | {'WER(Ref)':<9} | {'CER':<6} | {'PESQ':<6} | {'STOI':<6} | {'LMSD':<6} | {'Raw LUFS':<9} | {'Mastered':<9} | {'RTF':<6} | {'Alpha':<6} | {'Beta':<6}")
    print("-" * 125)
    for m in all_metrics:
        print(f"{m['sample_index']:<2} | {m['audio_id']:<11} | {m['wer_vs_ref_pct']:<8.1f}% | {m['cer_vs_ref_pct']:<5.1f}% | {m['pesq_wb']:<6.3f} | {m['stoi']:<6.3f} | {m['lmsd_l1']:<6.2f} | {m['raw_loudness_lufs']:<8.1f} | {m['mastered_loudness_lufs']:<8.1f} | {m['rtf']:<6.3f} | {m['target_alpha_final']:<6.3f} | {m['noise_beta_final']:<6.3f}")
    print("=" * 125)
    print(f"Summary JSON saved: {summary_path}")


if __name__ == "__main__":
    main()
