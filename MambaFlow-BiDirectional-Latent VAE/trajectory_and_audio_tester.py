import os
import sys
import time
import json
import ast
import argparse
import inspect
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import WhisperProcessor, WhisperForConditionalGeneration

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def enable_mamba_cpu_fallback():
    """
    Enables pure PyTorch CPU fallbacks for Mamba2, CausalConv1d, and SSD kernels
    when CUDA is unavailable or NVIDIA driver is offloaded.
    """
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
        print("[+] Enabled PyTorch CPU fallbacks for Mamba2, CausalConv1d, and SSD kernels.")
    except Exception as e:
        print(f"[!] Warning: Could not patch Mamba CPU fallback: {e}")


# Automatically enable fallback if CUDA is not available
if not torch.cuda.is_available():
    enable_mamba_cpu_fallback()

from TTSDataModule import TTSMODEL
from preprocessing.text import text_to_sequence
from stats import denormalize_latent, normalize_latent
from generate_audio import setup_dacvae, decode_latent_to_audio, synthesize_text


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Computes Word Error Rate (WER) using Levenshtein distance."""
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    r_len = len(ref_words)
    h_len = len(hyp_words)
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
                d[i, j] = min(
                    d[i - 1, j] + 1,      # deletion
                    d[i, j - 1] + 1,      # insertion
                    d[i - 1, j - 1] + 1,  # substitution
                )
    return float(d[r_len, h_len] / r_len)


def run_full_diagnosis(
    ckpt_path="TTS_checkpoints/mambaflow_ca_bimamba_256d_epoch_268_loss_0.9713.ckpt",
    output_dir="generated_audio",
    val_csv="/home/monesh/ljspeech/LJSpeech-1.1/val.csv",
    num_samples=5,
    temperature=1.0,
    n_steps=30,
    solver="euler",
):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 85)
    print("  MAMBAFLOW-TTS: TRAJECTORY MAPPING, RMS ENERGY & WHISPER ASR DIAGNOSTICS")
    print(f"  Checkpoint:   {ckpt_path}")
    print(f"  Temperature:  {temperature}")
    print(f"  Euler Steps:  {n_steps}")
    print(f"  Solver:       {solver}")
    print("=" * 85)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Loading model on {device}...")
    pl_module = TTSMODEL.load_from_checkpoint(ckpt_path, map_location=device)
    model = pl_module.model.to(device).eval()

    print("[+] Loading DACVAE decoder...")
    dacvae_decoder, sr, _ = setup_dacvae(device=device)

    print("[+] Loading Whisper ASR (tiny.en) on CPU...")
    whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").eval().to("cpu")
    resampler_16k = T.Resample(48000, 16000)

    df = pd.read_csv(val_csv)
    cleaner_names = ["english_cleaners2"]

    results = []
    all_trajectories = []

    for idx in range(num_samples):
        row = df.iloc[idx]
        audio_id = row["audio_id"]
        text = str(row.get("normalized_text", row.get("text", "")))
        latent_path = row["latent"]
        dur_list = ast.literal_eval(row["durations"]) if isinstance(row["durations"], str) else row["durations"]

        print(f"\n" + "-" * 75)
        print(f"[{idx+1}/{num_samples}] Analyzing Sample: {audio_id}")
        print(f"Text: \"{text}\"")

        # Load Ground Truth Latent & Waveform
        gt_raw_latent = np.squeeze(np.load(latent_path))  # [128, T] or [T, 128]
        if gt_raw_latent.shape[0] != 128 and gt_raw_latent.shape[1] == 128:
            gt_raw_latent = gt_raw_latent.T
        T_gt = gt_raw_latent.shape[1]

        # Normalized GT latent [1, T, 128]
        gt_tensor = torch.tensor(gt_raw_latent, dtype=torch.float32).unsqueeze(0).to(device)
        gt_norm = normalize_latent(gt_tensor).transpose(1, 2)  # [1, T, 128]

        # Decode GT Audio
        with torch.inference_mode():
            gt_wav = decode_latent_to_audio(dacvae_decoder, gt_tensor, target_device=device)

        # Encode Text & Durations
        seq = text_to_sequence(text, cleaner_names)[0]
        text_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        text_mask = torch.ones_like(text_tensor, dtype=torch.bool)
        durations = torch.tensor(dur_list, dtype=torch.long, device=device).unsqueeze(0)

        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            x_enc = model.txt_emb(text_tensor)
            enc_out = model.encoder(x_enc, mask=text_mask)

            # Trajectory Tracking with Euler N steps
            dt = 1.0 / n_steps
            torch.manual_seed(42 + idx)
            # Sample initial Gaussian noise scaled by temperature
            x_0 = torch.randn_like(gt_norm) * float(temperature)
            x_curr = x_0.clone()

            alphas = []
            betas = []
            cos_sims = []
            l2_dists = []
            rms_vals = []

            for step in range(n_steps):
                t_val = step * dt
                t_tensor = torch.full((1,), t_val, device=device, dtype=torch.float32)

                # Measure current state metrics relative to target and noise
                x_flat = x_curr.float().flatten()
                gt_flat = gt_norm.float().flatten()
                x0_flat = x_0.float().flatten()

                # Projection onto GT target and initial noise basis
                A = torch.stack([gt_flat, x0_flat], dim=-1)
                sol = torch.linalg.lstsq(A, x_flat).solution
                alpha_val = sol[0].item()
                beta_val = sol[1].item()

                cos_sim = torch.cosine_similarity(x_flat.unsqueeze(0), gt_flat.unsqueeze(0)).item()
                l2_dist = torch.norm(x_flat - gt_flat).item()
                rms = torch.sqrt(torch.mean(x_flat ** 2)).item()

                alphas.append(alpha_val)
                betas.append(beta_val)
                cos_sims.append(cos_sim)
                l2_dists.append(l2_dist)
                rms_vals.append(rms)

                # Velocity step
                v = model.decoder.predict_velocity(
                    x=x_curr,
                    t=t_tensor,
                    text_repr=enc_out,
                    durations=durations,
                    text_mask=text_mask,
                )

                if solver == "midpoint" and step < n_steps - 1:
                    x_mid = x_curr + (dt / 2.0) * v
                    t_mid = torch.full((1,), t_val + dt / 2.0, device=device, dtype=torch.float32)
                    v_mid = model.decoder.predict_velocity(
                        x=x_mid,
                        t=t_mid,
                        text_repr=enc_out,
                        durations=durations,
                        text_mask=text_mask,
                    )
                    x_curr = x_curr + dt * v_mid
                else:
                    x_curr = x_curr + dt * v

            # Final state t = 1.0
            x_1 = x_curr
            x_flat = x_1.float().flatten()
            A = torch.stack([gt_flat, x0_flat], dim=-1)
            sol = torch.linalg.lstsq(A, x_flat).solution
            alpha_final = sol[0].item()
            beta_final = sol[1].item()
            cos_final = torch.cosine_similarity(x_flat.unsqueeze(0), gt_flat.unsqueeze(0)).item()
            l2_final = torch.norm(x_flat - gt_flat).item()
            rms_final = torch.sqrt(torch.mean(x_flat ** 2)).item()
            alphas.append(alpha_final)
            betas.append(beta_final)
            cos_sims.append(cos_final)
            l2_dists.append(l2_final)
            rms_vals.append(rms_final)

            # Denormalization to DACAE continuous latent space
            pred_latent_denorm = denormalize_latent(x_1.float())
            pred_latent_128_t = pred_latent_denorm.transpose(1, 2)
            pred_wav = decode_latent_to_audio(dacvae_decoder, pred_latent_128_t, target_device=device)

        # -------------------------------------------------------------
        # Energy & RMS Analysis: Normalized, Denormalized, and Audio
        # -------------------------------------------------------------
        # 1. Normalized space (Z-score standardized)
        norm_gt_rms = torch.sqrt(torch.mean(gt_norm.float() ** 2)).item()
        norm_pred_rms = rms_final
        norm_gt_std = gt_norm.float().std().item()
        norm_pred_std = x_1.float().std().item()

        # 2. Denormalized latent space (Raw DACAE)
        raw_gt_rms = torch.sqrt(torch.mean(gt_tensor.float() ** 2)).item()
        raw_pred_rms = torch.sqrt(torch.mean(pred_latent_denorm.float() ** 2)).item()
        raw_gt_std = gt_tensor.float().std().item()
        raw_pred_std = pred_latent_denorm.float().std().item()

        # 3. 48kHz Decoded Audio
        audio_gt_rms = float(np.sqrt(np.mean(gt_wav ** 2)))
        audio_pred_rms = float(np.sqrt(np.mean(pred_wav ** 2)))
        audio_gt_peak = float(np.max(np.abs(gt_wav)))
        audio_pred_peak = float(np.max(np.abs(pred_wav)))

        audio_rms_ratio = (audio_pred_rms / audio_gt_rms) if audio_gt_rms > 0 else 0
        raw_rms_ratio = (raw_pred_rms / raw_gt_rms) if raw_gt_rms > 0 else 0

        # Frame-level Energy Profile (Audio broken into 20ms frames)
        frame_len = int(48000 * 0.02)
        n_frames = min(len(gt_wav), len(pred_wav)) // frame_len
        gt_frame_energy = [float(np.sqrt(np.mean(gt_wav[f*frame_len:(f+1)*frame_len]**2))) for f in range(n_frames)]
        pred_frame_energy = [float(np.sqrt(np.mean(pred_wav[f*frame_len:(f+1)*frame_len]**2))) for f in range(n_frames)]

        # -------------------------------------------------------------
        # Whisper ASR Intelligibility & WER Testing
        # -------------------------------------------------------------
        gt_wav_16k = resampler_16k(torch.tensor(gt_wav).float().unsqueeze(0)).squeeze().numpy()
        pred_wav_16k = resampler_16k(torch.tensor(pred_wav).float().unsqueeze(0)).squeeze().numpy()

        with torch.no_grad():
            gt_feats = whisper_proc(gt_wav_16k, sampling_rate=16000, return_tensors="pt").input_features
            gt_ids = whisper_model.generate(gt_feats)
            gt_transcript = whisper_proc.batch_decode(gt_ids, skip_special_tokens=True)[0].strip()

            pred_feats = whisper_proc(pred_wav_16k, sampling_rate=16000, return_tensors="pt").input_features
            pred_ids = whisper_model.generate(pred_feats)
            pred_transcript = whisper_proc.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

        wer_vs_ref = calculate_wer(text, pred_transcript)
        wer_vs_gt = calculate_wer(gt_transcript, pred_transcript)

        print(f"  [Trajectory End]: Alpha(Target)={alpha_final:.3f} | Beta(Noise)={beta_final:.3f} | CosSim={cos_final:.3f} | L2={l2_final:.3f}")
        print(f"  [Norm Latent RMS]: Pred={norm_pred_rms:.3f} vs GT={norm_gt_rms:.3f} (Std: {norm_pred_std:.3f} vs {norm_gt_std:.3f})")
        print(f"  [Raw Latent RMS]:  Pred={raw_pred_rms:.3f} vs GT={raw_gt_rms:.3f} (Ratio: {raw_rms_ratio*100:.1f}%)")
        print(f"  [Audio RMS]:       Pred={audio_pred_rms:.4f} vs GT={audio_gt_rms:.4f} (Ratio: {audio_rms_ratio*100:.1f}%)")
        print(f"  [Audio Peak]:      Pred={audio_pred_peak:.4f} vs GT={audio_gt_peak:.4f}")
        print(f"  [Whisper GT]:      \"{gt_transcript}\"")
        print(f"  [Whisper Pred]:    \"{pred_transcript}\"")
        print(f"  [WER (vs Text)]:   {wer_vs_ref*100:.1f}% | [WER (vs GT Audio)]: {wer_vs_gt*100:.1f}%")

        # Save audio files
        temp_str = f"temp{temperature:.1f}"
        pred_path = os.path.join(output_dir, f"diag_{temp_str}_{idx+1:02d}_{audio_id}_pred.wav")
        gt_path = os.path.join(output_dir, f"diag_{temp_str}_{idx+1:02d}_{audio_id}_gt.wav")
        sf.write(pred_path, pred_wav, sr)
        sf.write(gt_path, gt_wav, sr)

        # Also save standard filenames for convenience
        sf.write(os.path.join(output_dir, f"diag_{idx+1:02d}_{audio_id}_pred.wav"), pred_wav, sr)
        sf.write(os.path.join(output_dir, f"diag_{idx+1:02d}_{audio_id}_gt.wav"), gt_wav, sr)

        all_trajectories.append({
            "audio_id": audio_id,
            "alphas": alphas,
            "betas": betas,
            "cos_sims": cos_sims,
            "l2_dists": l2_dists,
            "rms_vals": rms_vals,
            "gt_energy": gt_frame_energy,
            "pred_energy": pred_frame_energy,
        })

        results.append({
            "sample_index": idx + 1,
            "audio_id": audio_id,
            "temperature": temperature,
            "n_steps": n_steps,
            "text": text,
            "whisper_gt": gt_transcript,
            "whisper_pred": pred_transcript,
            "wer_vs_ref_pct": round(wer_vs_ref * 100, 1),
            "wer_vs_gt_pct": round(wer_vs_gt * 100, 1),
            "trajectory_target_alpha": round(alpha_final, 3),
            "trajectory_noise_beta": round(beta_final, 3),
            "final_cosine_sim": round(cos_final, 3),
            "final_l2_dist": round(l2_final, 3),
            "norm_pred_rms": round(norm_pred_rms, 3),
            "norm_gt_rms": round(norm_gt_rms, 3),
            "raw_pred_rms": round(raw_pred_rms, 3),
            "raw_gt_rms": round(raw_gt_rms, 3),
            "raw_rms_pct": round(raw_rms_ratio * 100, 1),
            "audio_pred_rms": round(audio_pred_rms, 4),
            "audio_gt_rms": round(audio_gt_rms, 4),
            "audio_rms_pct": round(audio_rms_ratio * 100, 1),
            "audio_pred_peak": round(audio_pred_peak, 4),
            "audio_gt_peak": round(audio_gt_peak, 4),
        })

    # -------------------------------------------------------------
    # Plot Trajectory & Energy Graph (6-Panel Comprehensive Layout)
    # -------------------------------------------------------------
    print("\n[+] Plotting Trajectory and Energy Graphs (6-Panel)...")
    fig, axs = plt.subplots(2, 3, figsize=(18, 11))

    timesteps = np.linspace(0.0, 1.0, n_steps + 1)
    t1 = all_trajectories[0]

    # Subplot 1: Target Alpha vs Noise Beta across Steps (Sample 1)
    axs[0, 0].plot(timesteps, t1["alphas"], "b-o", label=r"Target Component $\alpha(t)$ (Ideal $\to$ 1.0)", linewidth=2)
    axs[0, 0].plot(timesteps, t1["betas"], "r--s", label=r"Noise Residual $\beta(t)$ (Ideal $\to$ 0.0)", linewidth=2)
    axs[0, 0].axhline(1.0, color="green", linestyle=":", label="Ideal Target (1.0)")
    axs[0, 0].axhline(0.0, color="gray", linestyle=":", label="Ideal Noise (0.0)")
    axs[0, 0].set_title(f"Component Evolution (Temp={temperature}): {t1['audio_id']}", fontsize=11, fontweight="bold")
    axs[0, 0].set_xlabel("Flow Timestep t (0 = Noise, 1 = Final)")
    axs[0, 0].set_ylabel("Basis Projection Weight")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=9)

    # Subplot 2: Trajectory Phase Plane (Noise Projection vs Target Projection)
    for traj in all_trajectories:
        axs[0, 1].plot(traj["alphas"], traj["betas"], "-o", label=traj["audio_id"], markersize=4)
    axs[0, 1].scatter([1.0], [0.0], color="green", s=150, zorder=5, marker="*", label="Target Destination (1, 0)")
    axs[0, 1].scatter([0.0], [temperature], color="red", s=100, zorder=5, marker="X", label=f"Start Noise (0, {temperature:.1f})")
    axs[0, 1].set_title(f"Phase Plane Trajectory (Temp={temperature})", fontsize=11, fontweight="bold")
    axs[0, 1].set_xlabel("Projection on Target Latent (Ideal = 1.0)")
    axs[0, 1].set_ylabel(r"Projection on Initial Noise (Ideal = 0.0)")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=9)

    # Subplot 3: Distance Trajectory (L2 Distance to Ground Truth Target over Steps)
    for traj in all_trajectories:
        axs[0, 2].plot(timesteps, traj["l2_dists"], "-^", label=traj["audio_id"], markersize=4)
    axs[0, 2].set_title(r"Distance Trajectory ($L_2$ Distance to Target)", fontsize=11, fontweight="bold")
    axs[0, 2].set_xlabel("Flow Timestep t")
    axs[0, 2].set_ylabel(r"Euclidean Distance $\|x_t - x_{gt}\|_2$")
    axs[0, 2].grid(True, alpha=0.3)
    axs[0, 2].legend(fontsize=9)

    # Subplot 4: Cosine Similarity Progression across Timesteps
    for traj in all_trajectories:
        axs[1, 0].plot(timesteps, traj["cos_sims"], "-d", label=traj["audio_id"], markersize=4)
    axs[1, 0].set_title(f"Cosine Similarity to Ground Truth over Time", fontsize=11, fontweight="bold")
    axs[1, 0].set_xlabel("Flow Timestep t")
    axs[1, 0].set_ylabel("Cosine Similarity")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=9)

    # Subplot 5: Latent RMS Energy Evolution across Timesteps
    for traj in all_trajectories:
        axs[1, 1].plot(timesteps, traj["rms_vals"], "-s", label=traj["audio_id"], markersize=4)
    axs[1, 1].axhline(1.0, color="gray", linestyle=":", label="Unit RMS Reference (1.0)")
    axs[1, 1].set_title(f"Normalized Latent RMS Energy Evolution", fontsize=11, fontweight="bold")
    axs[1, 1].set_xlabel("Flow Timestep t")
    axs[1, 1].set_ylabel(r"Latent RMS $\sqrt{E[x_t^2]}$")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend(fontsize=9)

    # Subplot 6: Frame-by-Frame Audio Energy Profile (Sample 2)
    t2 = all_trajectories[1]
    time_axis = np.arange(len(t2["gt_energy"])) * 0.02
    axs[1, 2].plot(time_axis, t2["gt_energy"], "g-", label="Ground Truth Audio Energy", linewidth=1.5)
    axs[1, 2].plot(time_axis, t2["pred_energy"], "m--", label=f"Pred Audio Energy (Temp={temperature})", linewidth=1.5)
    axs[1, 2].set_title(f"Audio Frame Energy Envelope (20ms frames): {t2['audio_id']}", fontsize=11, fontweight="bold")
    axs[1, 2].set_xlabel("Time (seconds)")
    axs[1, 2].set_ylabel("RMS Energy")
    axs[1, 2].grid(True, alpha=0.3)
    axs[1, 2].legend(fontsize=9)

    plt.tight_layout()
    plot_path_temp = os.path.join(output_dir, f"trajectory_and_energy_map_temp{temperature:.1f}.png")
    plot_path_std = os.path.join(output_dir, "trajectory_and_energy_map.png")
    plt.savefig(plot_path_temp, dpi=180)
    plt.savefig(plot_path_std, dpi=180)
    plt.close()
    print(f"[+] Saved trajectory plots to:\n    - {plot_path_temp}\n    - {plot_path_std}")

    # Save summary JSON
    summary_path_temp = os.path.join(output_dir, f"trajectory_diagnosis_results_temp{temperature:.1f}.json")
    summary_path_std = os.path.join(output_dir, "trajectory_diagnosis_results.json")
    with open(summary_path_temp, "w") as f:
        json.dump(results, f, indent=2)
    with open(summary_path_std, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved diagnostic results to:\n    - {summary_path_temp}\n    - {summary_path_std}")

    # Print summary table
    print("\n" + "=" * 95)
    print(f"  DIAGNOSTIC SUMMARY TABLE (Temperature={temperature}, Steps={n_steps}, Solver={solver})")
    print("=" * 95)
    print(f"{'#':<2} | {'Audio ID':<11} | {'Alpha(T)':<8} | {'Beta(N)':<8} | {'CosSim':<7} | {'L2 Dist':<8} | {'Raw RMS%':<9} | {'Audio RMS%':<10} | {'WER(Ref)':<8}")
    print("-" * 95)
    for r in results:
        print(f"{r['sample_index']:<2} | {r['audio_id']:<11} | {r['trajectory_target_alpha']:<8.3f} | {r['trajectory_noise_beta']:<8.3f} | {r['final_cosine_sim']:<7.3f} | {r['final_l2_dist']:<8.3f} | {r['raw_rms_pct']:<8.1f}% | {r['audio_rms_pct']:<9.1f}% | {r['wer_vs_ref_pct']:<7.1f}%")
    print("=" * 95)

    # -------------------------------------------------------------
    # Synthesize Custom Speech Prompt at temp=1.0, euler=30
    # -------------------------------------------------------------
    custom_prompt = "There was also a good supply of Bibles and prayer books in the library."
    print(f"\n[+] Synthesizing Custom Prompt with Temp={temperature}, Steps={n_steps}: \"{custom_prompt}\"")
    custom_wav, n_frames, t_infer = synthesize_text(
        model=model,
        text=custom_prompt,
        dacvae_decoder=dacvae_decoder,
        device=device,
        timesteps=n_steps,
        temperature=temperature,
        solver=solver,
    )
    custom_audio_path = os.path.join(output_dir, f"custom_prompt_temp{temperature:.1f}_euler{n_steps}.wav")
    sf.write(custom_audio_path, custom_wav, sr)
    custom_rms = float(np.sqrt(np.mean(custom_wav ** 2)))
    custom_peak = float(np.max(np.abs(custom_wav)))

    # Transcribe custom prompt with Whisper
    custom_wav_16k = resampler_16k(torch.tensor(custom_wav).float().unsqueeze(0)).squeeze().numpy()
    with torch.no_grad():
        c_feats = whisper_proc(custom_wav_16k, sampling_rate=16000, return_tensors="pt").input_features
        c_ids = whisper_model.generate(c_feats)
        custom_transcript = whisper_proc.batch_decode(c_ids, skip_special_tokens=True)[0].strip()

    print(f"  [Custom Audio Saved]: {custom_audio_path}")
    print(f"  [Custom Audio RMS]:   {custom_rms:.4f} | Peak: {custom_peak:.4f} | Frames: {n_frames} | Time: {t_infer:.2f}s")
    print(f"  [Custom Whisper]:     \"{custom_transcript}\"")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MambaFlow Trajectory, RMS and Whisper ASR Tester")
    parser.add_argument("--ckpt", type=str, default="TTS_checkpoints/mambaflow_ca_bimamba_256d_epoch_268_loss_0.9713.ckpt")
    parser.add_argument("--temp", "--temperature", dest="temperature", type=float, default=1.0, help="Sampling noise temperature (default: 1.0)")
    parser.add_argument("--steps", "--n_steps", dest="n_steps", type=int, default=30, help="Euler integration steps (default: 30)")
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "midpoint"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="generated_audio")
    parser.add_argument("--val_csv", type=str, default="/home/monesh/ljspeech/LJSpeech-1.1/val.csv")
    args = parser.parse_args()

    run_full_diagnosis(
        ckpt_path=args.ckpt,
        output_dir=args.output_dir,
        val_csv=args.val_csv,
        num_samples=args.num_samples,
        temperature=args.temperature,
        n_steps=args.n_steps,
        solver=args.solver,
    )
