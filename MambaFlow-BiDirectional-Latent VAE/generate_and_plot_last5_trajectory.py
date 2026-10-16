import os
import sys
import time
import json
import ast
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from TTSDataModule import TTSMODEL
from preprocessing.text import text_to_sequence
from stats import denormalize_latent, normalize_latent
from generate_audio import setup_dacvae, decode_latent_to_audio, find_latest_checkpoint, synthesize_text


def compute_velocity_trajectory_for_sample(
    model,
    dacvae_decoder,
    sample_row,
    sample_index,
    device="cuda",
    dacvae_device="cpu",
    n_steps=30,
    temperature=1.0,
    output_dir="last_5_val_evaluation",
    sr=48000,
):
    """
    Computes Euler integration trajectory, tracks velocity direction relative to ground-truth destination,
    and synthesizes 48kHz audio for predicted and ground-truth latents.
    """
    audio_id = sample_row["audio_id"]
    text = str(sample_row.get("normalized_text", sample_row.get("text", "")))
    latent_path = sample_row["latent"]
    dur_list = ast.literal_eval(sample_row["durations"]) if isinstance(sample_row["durations"], str) else sample_row["durations"]

    print(f"\n{'='*80}")
    print(f"  [Sample {sample_index}/5] Audio ID: {audio_id}")
    print(f"  Text: \"{text}\"")
    print(f"  Frames: {sum(dur_list)} | Phonemes: {len(dur_list)}")
    print(f"{'='*80}")

    # 1. Load Ground Truth Latent & Normalize
    gt_raw = np.squeeze(np.load(latent_path))  # [128, T] or [T, 128]
    if gt_raw.ndim == 2:
        if gt_raw.shape[0] != 128 and gt_raw.shape[1] == 128:
            gt_raw = gt_raw.T
    T_gt = gt_raw.shape[1]

    gt_tensor = torch.tensor(gt_raw, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 128, T]
    # gt_norm: normalized ground truth latent [1, T, 128] (The DESTINATION it needed to go)
    gt_norm = normalize_latent(gt_tensor).transpose(1, 2)  # [1, T, 128]

    # Decode Ground Truth Audio
    print(f"[+] Decoding Ground Truth Audio ({T_gt} frames)...")
    gt_wav = decode_latent_to_audio(dacvae_decoder, gt_tensor, target_device=dacvae_device)
    gt_wav_path = os.path.join(output_dir, f"val_last5_s{sample_index}_{audio_id}_gt.wav")
    sf.write(gt_wav_path, gt_wav, sr)

    # 2. Text Encoding & Duration Tensor Setup
    cleaner_names = ["english_cleaners2"]
    seq = text_to_sequence(text, cleaner_names)[0]
    text_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    text_mask = torch.ones_like(text_tensor, dtype=torch.bool)
    durations = torch.tensor(dur_list, dtype=torch.long, device=device).unsqueeze(0)

    # Ensure total durations match T_gt
    dur_sum = int(durations.sum().item())
    if dur_sum != T_gt:
        diff = T_gt - dur_sum
        durations[0, -1] = max(1, durations[0, -1] + diff)

    audio_lengths = durations.sum(dim=-1)
    audio_mask = torch.arange(T_gt, device=device)[None, :] < audio_lengths[:, None]

    # 3. Euler 30-Step Trajectory Tracking with Velocity Recording
    dt = 1.0 / n_steps
    torch.manual_seed(100 + sample_index)

    # Initial Gaussian noise scaled by temperature = 1.0
    x_0 = torch.randn_like(gt_norm) * float(temperature)
    x_curr = x_0.clone()

    trajectory_states = [x_curr.clone()]  # list of [1, T, 128]
    velocity_vectors = []                 # list of [1, T, 128]
    timesteps_list = [0.0]

    cos_sim_to_dest = []    # cos_sim(v_t, x_gt - x_t) -> "Which direction velocity points to destination"
    angle_to_dest = []      # angle in degrees between v_t and (x_gt - x_t)
    cos_sim_to_ot = []      # cos_sim(v_t, x_gt - x_0) -> alignment with constant OT velocity
    dist_to_dest = []       # ||x_t - x_gt||_2
    rmse_to_dest = []       # sqrt(mean((x_t - x_gt)^2))
    vel_norms = []          # ||v_t||_2
    ideal_speeds = []       # ||x_gt - x_t|| / (1 - t)
    alphas = []             # projection weight on GT target
    betas = []              # projection weight on initial noise
    rms_vals = []           # sqrt(mean(x_t^2))

    # Helper function to compute projection on basis {gt_flat, x0_flat}
    gt_flat = gt_norm.float().flatten()
    x0_flat = x_0.float().flatten()
    A_basis = torch.stack([gt_flat, x0_flat], dim=-1)  # [Dim, 2]

    # Initial state metrics at t=0
    x_flat_0 = x_curr.float().flatten()
    sol_0 = torch.linalg.lstsq(A_basis, x_flat_0).solution
    alphas.append(sol_0[0].item())
    betas.append(sol_0[1].item())
    initial_l2 = torch.norm(x_flat_0 - gt_flat).item()
    dist_to_dest.append(initial_l2)
    rmse_to_dest.append(torch.sqrt(torch.mean((x_flat_0 - gt_flat) ** 2)).item())
    rms_vals.append(torch.sqrt(torch.mean(x_flat_0 ** 2)).item())

    t_start = time.time()
    with torch.inference_mode():
        x_enc = model.txt_emb(text_tensor)
        enc_out = model.encoder(x_enc, mask=text_mask)

        for step in range(n_steps):
            t_val = step * dt
            t_tensor = torch.full((1,), t_val, device=device, dtype=torch.float32)

            # Predict Velocity v_t at current state x_t and time t
            v = model.decoder.predict_velocity(
                x=x_curr,
                t=t_tensor,
                text_repr=enc_out,
                durations=durations,
                text_mask=text_mask,
                audio_mask=audio_mask,
            )

            # Record velocity
            velocity_vectors.append(v.clone())

            # Direction from current state to destination
            dest_dir = gt_norm - x_curr  # vector pointing straight to target
            ot_dir = gt_norm - x_0       # global straight vector from noise to target

            v_flat = v.float().flatten()
            dest_flat = dest_dir.float().flatten()
            ot_flat = ot_dir.float().flatten()

            v_norm = torch.norm(v_flat).item()
            dest_norm = torch.norm(dest_flat).item()
            ot_norm = torch.norm(ot_flat).item()

            # Cosine similarity between velocity v_t and destination direction
            if v_norm > 1e-7 and dest_norm > 1e-7:
                cos_dest = torch.dot(v_flat, dest_flat).item() / (v_norm * dest_norm)
                cos_dest = float(np.clip(cos_dest, -1.0, 1.0))
                angle_deg = float(np.arccos(cos_dest) * 180.0 / np.pi)
            else:
                cos_dest = 0.0
                angle_deg = 90.0

            if v_norm > 1e-7 and ot_norm > 1e-7:
                cos_ot = torch.dot(v_flat, ot_flat).item() / (v_norm * ot_norm)
                cos_ot = float(np.clip(cos_ot, -1.0, 1.0))
            else:
                cos_ot = 0.0

            ideal_speed = dest_norm / max(1.0 - t_val, 1e-4)

            cos_sim_to_dest.append(cos_dest)
            angle_to_dest.append(angle_deg)
            cos_sim_to_ot.append(cos_ot)
            vel_norms.append(v_norm)
            ideal_speeds.append(ideal_speed)

            # Euler step: x_{t+dt} = x_t + dt * v_t
            x_curr = x_curr + dt * v

            if audio_mask is not None:
                x_curr = x_curr * audio_mask.unsqueeze(-1).float()

            trajectory_states.append(x_curr.clone())
            timesteps_list.append((step + 1) * dt)

            # Metrics for new state
            x_flat = x_curr.float().flatten()
            sol = torch.linalg.lstsq(A_basis, x_flat).solution
            alphas.append(sol[0].item())
            betas.append(sol[1].item())
            l2_d = torch.norm(x_flat - gt_flat).item()
            dist_to_dest.append(l2_d)
            rmse_to_dest.append(torch.sqrt(torch.mean((x_flat - gt_flat) ** 2)).item())
            rms_vals.append(torch.sqrt(torch.mean(x_flat ** 2)).item())

    total_gen_time = time.time() - t_start

    # Final state at t = 1.0
    x_1 = trajectory_states[-1]
    final_l2 = dist_to_dest[-1]
    final_rmse = rmse_to_dest[-1]
    final_alpha = alphas[-1]
    final_beta = betas[-1]

    # Decode Predicted Audio from Final Latent x_1
    pred_latent_denorm = denormalize_latent(x_1.float())
    pred_latent_128_t = pred_latent_denorm.transpose(1, 2)
    print(f"[+] Decoding Predicted Audio from Trajectory End ({pred_latent_128_t.shape[-1]} frames)...")
    pred_wav = decode_latent_to_audio(dacvae_decoder, pred_latent_128_t, target_device=dacvae_device)
    pred_wav_path = os.path.join(output_dir, f"val_last5_s{sample_index}_{audio_id}_pred.wav")
    sf.write(pred_wav_path, pred_wav, sr)

    # Also synthesize full TTS audio with predicted durations for complete demonstration
    print(f"[+] Synthesizing full TTS audio with predicted durations...")
    tts_wav, tts_frames, _ = synthesize_text(
        model=model,
        text=text,
        dacvae_decoder=dacvae_decoder,
        device=device,
        timesteps=n_steps,
        temperature=temperature,
        solver="euler",
    )
    tts_wav_path = os.path.join(output_dir, f"val_last5_s{sample_index}_{audio_id}_tts_pred_dur.wav")
    sf.write(tts_wav_path, tts_wav, sr)

    pred_dur = len(pred_wav) / sr
    gt_dur = len(gt_wav) / sr
    tts_dur = len(tts_wav) / sr

    pred_rms = float(np.sqrt(np.mean(pred_wav ** 2)))
    gt_rms = float(np.sqrt(np.mean(gt_wav ** 2)))
    rms_ratio = (pred_rms / gt_rms * 100.0) if gt_rms > 0 else 0.0

    print(f"  [Trajectory End]: Alpha(Target)={final_alpha:.3f} | Beta(Noise)={final_beta:.3f} | Final L2={final_l2:.2f} | Final RMSE={final_rmse:.4f}")
    print(f"  [Average Velocity Alignment to Destination]: {np.mean(cos_sim_to_dest):.3f} (Angle: {np.mean(angle_to_dest):.1f}°)")
    print(f"  [Audio Duration]: Pred={pred_dur:.2f}s | GT={gt_dur:.2f}s | TTS(PredDur)={tts_dur:.2f}s")
    print(f"  [Audio RMS]:      Pred={pred_rms:.4f} | GT={gt_rms:.4f} (Ratio: {rms_ratio:.1f}%)")
    print(f"  [Saved Files]:")
    print(f"    - Pred Audio: {pred_wav_path}")
    print(f"    - GT Audio:   {gt_wav_path}")
    print(f"    - TTS Audio:  {tts_wav_path}")

    # 4. Perform 2D PCA on [x_0, x_t1, ..., x_t30, x_gt]
    traj_matrix = torch.stack([s.float().flatten() for s in trajectory_states] + [gt_flat], dim=0).cpu().numpy()
    pca = PCA(n_components=2)
    traj_pca = pca.fit_transform(traj_matrix)
    pca_steps = traj_pca[:-1]       # 31 steps (0 to 30)
    pca_gt = traj_pca[-1]           # Ground truth destination

    sample_data = {
        "sample_index": sample_index,
        "audio_id": audio_id,
        "text": text,
        "T_frames": T_gt,
        "durations": dur_list,
        "timesteps": timesteps_list,
        "alphas": alphas,
        "betas": betas,
        "dist_to_dest": dist_to_dest,
        "rmse_to_dest": rmse_to_dest,
        "cos_sim_to_dest": cos_sim_to_dest,
        "angle_to_dest": angle_to_dest,
        "cos_sim_to_ot": cos_sim_to_ot,
        "vel_norms": vel_norms,
        "ideal_speeds": ideal_speeds,
        "rms_vals": rms_vals,
        "pca_steps_x": pca_steps[:, 0].tolist(),
        "pca_steps_y": pca_steps[:, 1].tolist(),
        "pca_gt_x": float(pca_gt[0]),
        "pca_gt_y": float(pca_gt[1]),
        "pca_explained_var": pca.explained_variance_ratio_.tolist(),
        "pred_wav_path": pred_wav_path,
        "gt_wav_path": gt_wav_path,
        "tts_wav_path": tts_wav_path,
        "pred_dur": pred_dur,
        "gt_dur": gt_dur,
        "tts_dur": tts_dur,
        "pred_rms": pred_rms,
        "gt_rms": gt_rms,
        "rms_ratio": rms_ratio,
        "final_alpha": final_alpha,
        "final_beta": final_beta,
        "final_l2": final_l2,
        "final_rmse": final_rmse,
        "mean_cos_dest": float(np.mean(cos_sim_to_dest)),
        "mean_angle_dest": float(np.mean(angle_to_dest)),
    }

    return sample_data


def plot_single_sample_deep_dive(sample_data, output_dir="last_5_val_evaluation"):
    """
    Plots a comprehensive 6-panel visualization for a single sample showing:
    1. Phase-plane trajectory (Alpha vs Beta) with velocity direction arrows, Start, End, and Destination Target.
    2. PCA 2D trajectory with velocity direction quiver arrows, Start, End, and Destination Target.
    3. Instantaneous velocity alignment to destination: cos(v_t, x_gt - x_t) across the 30 Euler steps.
    4. Distance to destination ||x_t - x_gt|| across the 30 Euler steps.
    5. Component evolution: Target alpha(t) vs Noise residual beta(t).
    6. Velocity magnitude ||v_t|| vs step.
    """
    s_idx = sample_data["sample_index"]
    aid = sample_data["audio_id"]
    text_short = sample_data["text"][:60] + "..." if len(sample_data["text"]) > 60 else sample_data["text"]

    fig, axs = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle(
        f"MambaFlow-TTS Velocity Trajectory Analysis: Sample {s_idx} ({aid})\n"
        f"Text: \"{text_short}\" | Euler 30 Steps | Temp = 1.0",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    t_steps = np.linspace(0.0, 1.0, len(sample_data["alphas"]))
    t_vel = np.linspace(0.0, 1.0 - 1.0 / 30, len(sample_data["cos_sim_to_dest"]))

    # -------------------------------------------------------------
    # Panel 1: Phase Plane (Projection on Target x_gt vs Noise x_0)
    # -------------------------------------------------------------
    ax = axs[0, 0]
    alphas = np.array(sample_data["alphas"])
    betas = np.array(sample_data["betas"])

    # Ideal direct path: (0, 1) -> (1, 0)
    ax.plot([0.0, 1.0], [1.0, 0.0], "k--", alpha=0.4, linewidth=2, label="Ideal Straight Flow Path")

    # Actual trajectory curve
    ax.plot(alphas, betas, color="#1f77b4", linewidth=2.5, label="Model Trajectory Path", zorder=3)

    # Velocity direction arrows along the curve (every 3 steps)
    for i in range(0, len(alphas) - 1, 3):
        dx = alphas[i + 1] - alphas[i]
        dy = betas[i + 1] - betas[i]
        scale = np.sqrt(dx ** 2 + dy ** 2) + 1e-8
        ax.annotate(
            "",
            xy=(alphas[i + 1], betas[i + 1]),
            xytext=(alphas[i], betas[i]),
            arrowprops=dict(arrowstyle="->", color="#0055d4", lw=1.8, shrinkA=0, shrinkB=0),
            zorder=4,
        )

    # Start Point
    ax.scatter([alphas[0]], [betas[0]], color="#d62728", s=130, marker="X", zorder=6, label=f"Start Noise x_0 (t=0): (0, 1.0)")
    # Destination Point it needed to reach
    ax.scatter([1.0], [0.0], color="#2ca02c", s=220, marker="*", zorder=6, label=f"Destination Target x_gt: (1.0, 0.0)")
    # End Point
    ax.scatter([alphas[-1]], [betas[-1]], color="#ff7f0e", s=140, marker="o", edgecolors="black", zorder=6,
               label=f"Trajectory End x_1 (t=1.0): ({alphas[-1]:.2f}, {betas[-1]:.2f})")

    # Residual error vector
    ax.plot([alphas[-1], 1.0], [betas[-1], 0.0], "r:", linewidth=2, label=f"Residual Gap (L2={sample_data['final_l2']:.1f})")

    ax.set_title("1. Phase-Plane Trajectory & Velocity Direction", fontsize=11, fontweight="bold")
    ax.set_xlabel("Projection on Destination Target x_gt (Ideal -> 1.0)")
    ax.set_ylabel("Projection on Noise x_0 (Ideal -> 0.0)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # -------------------------------------------------------------
    # Panel 2: 2D PCA Trajectory & Velocity Vector Field
    # -------------------------------------------------------------
    ax = axs[0, 1]
    px = np.array(sample_data["pca_steps_x"])
    py = np.array(sample_data["pca_steps_y"])
    gt_px = sample_data["pca_gt_x"]
    gt_py = sample_data["pca_gt_y"]

    # Trajectory path in PCA
    ax.plot(px, py, color="#9467bd", linewidth=2.5, label="Trajectory in PCA", zorder=3)
    ax.plot([px[0], gt_px], [py[0], gt_py], "k--", alpha=0.4, linewidth=2, label="Ideal Vector (x_0 -> x_gt)")

    # Velocity arrows in PCA
    for i in range(0, len(px) - 1, 3):
        ax.annotate(
            "",
            xy=(px[i + 1], py[i + 1]),
            xytext=(px[i], py[i]),
            arrowprops=dict(arrowstyle="->", color="#6a3d9a", lw=1.8, shrinkA=0, shrinkB=0),
            zorder=4,
        )

    # Start, End, Destination
    ax.scatter([px[0]], [py[0]], color="#d62728", s=130, marker="X", zorder=6, label=f"Start Noise x_0")
    ax.scatter([gt_px], [gt_py], color="#2ca02c", s=220, marker="*", zorder=6, label=f"Destination Target x_gt")
    ax.scatter([px[-1]], [py[-1]], color="#ff7f0e", s=140, marker="o", edgecolors="black", zorder=6, label=f"Trajectory End x_1")
    ax.plot([px[-1], gt_px], [py[-1], gt_py], "r:", linewidth=2, label="Residual Gap")

    ev = sample_data["pca_explained_var"]
    ax.set_title(f"2. 2D PCA Trajectory (PC1: {ev[0]*100:.1f}%, PC2: {ev[1]*100:.1f}%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # -------------------------------------------------------------
    # Panel 3: Velocity Direction Alignment to Destination vs Step
    # -------------------------------------------------------------
    ax = axs[0, 2]
    cos_dest = np.array(sample_data["cos_sim_to_dest"])
    cos_ot = np.array(sample_data["cos_sim_to_ot"])
    steps_arr = np.arange(len(cos_dest))

    ax.plot(steps_arr, cos_dest, "b-o", linewidth=2, markersize=5, label=r"$\cos(v_t, x_{gt} - x_t)$ (Direction to Destination)")
    ax.plot(steps_arr, cos_ot, "g--s", linewidth=1.5, markersize=4, label=r"$\cos(v_t, x_{gt} - x_0)$ (Global OT Alignment)")
    ax.axhline(1.0, color="green", linestyle=":", label="Direct at Destination (+1.0)")
    ax.axhline(0.0, color="gray", linestyle=":", label="Perpendicular Drift (0.0)")

    # Fill positive alignment
    ax.fill_between(steps_arr, 0.0, cos_dest, where=(cos_dest >= 0), color="blue", alpha=0.15, label="Positive Target Heading")

    ax.set_title("3. Instantaneous Velocity Direction toward Destination", fontsize=11, fontweight="bold")
    ax.set_xlabel("Euler Step k (0 -> 29)")
    ax.set_ylabel("Cosine Similarity (Direction Alignment)")
    ax.set_ylim(-0.2, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # -------------------------------------------------------------
    # Panel 4: Distance to Destination Target ||x_t - x_gt||
    # -------------------------------------------------------------
    ax = axs[1, 0]
    dist_arr = np.array(sample_data["dist_to_dest"])
    rmse_arr = np.array(sample_data["rmse_to_dest"])

    ax.plot(t_steps, dist_arr, "r-^", linewidth=2, markersize=5, label=r"Euclidean Distance $\|x_t - x_{gt}\|_2$")
    ax.set_ylabel(r"Euclidean Distance $\|x_t - x_{gt}\|_2$", color="red")
    ax.tick_params(axis="y", labelcolor="red")
    ax.set_xlabel("Flow Timestep t (0 = Noise, 1 = Final)")
    ax.grid(True, alpha=0.3)

    # Second y-axis for RMSE
    ax2 = ax.twinx()
    ax2.plot(t_steps, rmse_arr, "m--d", linewidth=1.8, markersize=4, label="RMSE per Latent Dim")
    ax2.set_ylabel("RMSE per Latent Dim", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    ax.set_title(f"4. Distance to Destination Target (Start: {dist_arr[0]:.1f} -> End: {dist_arr[-1]:.1f})", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 5: Component Evolution: Alpha(t) vs Beta(t)
    # -------------------------------------------------------------
    ax = axs[1, 1]
    ax.plot(t_steps, alphas, "b-o", linewidth=2, markersize=4, label=r"Target Component $\alpha(t)$ (Ideal $\to$ 1.0)")
    ax.plot(t_steps, betas, "r--s", linewidth=2, markersize=4, label=r"Noise Residual $\beta(t)$ (Ideal $\to$ 0.0)")
    ax.axhline(1.0, color="green", linestyle=":", label="Ideal Target (1.0)")
    ax.axhline(0.0, color="gray", linestyle=":", label="Ideal Noise Residual (0.0)")

    ax.set_title(f"5. Basis Evolution: Target $\\alpha={alphas[-1]:.2f}$ | Noise $\\beta={betas[-1]:.2f}$", fontsize=11, fontweight="bold")
    ax.set_xlabel("Flow Timestep t")
    ax.set_ylabel("Projection Weight")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="center left")

    # -------------------------------------------------------------
    # Panel 6: Velocity Norm ||v_t|| vs Required Speed
    # -------------------------------------------------------------
    ax = axs[1, 2]
    v_norms = np.array(sample_data["vel_norms"])
    ideal_sp = np.array(sample_data["ideal_speeds"])

    ax.plot(t_vel, v_norms, "c-o", linewidth=2, markersize=4, label=r"Model Velocity Norm $\|v_t\|_2$")
    ax.plot(t_vel, ideal_sp, "k--", linewidth=1.5, alpha=0.7, label=r"Ideal Required Speed $\|x_{gt} - x_t\|/(1-t)$")

    ax.set_title("6. Velocity Magnitude Progression across Steps", fontsize=11, fontweight="bold")
    ax.set_xlabel("Flow Timestep t")
    ax.set_ylabel(r"Velocity Norm $\|v_t\|_2$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_file = os.path.join(output_dir, f"sample_{s_idx}_{aid}_velocity_trajectory.png")
    plt.savefig(plot_file, dpi=180)
    plt.close()
    print(f"[+] Saved individual deep-dive plot: {plot_file}")
    return plot_file


def plot_all_samples_unified_dashboard(all_samples_data, output_dir="last_5_val_evaluation"):
    """
    Plots a multi-panel comparison dashboard across all 5 samples:
    Top row: Phase-plane trajectories with velocity arrows for all 5 samples.
    Middle row: PCA trajectories with velocity arrows for all 5 samples.
    Bottom row:
      1. Velocity Direction Alignment to Destination cos(v_t, x_gt - x_t)
      2. Distance to Destination ||x_t - x_gt||
      3. Target Component Alpha(t)
      4. Noise Residual Beta(t)
      5. Velocity Norm ||v_t||
    """
    n_samples = len(all_samples_data)
    fig = plt.figure(figsize=(24, 15))
    fig.suptitle(
        "MAMBAFLOW-TTS: VELOCITY TRAJECTORY & DESTINATION ANALYSIS (LAST 5 SAMPLES IN VAL.CSV)\n"
        "Numerical Integration: Euler 30 Steps | Temperature = 1.0 | Checkpoint: 256D BiMamba-2 + Cross-Attention",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]

    # Top Row: 5 Phase-plane Trajectory plots (columns 0..4)
    for i, sdata in enumerate(all_samples_data):
        ax = plt.subplot2grid((3, 5), (0, i))
        alphas = np.array(sdata["alphas"])
        betas = np.array(sdata["betas"])

        # Ideal straight line
        ax.plot([0.0, 1.0], [1.0, 0.0], "k--", alpha=0.35, linewidth=1.5)

        # Actual curve
        ax.plot(alphas, betas, color=colors[i], linewidth=2.2, label="Traj Path", zorder=3)

        # Velocity arrows (every 4 steps)
        for step in range(0, len(alphas) - 1, 4):
            ax.annotate(
                "",
                xy=(alphas[step + 1], betas[step + 1]),
                xytext=(alphas[step], betas[step]),
                arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.5, shrinkA=0, shrinkB=0),
                zorder=4,
            )

        # Start, End, Destination
        ax.scatter([alphas[0]], [betas[0]], color="red", s=80, marker="X", zorder=5, label="Start (t=0)")
        ax.scatter([1.0], [0.0], color="green", s=180, marker="*", zorder=5, label="Dest (x_gt)")
        ax.scatter([alphas[-1]], [betas[-1]], color="orange", s=90, marker="o", edgecolors="black", zorder=5, label="End (t=1)")

        # Residual line
        ax.plot([alphas[-1], 1.0], [betas[-1], 0.0], "r:", alpha=0.7, linewidth=1.5)

        ax.set_title(f"S{sdata['sample_index']}: {sdata['audio_id']}\nEnd: ({alphas[-1]:.2f}, {betas[-1]:.2f})", fontsize=10, fontweight="bold")
        ax.set_xlabel(r"Target Proj $\alpha$ (Goal=1)", fontsize=9)
        ax.set_ylabel(r"Noise Proj $\beta$ (Goal=0)", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Middle Row: 5 PCA Trajectory plots (columns 0..4)
    for i, sdata in enumerate(all_samples_data):
        ax = plt.subplot2grid((3, 5), (1, i))
        px = np.array(sdata["pca_steps_x"])
        py = np.array(sdata["pca_steps_y"])
        gt_px = sdata["pca_gt_x"]
        gt_py = sdata["pca_gt_y"]

        # Trajectory path in PCA
        ax.plot(px, py, color=colors[i], linewidth=2.2, zorder=3)
        ax.plot([px[0], gt_px], [py[0], gt_py], "k--", alpha=0.35, linewidth=1.5)

        # Arrows
        for step in range(0, len(px) - 1, 4):
            ax.annotate(
                "",
                xy=(px[step + 1], py[step + 1]),
                xytext=(px[step], py[step]),
                arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.5, shrinkA=0, shrinkB=0),
                zorder=4,
            )

        ax.scatter([px[0]], [py[0]], color="red", s=80, marker="X", zorder=5)
        ax.scatter([gt_px], [gt_py], color="green", s=180, marker="*", zorder=5)
        ax.scatter([px[-1]], [py[-1]], color="orange", s=90, marker="o", edgecolors="black", zorder=5)
        ax.plot([px[-1], gt_px], [py[-1], gt_py], "r:", alpha=0.7, linewidth=1.5)

        ev = sdata["pca_explained_var"]
        ax.set_title(f"PCA Map (Var: {(ev[0]+ev[1])*100:.1f}%)\n{sdata['audio_id']}", fontsize=10, fontweight="bold")
        ax.set_xlabel("PC 1", fontsize=9)
        ax.set_ylabel("PC 2", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Bottom Row: 5 Comparative Line Charts
    steps_arr = np.arange(30)
    timesteps = np.linspace(0.0, 1.0, 31)

    # Subplot B1: Velocity Direction Alignment to Destination: cos(v_t, x_gt - x_t)
    ax = plt.subplot2grid((3, 5), (2, 0))
    for i, sdata in enumerate(all_samples_data):
        ax.plot(steps_arr, sdata["cos_sim_to_dest"], color=colors[i], marker=markers[i], markersize=3.5, label=f"S{sdata['sample_index']}: {sdata['audio_id']}")
    ax.axhline(1.0, color="green", linestyle=":", alpha=0.8, label="Ideal Heading (+1.0)")
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.6)
    ax.set_title("Velocity Heading to Destination\ncos(v_t, x_gt - x_t)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Euler Step k", fontsize=9)
    ax.set_ylabel("Cosine Similarity", fontsize=9)
    ax.set_ylim(-0.2, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    # Subplot B2: Distance to Destination Target ||x_t - x_gt||
    ax = plt.subplot2grid((3, 5), (2, 1))
    for i, sdata in enumerate(all_samples_data):
        ax.plot(timesteps, sdata["dist_to_dest"], color=colors[i], marker=markers[i], markersize=3.5, label=f"S{sdata['sample_index']}")
    ax.set_title("Distance to Destination\n||x_t - x_gt||_2", fontsize=10, fontweight="bold")
    ax.set_xlabel("Flow Timestep t", fontsize=9)
    ax.set_ylabel("Euclidean Distance", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    # Subplot B3: Target Component Alpha(t) (Ideal -> 1.0)
    ax = plt.subplot2grid((3, 5), (2, 2))
    for i, sdata in enumerate(all_samples_data):
        ax.plot(timesteps, sdata["alphas"], color=colors[i], marker=markers[i], markersize=3.5, label=f"S{sdata['sample_index']}")
    ax.axhline(1.0, color="green", linestyle=":", alpha=0.8, label="Target Goal (1.0)")
    ax.set_title("Target Basis Component\nAlpha(t) (Goal -> 1.0)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Flow Timestep t", fontsize=9)
    ax.set_ylabel("Alpha Weight", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    # Subplot B4: Noise Residual Beta(t) (Ideal -> 0.0)
    ax = plt.subplot2grid((3, 5), (2, 3))
    for i, sdata in enumerate(all_samples_data):
        ax.plot(timesteps, sdata["betas"], color=colors[i], marker=markers[i], markersize=3.5, label=f"S{sdata['sample_index']}")
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.8, label="Noise Free (0.0)")
    ax.set_title("Noise Residual Component\nBeta(t) (Goal -> 0.0)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Flow Timestep t", fontsize=9)
    ax.set_ylabel("Beta Weight", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    # Subplot B5: Velocity Norm ||v_t||
    ax = plt.subplot2grid((3, 5), (2, 4))
    t_vel = np.linspace(0.0, 1.0 - 1.0 / 30, 30)
    for i, sdata in enumerate(all_samples_data):
        ax.plot(t_vel, sdata["vel_norms"], color=colors[i], marker=markers[i], markersize=3.5, label=f"S{sdata['sample_index']}")
    ax.set_title("Velocity Magnitude\n||v_t||_2 across Timesteps", fontsize=10, fontweight="bold")
    ax.set_xlabel("Flow Timestep t", fontsize=9)
    ax.set_ylabel("Velocity Norm", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    dashboard_file = os.path.join(output_dir, "all_5_samples_velocity_trajectory_dashboard.png")
    plt.savefig(dashboard_file, dpi=180)
    plt.close()
    print(f"\n[+] Saved unified 5-sample comparison dashboard: {dashboard_file}")
    return dashboard_file


def main():
    parser = argparse.ArgumentParser(description="MambaFlow-TTS Velocity Trajectory & Audio Generation for Last 5 val.csv Samples")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (default: auto-detect latest in TTS_checkpoints/)")
    parser.add_argument("--val_csv", type=str, default="/home/monesh/ljspeech/LJSpeech-1.1/val.csv")
    parser.add_argument("--steps", type=int, default=30, help="Euler integration steps (default: 30)")
    parser.add_argument("--temp", type=float, default=1.0, help="Noise temperature (default: 1.0)")
    parser.add_argument("--output_dir", type=str, default="last_5_val_evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dacvae_device", type=str, default="cpu")  # CPU keeps GPU memory 100% safe for active training
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Resolve Checkpoint
    ckpt_path = args.ckpt
    if ckpt_path is None or not os.path.exists(ckpt_path):
        detected = find_latest_checkpoint("TTS_checkpoints")
        if detected is None:
            raise FileNotFoundError("Could not find any checkpoint in TTS_checkpoints/")
        ckpt_path = detected
    print(f"[+] Using checkpoint: {ckpt_path}")

    # 2. Load Model & DACVAE
    print(f"[+] Loading MambaFlow model on {args.device}...")
    pl_module = TTSMODEL.load_from_checkpoint(ckpt_path, map_location=args.device)
    model = pl_module.model.to(args.device).eval()

    print(f"[+] Loading DACVAE decoder on {args.dacvae_device}...")
    dacvae_decoder, sr, _ = setup_dacvae(device=args.dacvae_device)

    # 3. Read Last 5 Samples of val.csv
    df = pd.read_csv(args.val_csv)
    last_5 = df.tail(5).reset_index(drop=True)
    print(f"\n[+] Processing last 5 samples from {args.val_csv} (Rows {len(df)-5} to {len(df)-1}):")

    all_samples_data = []
    for i, (_, row) in enumerate(last_5.iterrows()):
        sdata = compute_velocity_trajectory_for_sample(
            model=model,
            dacvae_decoder=dacvae_decoder,
            sample_row=row,
            sample_index=i + 1,
            device=args.device,
            dacvae_device=args.dacvae_device,
            n_steps=args.steps,
            temperature=args.temp,
            output_dir=args.output_dir,
            sr=sr,
        )
        all_samples_data.append(sdata)

        # Plot individual deep-dive figure
        plot_single_sample_deep_dive(sdata, output_dir=args.output_dir)

    # 4. Plot Unified 5-Sample Comparison Dashboard
    dashboard_file = plot_all_samples_unified_dashboard(all_samples_data, output_dir=args.output_dir)

    # 5. Save Summary Table & JSON
    summary_json_path = os.path.join(args.output_dir, "last_5_samples_trajectory_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(all_samples_data, f, indent=2)
    print(f"\n[+] Saved detailed numerical results to: {summary_json_path}")

    print("\n" + "=" * 115)
    print(f"  MAMBAFLOW-TTS: LAST 5 VAL SAMPLES TRAJECTORY & AUDIO SUMMARY (Euler 30 Steps, Temp=1.0)")
    print("=" * 115)
    print(f"{'#':<2} | {'Audio ID':<11} | {'Frames':<6} | {'Target Alpha':<13} | {'Noise Beta':<11} | {'Final L2':<9} | {'Mean Align':<11} | {'Pred Dur':<9} | {'GT Dur':<8} | {'Audio RMS%':<10}")
    print("-" * 115)
    for s in all_samples_data:
        print(f"{s['sample_index']:<2} | {s['audio_id']:<11} | {s['T_frames']:<6} | {s['final_alpha']:<13.3f} | {s['final_beta']:<11.3f} | {s['final_l2']:<9.2f} | {s['mean_cos_dest']:<11.3f} | {s['pred_dur']:<8.2f}s | {s['gt_dur']:<7.2f}s | {s['rms_ratio']:<9.1f}%")
    print("=" * 115)


if __name__ == "__main__":
    main()
