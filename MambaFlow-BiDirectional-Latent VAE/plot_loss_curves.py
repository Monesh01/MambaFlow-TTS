import json
import matplotlib.pyplot as plt
import numpy as np

with open("scratch_loss_data.json", "r") as f:
    data = json.load(f)

epochs = [d["epoch"] for d in data]
train_loss = [d["train_loss"] for d in data]
val_loss = [d["val_loss"] for d in data]
train_cfm = [d["train_cfm"] for d in data]
val_cfm = [d["val_cfm"] for d in data]
train_dur = [d["train_dur"] for d in data]
val_dur = [d["val_dur"] for d in data]
lrs = [d["lr"] for d in data]

# Find best validation loss
best_idx = int(np.argmin(val_loss))
best_epoch = epochs[best_idx]
best_val = val_loss[best_idx]

# Style settings
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=200)

# Colors
c_train = "#1f77b4"  # deep blue
c_val = "#d62728"    # vibrant red/crimson
c_best = "#2ca02c"   # emerald green
c_lr = "#9467bd"     # purple

# 1. Total Loss (Train vs Val)
ax = axes[0, 0]
ax.plot(epochs, train_loss, label="Train Loss (Epoch)", color=c_train, linewidth=2.0, alpha=0.9)
ax.plot(epochs, val_loss, label="Val Loss", color=c_val, linewidth=2.0, alpha=0.9)
ax.scatter([best_epoch], [best_val], color=c_best, s=120, zorder=5, label=f"Best Val: {best_val:.4f} (Ep {best_epoch})")
ax.axvline(best_epoch, color=c_best, linestyle="--", alpha=0.5)
ax.set_title("Total Loss: Train vs Validation", fontsize=14, fontweight="bold", pad=10)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.6)

# 2. CFM Flow Matching Loss (Velocity Vector Field MSE)
ax = axes[0, 1]
ax.plot(epochs, train_cfm, label="Train CFM Loss", color=c_train, linewidth=2.0, alpha=0.9)
ax.plot(epochs, val_cfm, label="Val CFM Loss", color=c_val, linewidth=2.0, alpha=0.9)
best_cfm_idx = int(np.argmin(val_cfm))
ax.scatter([epochs[best_cfm_idx]], [val_cfm[best_cfm_idx]], color=c_best, s=120, zorder=5, 
           label=f"Best CFM Val: {val_cfm[best_cfm_idx]:.4f} (Ep {epochs[best_cfm_idx]})")
ax.set_title("CFM Loss (Optimal Transport Flow Matching)", fontsize=14, fontweight="bold", pad=10)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("CFM Loss", fontsize=11)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.6)

# 3. Duration Predictor Loss (Log-Duration MSE)
ax = axes[1, 0]
ax.plot(epochs, train_dur, label="Train Duration Loss", color=c_train, linewidth=2.0, alpha=0.9)
ax.plot(epochs, val_dur, label="Val Duration Loss", color=c_val, linewidth=2.0, alpha=0.9)
best_dur_idx = int(np.argmin(val_dur))
ax.scatter([epochs[best_dur_idx]], [val_dur[best_dur_idx]], color=c_best, s=120, zorder=5, 
           label=f"Best Dur Val: {val_dur[best_dur_idx]:.4f} (Ep {epochs[best_dur_idx]})")
ax.set_title("Duration Predictor Loss (Phone Duration Head)", fontsize=14, fontweight="bold", pad=10)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Duration Loss", fontsize=11)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.6)

# 4. Learning Rate Schedule & Generalization Gap
ax = axes[1, 1]
color_gap = "#ff7f0e"
gap = np.array(val_loss) - np.array(train_loss)
line1 = ax.plot(epochs, gap, label="Generalization Gap (Val - Train)", color=color_gap, linewidth=1.8, linestyle="-")
ax.set_ylabel("Loss Gap (Val - Train)", color=color_gap, fontsize=11)
ax.tick_params(axis="y", labelcolor=color_gap)
ax.set_xlabel("Epoch", fontsize=11)

ax2 = ax.twinx()
line2 = ax2.plot(epochs, lrs, label="Learning Rate", color=c_lr, linewidth=2.0, linestyle=":")
ax2.set_ylabel("Learning Rate (AdamW)", color=c_lr, fontsize=11)
ax2.tick_params(axis="y", labelcolor=c_lr)
ax2.set_yscale("log")

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc="upper right", fontsize=10)
ax.set_title("Generalization Gap & Learning Rate Schedule", fontsize=14, fontweight="bold", pad=10)
ax.grid(True, linestyle="--", alpha=0.6)

plt.suptitle(
    f"MambaFlow-TTS (448D CFM + 15% CFG) Training Dynamics (Epochs 0 - {epochs[-1]})\n"
    f"Active Training Run on NVIDIA RTX 3050 | Current Best Val Loss = {best_val:.4f} (Epoch {best_epoch})",
    fontsize=16,
    fontweight="bold",
    y=0.99,
)

plt.tight_layout()
out_plot_path = "loss_curve_mambaflow_448d.png"
plt.savefig(out_plot_path, bbox_inches="tight", dpi=200)
plt.close()
print(f"[+] Saved high-resolution loss curves plot to: {out_plot_path}")
