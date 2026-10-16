import os, json, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = "TTS_logs/mambaflow_ca_bimamba_448d_cfg15_scratch/version_0"
event_files = glob.glob(f"{log_dir}/events.out.tfevents*")
if not event_files:
    print("No event files found!")
    exit(1)

ea = EventAccumulator(event_files[0])
ea.Reload()

epochs_scalars = ea.Scalars("epoch")
val_losses = ea.Scalars("val_loss")
val_cfms = ea.Scalars("val_loss_cfm")
val_durs = ea.Scalars("val_loss_dur")
train_losses = ea.Scalars("train_loss_epoch")
train_cfms = ea.Scalars("train_loss_cfm_epoch")
train_durs = ea.Scalars("train_loss_dur_epoch")
lrs = ea.Scalars("lr-AdamW/remaining_decoder")

step_to_epoch = {e.step: int(e.value) for e in epochs_scalars}

data = []
for vl in val_losses:
    s = vl.step
    ep = step_to_epoch.get(s, len(data))
    tl = [t.value for t in train_losses if t.step == s]
    t_cfm = [t.value for t in train_cfms if t.step == s]
    t_dur = [t.value for t in train_durs if t.step == s]
    v_cfm = [t.value for t in val_cfms if t.step == s]
    v_dur = [t.value for t in val_durs if t.step == s]
    lr_vals = [t.value for t in lrs if t.step <= s]

    data.append({
        "step": s,
        "epoch": ep,
        "train_loss": tl[0] if tl else None,
        "train_cfm": t_cfm[0] if t_cfm else None,
        "train_dur": t_dur[0] if t_dur else None,
        "val_loss": vl.value,
        "val_cfm": v_cfm[0] if v_cfm else None,
        "val_dur": v_dur[0] if v_dur else None,
        "lr": lr_vals[-1] if lr_vals else None,
    })

best = min(data, key=lambda x: x["val_loss"])
latest = data[-1]
first = data[0]

print("=== LOSS CURVE SUMMARY ===")
print(f"Total Epochs Completed: {len(data)}")
print(f"Epoch 0:   Train Loss = {first['train_loss']:.4f} | Val Loss = {first['val_loss']:.4f}")
print(f"Latest (Epoch {latest['epoch']}): Train Loss = {latest['train_loss']:.4f} | Val Loss = {latest['val_loss']:.4f}")
print(f"Best Val Loss: {best['val_loss']:.4f} at Epoch {best['epoch']} (Step {best['step']})")

gaps = [d["val_loss"] - d["train_loss"] for d in data if d["train_loss"] is not None]
print(f"Mean Val-Train Gap: {sum(gaps)/len(gaps):.4f}")
print(f"Recent Val-Train Gap (Last 10): {sum(gaps[-10:])/10:.4f}")

with open("scratch_loss_data.json", "w") as f:
    json.dump(data, f, indent=2)
print("Saved scratch_loss_data.json successfully.")
