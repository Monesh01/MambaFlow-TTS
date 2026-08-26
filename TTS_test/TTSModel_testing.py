import os
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sys

# Add parent dir to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TTSDataModule import TTSDataModule, TTSMODEL
from TTSDatasetModule import denormalize_mel
import bigvgan
from bigvgan import BigVGAN
from bigvgan import AttrDict

def get_best_checkpoint(ckpt_dir="TTS_checkpoints/"):
    if not os.path.exists(ckpt_dir):
        return None
    ckpts = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".ckpt") and not f.endswith(".tmp") and not os.path.basename(f).startswith("last")
    ]
    if not ckpts:
        last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        return last_ckpt if os.path.exists(last_ckpt) else None
    
    def extract_val_loss(p):
        import re
        match = re.search(r"val_loss=([0-9]+\.[0-9]+)", os.path.basename(p))
        if match:
            return float(match.group(1))
        return float("inf")
    
    return min(ckpts, key=extract_val_loss)

@torch.no_grad()
def run_validation_test():
    output_dir = "/home/monesh/TTSModel/TTS_test"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running validation test on: {device}", flush=True)

    # 1. Load TTS Model
    ckpt_path = get_best_checkpoint("/home/monesh/TTSModel/TTS_checkpoints/")
    if ckpt_path is None:
        print("No checkpoint found.")
        return
        
    print(f"Loading TTS checkpoint: {ckpt_path}", flush=True)
    lightning_model = TTSMODEL.load_from_checkpoint(ckpt_path, map_location=device)
    lightning_model.eval()
    lightning_model.to(device)
    model = lightning_model.model
    model.eval()

    # 2. Load BigVGAN Vocoder
    print("Loading BigVGAN vocoder...", flush=True)
    config_path = "/home/monesh/bigvgan_model/config_14M.json"
    checkpoint_path = "/home/monesh/bigvgan_model/bigvgan_generator_14M.pt"
    
    with open(config_path) as f:
        config = json.load(f)
    h = AttrDict(config)
    
    bigvgan_model = BigVGAN(h, use_cuda_kernel=False).to(device)
    bigvgan_checkpoint = torch.load(checkpoint_path, map_location=device)
    if "generator" in bigvgan_checkpoint:
        bigvgan_model.load_state_dict(bigvgan_checkpoint["generator"])
    else:
        bigvgan_model.load_state_dict(bigvgan_checkpoint)
    bigvgan_model.eval()
    bigvgan_model.remove_weight_norm()
    sample_rate = h.sampling_rate

    # 3. Load Validation Dataset
    dm = TTSDataModule(
        train_file="/home/monesh/ljspeech/LJSpeech-1.1/train.csv",
        val_file="/home/monesh/ljspeech/LJSpeech-1.1/val.csv",
        batch_size=1, # process one by one to capture individual metrics properly
        num_workers=0
    )
    dm.setup()
    val_loader = dm.val_dataloader()

    results = []
    MAX_SAMPLES = 20 # Limit to 20 samples to keep testing fast but informative
    steps = 25 # Euler steps
    
    print(f"\nStarting Validation Testing (max {MAX_SAMPLES} samples)...")
    
    total_tts_time = 0
    total_audio_dur = 0
    
    for i, batch in enumerate(val_loader):
        if i >= MAX_SAMPLES:
            break
            
        x = batch['ipa'].to(device)
        y_mel = batch['mel'].to(device) # ground truth mel [B, T_mel, 100]
        x_lengths = batch['ipa_lengths']
        y_lengths = batch['lengths']
        
        # Inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        # The model usually predicts its own duration in inference
        mel_pred_norm, latent_expanded = model(
            x=x,
            target_latent=None,
            n_timesteps=steps,
            temperature=0.3, # lower temperature for less noise
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        tts_time = t1 - t0
        
        # We need to compute metrics. Since the predicted mel and ground truth mel might
        # have different lengths due to predicted durations, we align them or crop to min length.
        gt_mel_len = y_mel.size(1)
        pred_mel_len = mel_pred_norm.size(1)
        min_len = min(gt_mel_len, pred_mel_len)
        
        pred_aligned = mel_pred_norm[0, :min_len, :]
        gt_aligned = y_mel[0, :min_len, :]
        
        # Calculate MSE and MAE on normalized mels
        mse = F.mse_loss(pred_aligned, gt_aligned).item()
        mae = F.l1_loss(pred_aligned, gt_aligned).item()
        
        # Denormalize
        mel_pred_raw = denormalize_mel(mel_pred_norm)
        gt_mel_raw = denormalize_mel(y_mel)
        
        # Generate Audio with BigVGAN
        mel_bigvgan = mel_pred_raw.transpose(1, 2)
        with torch.inference_mode():
            audio = bigvgan_model(mel_bigvgan)
        
        audio_cpu = audio.squeeze().detach().cpu().numpy()
        audio_duration = len(audio_cpu) / sample_rate
        
        total_tts_time += tts_time
        total_audio_dur += audio_duration
        rtf = tts_time / audio_duration if audio_duration > 0 else 0
        
        # Peak Norm
        peak = np.abs(audio_cpu).max()
        if peak > 1e-6:
            audio_cpu = audio_cpu / peak * 0.95
            
        audio_path = os.path.join(output_dir, f"val_sample_{i}.wav")
        sf.write(audio_path, audio_cpu, sample_rate)
        
        # Plot Mel Spectrogram Comparison
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(gt_mel_raw[0].cpu().numpy().T, aspect='auto', origin='lower')
        plt.title(f"Ground Truth Mel (Len: {gt_mel_len})")
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(mel_pred_raw[0].cpu().numpy().T, aspect='auto', origin='lower')
        plt.title(f"Predicted Mel (Len: {pred_mel_len})")
        plt.colorbar()
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"val_mel_compare_{i}.png")
        plt.savefig(plot_path)
        plt.close()
        
        results.append({
            "sample_idx": i,
            "gt_length": gt_mel_len,
            "pred_length": pred_mel_len,
            "mse": mse,
            "mae": mae,
            "rtf": rtf,
            "audio_duration": audio_duration
        })
        
        print(f"Sample {i}: MSE={mse:.4f}, MAE={mae:.4f}, RTF={rtf:.4f}")
        
    avg_mse = sum(r["mse"] for r in results) / len(results)
    avg_mae = sum(r["mae"] for r in results) / len(results)
    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    
    summary = {
        "num_samples_tested": len(results),
        "average_mse": avg_mse,
        "average_mae": avg_mae,
        "average_rtf": avg_rtf,
        "total_audio_generated_sec": total_audio_dur,
        "total_tts_inference_sec": total_tts_time,
        "global_rtf": total_tts_time / total_audio_dur if total_audio_dur > 0 else 0,
        "samples": results
    }
    
    report_path = os.path.join(output_dir, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nTesting Complete! Report saved to {report_path}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average RTF: {avg_rtf:.4f}")

if __name__ == "__main__":
    run_validation_test()
