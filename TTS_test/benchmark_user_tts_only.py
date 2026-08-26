import os
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import soundfile as sf
import sys
from tqdm import tqdm

# Add parent dir to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Monkey patch torch.load to bypass weights_only restriction in PyTorch 2.6+
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

from TTSDataModule import TTSMODEL
from TTSDatasetModule import denormalize_mel
import bigvgan
from bigvgan import BigVGAN
from bigvgan import AttrDict
from preprocessing.text import text_to_sequence

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
def run_benchmark():
    output_dir = "/home/monesh/TTSModel/TTS_test/benchmark_outputs_user_only"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running User TTS full validation benchmark on: {device}", flush=True)

    # 1. Load User TTS Model
    ckpt_path = get_best_checkpoint("/home/monesh/TTSModel/TTS_checkpoints/")
    print(f"Loading User TTS checkpoint: {ckpt_path}", flush=True)
    lightning_model = TTSMODEL.load_from_checkpoint(ckpt_path, map_location=device)
    lightning_model.eval()
    lightning_model.to(device)
    user_model = lightning_model.model
    user_model.eval()

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
    val_csv_path = "/home/monesh/ljspeech/LJSpeech-1.1/val.csv"
    df = pd.read_csv(val_csv_path)
    print(f"Loaded validation set: {len(df)} samples.", flush=True)

    total_tts_time = 0
    total_vocoder_time = 0
    total_audio_dur = 0
    
    # Warmup
    print("Warming up...", flush=True)
    dummy_seq = text_to_sequence("Warmup sentence for TTS model.", ["english_cleaners2"])[0]
    dummy_x = torch.tensor(dummy_seq, dtype=torch.long, device=device).unsqueeze(0)
    _ = user_model(x=dummy_x, target_latent=None, n_timesteps=10, temperature=0.3)
    
    print("\nStarting User TTS Benchmark with n_timesteps=10 (same as Matcha)...")
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Benchmarking User TTS"):
        text = ""
        for col in ['normalized_text', 'normalized', 'text', 'transcript', 'txt']:
            if col in row and pd.notna(row[col]):
                text = str(row[col])
                break
        
        if not text.strip():
            continue
            
        seq = text_to_sequence(text, ["english_cleaners2"])[0]
        x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        
        # -------------------------------------------------------------
        # USER MODEL INFERENCE (10 Euler steps)
        # -------------------------------------------------------------
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        user_mel_pred_norm, _ = user_model(
            x=x,
            target_latent=None,
            n_timesteps=10,  # 10 steps to match Matcha
            temperature=0.3,
        )
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        user_mel_raw = denormalize_mel(user_mel_pred_norm)
        user_mel_bigvgan = user_mel_raw.transpose(1, 2)
        
        # BigVGAN Vocoder
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        with torch.inference_mode():
            user_audio = bigvgan_model(user_mel_bigvgan)
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t3 = time.perf_counter()
        
        user_audio_cpu = user_audio.squeeze().detach().cpu().numpy()
        user_audio_dur = len(user_audio_cpu) / sample_rate
        
        tts_time = t1 - t0
        vocoder_time = t3 - t2
        
        total_tts_time += tts_time
        total_vocoder_time += vocoder_time
        total_audio_dur += user_audio_dur
        
        # Peak Norm and save audio
        peak = np.abs(user_audio_cpu).max()
        if peak > 1e-6:
            user_audio_cpu = user_audio_cpu / peak * 0.95
        sf.write(os.path.join(output_dir, f"user_sample_{i}.wav"), user_audio_cpu, sample_rate)

    print("\n==========================================")
    print("--- USER TTS MODEL BENCHMARK RESULTS ---")
    print("==========================================")
    print(f"Total Samples:      {len(df)}")
    print(f"ODE Steps:          10 (same as Matcha)")
    print(f"Total Audio Dur:    {total_audio_dur:.2f} s")
    print(f"Total TTS Time:     {total_tts_time:.2f} s")
    print(f"Total Vocoder Time: {total_vocoder_time:.2f} s")
    print(f"Total E2E Time:     {total_tts_time + total_vocoder_time:.2f} s")
    
    avg_tts_rtf = total_tts_time / total_audio_dur
    avg_vocoder_rtf = total_vocoder_time / total_audio_dur
    avg_e2e_rtf = (total_tts_time + total_vocoder_time) / total_audio_dur
    print(f"\nAverage TTS RTF:     {avg_tts_rtf:.4f}")
    print(f"Average Vocoder RTF: {avg_vocoder_rtf:.4f}")
    print(f"Average E2E RTF:     {avg_e2e_rtf:.4f}")
    print("==========================================")

if __name__ == "__main__":
    run_benchmark()
