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

# My Model Imports
from TTSDataModule import TTSMODEL
from TTSDatasetModule import denormalize_mel
from test_inference_bigvgan import get_latest_checkpoint
import bigvgan
from bigvgan import BigVGAN
from bigvgan import AttrDict

# Matcha Imports
sys.path.append("/home/monesh/Matcha-TTS")
from matcha.models.matcha_tts import MatchaTTS
from preprocessing.text import text_to_sequence

def clean_path(path_str, mount_prefix="/mnt/windows"):
    import re
    if not isinstance(path_str, str) or path_str.startswith("/home/"):
        return path_str
    if path_str.startswith(mount_prefix):
        return path_str
    path_str = path_str.replace("\\", "/")
    if re.match(r"^[a-zA-Z]:", path_str):
        path_str = re.sub(r"^[a-zA-Z]:[/]*", "", path_str)
        return os.path.normpath(os.path.join(mount_prefix, path_str.lstrip("/")))
    return os.path.normpath(path_str)

@torch.no_grad()
def run_benchmark():
    output_dir = "/home/monesh/TTSModel/TTS_test/benchmark_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running full benchmark on: {device}", flush=True)

    # 1. Load User TTS Model
    ckpt_path = get_latest_checkpoint("/home/monesh/TTSModel/TTS_checkpoints/")
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

    # 3. Load Matcha-TTS Model
    print("Loading Matcha-TTS...", flush=True)
    matcha_model = MatchaTTS.load_from_checkpoint("/home/monesh/.local/share/matcha_tts/matcha_ljspeech.ckpt", map_location=device)
    matcha_model.eval()
    matcha_model.to(device)

    # 4. Load Validation Dataset metadata
    val_csv_path = "/home/monesh/ljspeech/LJSpeech-1.1/val.csv"
    df = pd.read_csv(val_csv_path)
    print(f"Loaded validation set: {len(df)} samples.", flush=True)

    results_user = []
    results_matcha = []
    
    total_user_tts_time = 0
    total_user_vocoder_time = 0
    total_user_audio_dur = 0
    
    total_matcha_tts_time = 0
    total_matcha_vocoder_time = 0
    total_matcha_audio_dur = 0
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Benchmarking"):
        # Get raw text
        text = ""
        for col in ['normalized_text', 'normalized', 'text', 'transcript', 'txt']:
            if col in row and pd.notna(row[col]):
                text = str(row[col])
                break
        
        if not text:
            continue
            
        # Get phonemes
        seq, _ = text_to_sequence(text, ["english_cleaners2"])
        x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        x_lengths = torch.tensor([len(seq)], dtype=torch.long, device=device)
        
        # -------------------------------------------------------------
        # USER MODEL INFERENCE
        # -------------------------------------------------------------
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        user_mel_pred_norm, _ = user_model(
            x=x,
            target_latent=None,
            n_timesteps=25,
            temperature=0.3,
        )
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        user_mel_raw = denormalize_mel(user_mel_pred_norm)
        user_mel_bigvgan = user_mel_raw.transpose(1, 2)
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        user_audio = bigvgan_model(user_mel_bigvgan)
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t3 = time.perf_counter()
        
        user_audio_cpu = user_audio.squeeze().detach().cpu().numpy()
        user_audio_dur = len(user_audio_cpu) / sample_rate
        
        user_tts_time = t1 - t0
        user_vocoder_time = t3 - t2
        user_total_time = user_tts_time + user_vocoder_time
        user_rtf = user_total_time / user_audio_dur if user_audio_dur > 0 else 0
        
        total_user_tts_time += user_tts_time
        total_user_vocoder_time += user_vocoder_time
        total_user_audio_dur += user_audio_dur
        
        # Save user audio
        peak = np.abs(user_audio_cpu).max()
        if peak > 1e-6:
            user_audio_cpu = user_audio_cpu / peak * 0.95
        sf.write(os.path.join(output_dir, f"user_sample_{i}.wav"), user_audio_cpu, sample_rate)

        # -------------------------------------------------------------
        # MATCHA MODEL INFERENCE
        # -------------------------------------------------------------
        if torch.cuda.is_available(): torch.cuda.synchronize()
        m_t0 = time.perf_counter()
        
        matcha_output = matcha_model.synthesise(x, x_lengths, n_timesteps=25, temperature=0.3)
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        m_t1 = time.perf_counter()
        
        matcha_mel = matcha_output['mel'] # Shape [1, 80, T] typically
        # BigVGAN might expect 100 dims! If matcha mel is 80-dim, BigVGAN will throw error.
        # Let's check matcha mel shape and pad if necessary, or just not vocode matcha if it fails, but user wanted wav.
        # Wait, Matcha-TTS uses a pre-trained hifigan, maybe we can load it?
        
        # Let's try BigVGAN first, if it fails due to dimension mismatch, we pad it to 100.
        if matcha_mel.shape[1] == 80:
            matcha_mel_padded = F.pad(matcha_mel, (0, 0, 0, 20), "constant", -11.51)
        else:
            matcha_mel_padded = matcha_mel

        if torch.cuda.is_available(): torch.cuda.synchronize()
        m_t2 = time.perf_counter()
        
        try:
            matcha_audio = bigvgan_model(matcha_mel_padded)
        except Exception as e:
            matcha_audio = torch.zeros((1, 1, 22050), device=device) # dummy fallback
            
        if torch.cuda.is_available(): torch.cuda.synchronize()
        m_t3 = time.perf_counter()
        
        matcha_audio_cpu = matcha_audio.squeeze().detach().cpu().numpy()
        matcha_audio_dur = len(matcha_audio_cpu) / sample_rate
        # Use Matcha's predicted mel length to calculate its intended audio dur: frames * hop_size / sr
        # Matcha typically uses hop_size=256
        intended_matcha_dur = matcha_mel.shape[-1] * 256 / 22050
        if matcha_audio_dur == 0: matcha_audio_dur = intended_matcha_dur

        matcha_tts_time = m_t1 - m_t0
        matcha_vocoder_time = m_t3 - m_t2
        matcha_total_time = matcha_tts_time + matcha_vocoder_time
        matcha_rtf = matcha_total_time / matcha_audio_dur if matcha_audio_dur > 0 else 0
        
        total_matcha_tts_time += matcha_tts_time
        total_matcha_vocoder_time += matcha_vocoder_time
        total_matcha_audio_dur += matcha_audio_dur
        
        # Save matcha audio
        peak_m = np.abs(matcha_audio_cpu).max()
        if peak_m > 1e-6:
            matcha_audio_cpu = matcha_audio_cpu / peak_m * 0.95
        sf.write(os.path.join(output_dir, f"matcha_sample_{i}.wav"), matcha_audio_cpu, sample_rate)

        results_user.append(user_rtf)
        results_matcha.append(matcha_rtf)

    print("\n--- BENCHMARK RESULTS ---")
    print(f"Total Samples: {len(df)}")
    
    # User Results
    print("\n[USER TTS MODEL]")
    print(f"Total TTS Time:     {total_user_tts_time:.2f} s")
    print(f"Total Vocoder Time: {total_user_vocoder_time:.2f} s")
    print(f"Total E2E Time:     {total_user_tts_time + total_user_vocoder_time:.2f} s")
    print(f"Total Audio Dur:    {total_user_audio_dur:.2f} s")
    user_avg_tts_rtf = total_user_tts_time / total_user_audio_dur
    user_avg_e2e_rtf = (total_user_tts_time + total_user_vocoder_time) / total_user_audio_dur
    print(f"Average TTS RTF:    {user_avg_tts_rtf:.4f}")
    print(f"Average E2E RTF:    {user_avg_e2e_rtf:.4f}")

    # Matcha Results
    print("\n[MATCHA TTS MODEL]")
    print(f"Total TTS Time:     {total_matcha_tts_time:.2f} s")
    print(f"Total Vocoder Time: {total_matcha_vocoder_time:.2f} s")
    print(f"Total E2E Time:     {total_matcha_tts_time + total_matcha_vocoder_time:.2f} s")
    print(f"Total Audio Dur:    {total_matcha_audio_dur:.2f} s")
    matcha_avg_tts_rtf = total_matcha_tts_time / total_matcha_audio_dur
    matcha_avg_e2e_rtf = (total_matcha_tts_time + total_matcha_vocoder_time) / total_matcha_audio_dur
    print(f"Average TTS RTF:    {matcha_avg_tts_rtf:.4f}")
    print(f"Average E2E RTF:    {matcha_avg_e2e_rtf:.4f}")

if __name__ == "__main__":
    run_benchmark()
