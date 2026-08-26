import os
import time
import torch
import pandas as pd
import soundfile as sf
import sys
from tqdm import tqdm

# Monkey patch torch.load to bypass weights_only restriction in PyTorch 2.6+
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

# Matcha Imports
sys.path.append("/home/monesh/Matcha-TTS")
from matcha.models.matcha_tts import MatchaTTS
from matcha.cli import load_vocoder, to_waveform, process_text

@torch.no_grad()
def run_benchmark():
    output_dir = "/home/monesh/TTSModel/TTS_test/benchmark_outputs_matcha_only"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Matcha-only full validation benchmark on: {device}", flush=True)

    # 1. Load Matcha-TTS Model
    print("Loading Matcha-TTS...", flush=True)
    matcha_model = MatchaTTS.load_from_checkpoint(
        "/home/monesh/.local/share/matcha_tts/matcha_ljspeech.ckpt",
        map_location=device
    )
    matcha_model.eval()
    matcha_model.to(device)

    # 2. Load Native Matcha Vocoder
    print("Loading Matcha Vocoder (HiFi-GAN)...", flush=True)
    vocoder, denoiser = load_vocoder(
        "hifigan_T2_v1",
        "/home/monesh/.local/share/matcha_tts/hifigan_T2_v1",
        device
    )

    # 3. Load Validation Dataset metadata
    val_csv_path = "/home/monesh/ljspeech/LJSpeech-1.1/val.csv"
    df = pd.read_csv(val_csv_path)
    print(f"Loaded validation set: {len(df)} samples.", flush=True)

    total_matcha_tts_time = 0
    total_matcha_vocoder_time = 0
    total_matcha_audio_dur = 0
    
    # Warmup
    print("Warming up...", flush=True)
    dummy_processed = process_text(0, "Warmup sentence for Matcha-TTS.", device)
    _ = matcha_model.synthesise(dummy_processed["x"], dummy_processed["x_lengths"], n_timesteps=10, temperature=0.667)
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Benchmarking Matcha"):
        text = ""
        for col in ['normalized_text', 'normalized', 'text', 'transcript', 'txt']:
            if col in row and pd.notna(row[col]):
                text = str(row[col])
                break
        
        if not text.strip():
            continue
            
        # Official Matcha preprocessing with blank interspersing
        text_processed = process_text(i, text, device)
        
        # -------------------------------------------------------------
        # MATCHA MODEL INFERENCE
        # -------------------------------------------------------------
        if torch.cuda.is_available(): torch.cuda.synchronize()
        m_t0 = time.perf_counter()
        
        matcha_output = matcha_model.synthesise(
            text_processed["x"],
            text_processed["x_lengths"],
            n_timesteps=10,
            temperature=0.667,
            spks=None,
            length_scale=0.95,
        )
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        m_t1 = time.perf_counter()
        
        matcha_mel = matcha_output['mel']
        
        # Native vocoder
        if torch.cuda.is_available(): torch.cuda.synchronize()
        m_t2 = time.perf_counter()
        
        matcha_audio = to_waveform(matcha_mel, vocoder, denoiser, denoiser_strength=0.00025)
            
        if torch.cuda.is_available(): torch.cuda.synchronize()
        m_t3 = time.perf_counter()
        
        matcha_audio_cpu = matcha_audio.numpy()
        sample_rate = 22050
        matcha_audio_dur = len(matcha_audio_cpu) / sample_rate

        matcha_tts_time = m_t1 - m_t0
        matcha_vocoder_time = m_t3 - m_t2
        
        total_matcha_tts_time += matcha_tts_time
        total_matcha_vocoder_time += matcha_vocoder_time
        total_matcha_audio_dur += matcha_audio_dur
        
        # Save matcha audio
        sf.write(os.path.join(output_dir, f"matcha_sample_{i}.wav"), matcha_audio_cpu, sample_rate, "PCM_24")

    print("\n==========================================")
    print("--- MATCHA-ONLY BENCHMARK RESULTS ---")
    print("==========================================")
    print(f"Total Samples:      {len(df)}")
    print(f"Total Audio Dur:    {total_matcha_audio_dur:.2f} s")
    print(f"Total TTS Time:     {total_matcha_tts_time:.2f} s")
    print(f"Total Vocoder Time: {total_matcha_vocoder_time:.2f} s")
    print(f"Total E2E Time:     {total_matcha_tts_time + total_matcha_vocoder_time:.2f} s")
    
    matcha_avg_tts_rtf = total_matcha_tts_time / total_matcha_audio_dur
    matcha_avg_e2e_rtf = (total_matcha_tts_time + total_matcha_vocoder_time) / total_matcha_audio_dur
    print(f"\nAverage TTS RTF:    {matcha_avg_tts_rtf:.4f}")
    print(f"Average E2E RTF:    {matcha_avg_e2e_rtf:.4f}")
    print("==========================================")

if __name__ == "__main__":
    run_benchmark()
