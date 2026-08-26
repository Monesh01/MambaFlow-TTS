import os
import sys
import time
import json
import torch
import numpy as np
import soundfile as sf

# Add paths
sys.path.insert(0, "/home/monesh/Matcha-TTS")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Monkey patch torch.load to bypass weights_only restriction in PyTorch 2.6+
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

# User Mamba TTS Imports
from TTSDataModule import TTSMODEL
from TTSDatasetModule import denormalize_mel
from test_inference_bigvgan import get_latest_checkpoint
import bigvgan
from bigvgan import BigVGAN
from bigvgan import AttrDict
from preprocessing.text import text_to_sequence

# Matcha-TTS Imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.cli import load_vocoder, to_waveform, process_text

@torch.no_grad()
def main():
    output_dir = "/home/monesh/TTSModel/long_audio_comparison"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 1-Minute Audio Generation Comparison (15 Euler Steps, Temp = 0.667) ===")
    print(f"Running on: {device}")

    # Long text (~150 words, designed for ~55-65 seconds of continuous speech)
    long_text = (
        "The rapid advancement of artificial intelligence and generative modeling has fundamentally transformed modern speech synthesis. "
        "Traditional text-to-speech pipelines often relied on multi-stage architectures with autoregressive decoders, which suffered from slow sequential generation and compounding error propagation. "
        "In recent years, continuous flow matching and state-space architectures have emerged as powerful alternatives. "
        "By modeling the velocity field of a probability path between simple prior distributions and complex acoustic targets, flow matching enables high-fidelity mel-spectrogram generation in just a handful of integration steps. "
        "Furthermore, replacing quadratic self-attention mechanisms with linear state-space models allows speech synthesis systems to scale effortlessly to long-form audio without running into memory bottlenecks or computational slowdowns. "
        "As these acoustic representations pass through high-capacity neural vocoders with anti-aliased periodic activations, the resulting synthesized speech exhibits exceptional naturalness, crystal-clear articulation, and smooth, human-like prosodic rhythm throughout the entire passage."
    )

    print(f"\nText character count: {len(long_text)} chars")
    print(f"Text word count: {len(long_text.split())} words")

    # -------------------------------------------------------------
    # 1. LOAD USER MAMBA-2 TTS MODEL & BIGVGAN
    # -------------------------------------------------------------
    print("\n[1/4] Loading Mamba-2 CFM TTS Model & BigVGAN Vocoder...")
    ckpt_path = get_latest_checkpoint("/home/monesh/TTSModel/TTS_checkpoints/")
    print(f"  TTS Checkpoint: {os.path.basename(ckpt_path)}")
    lightning_model = TTSMODEL.load_from_checkpoint(ckpt_path, map_location=device)
    lightning_model.eval()
    lightning_model.to(device)
    user_model = lightning_model.model
    user_model.eval()

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
    user_sr = h.sampling_rate

    # -------------------------------------------------------------
    # 2. GENERATE USER MAMBA-2 TTS AUDIO
    # -------------------------------------------------------------
    print("\n[2/4] Synthesizing 1-minute audio with Mamba-2 TTS (15 steps, temp=0.667)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0_mamba = time.perf_counter()

    seq = text_to_sequence(long_text, ["english_cleaners2"])[0]
    x_mamba = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

    user_mel_pred_norm, _ = user_model(
        x=x_mamba,
        target_latent=None,
        n_timesteps=15,
        temperature=0.667,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1_mamba = time.perf_counter()

    user_mel_raw = denormalize_mel(user_mel_pred_norm)
    user_mel_bigvgan = user_mel_raw.transpose(1, 2)

    with torch.inference_mode():
        mamba_wav = bigvgan_model(user_mel_bigvgan).squeeze().cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2_mamba = time.perf_counter()
    mamba_peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    # Normalize audio
    max_val_mamba = np.max(np.abs(mamba_wav))
    if max_val_mamba > 0:
        mamba_wav = (mamba_wav / max_val_mamba) * 0.95

    mamba_dur = len(mamba_wav) / user_sr
    mamba_tts_time = t1_mamba - t0_mamba
    mamba_total_time = t2_mamba - t0_mamba
    mamba_rtf = mamba_total_time / mamba_dur

    mamba_path = os.path.join(output_dir, "mamba_tts_1min_15steps.wav")
    sf.write(mamba_path, mamba_wav, user_sr)
    print(f"  [+] Saved Mamba-2 Audio: {mamba_path}")
    print(f"      Duration: {mamba_dur:.2f}s | ODE Time: {mamba_tts_time:.2f}s | Total Time: {mamba_total_time:.2f}s | RTF: {mamba_rtf:.4f} | Peak VRAM: {mamba_peak_vram:.1f} MB")

    # Clean memory before Matcha
    del user_model, lightning_model, bigvgan_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 3. LOAD MATCHA-TTS & HIFIGAN
    # -------------------------------------------------------------
    print("\n[3/4] Loading Matcha-TTS & HiFi-GAN Vocoder...")
    matcha_model = MatchaTTS.load_from_checkpoint(
        "/home/monesh/.local/share/matcha_tts/matcha_ljspeech.ckpt",
        map_location=device
    )
    matcha_model.eval()
    matcha_model.to(device)

    vocoder, denoiser = load_vocoder(
        "hifigan_T2_v1",
        "/home/monesh/.local/share/matcha_tts/hifigan_T2_v1",
        device
    )
    matcha_sr = 22050

    # -------------------------------------------------------------
    # 4. GENERATE MATCHA-TTS AUDIO
    # -------------------------------------------------------------
    print("\n[4/4] Synthesizing 1-minute audio with Matcha-TTS (15 steps, temp=0.667)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0_matcha = time.perf_counter()

    text_processed = process_text(0, long_text, device)
    matcha_output = matcha_model.synthesise(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=15,
        temperature=0.667,
        spks=None,
        length_scale=0.95,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1_matcha = time.perf_counter()

    matcha_mel = matcha_output['mel']
    matcha_audio = to_waveform(matcha_mel, vocoder, denoiser, denoiser_strength=0.00025)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2_matcha = time.perf_counter()
    matcha_peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    matcha_wav = matcha_audio.squeeze().cpu().numpy()
    max_val_matcha = np.max(np.abs(matcha_wav))
    if max_val_matcha > 0:
        matcha_wav = (matcha_wav / max_val_matcha) * 0.95

    matcha_dur = len(matcha_wav) / matcha_sr
    matcha_tts_time = t1_matcha - t0_matcha
    matcha_total_time = t2_matcha - t0_matcha
    matcha_rtf = matcha_total_time / matcha_dur

    matcha_path = os.path.join(output_dir, "matcha_tts_1min_15steps.wav")
    sf.write(matcha_path, matcha_wav, matcha_sr)
    print(f"  [+] Saved Matcha-TTS Audio: {matcha_path}")
    print(f"      Duration: {matcha_dur:.2f}s | ODE Time: {matcha_tts_time:.2f}s | Total Time: {matcha_total_time:.2f}s | RTF: {matcha_rtf:.4f} | Peak VRAM: {matcha_peak_vram:.1f} MB")

    # Save summary metadata
    summary = {
        "text": long_text,
        "steps": 15,
        "temperature": 0.667,
        "mamba_tts": {
            "file": mamba_path,
            "duration_sec": round(mamba_dur, 2),
            "ode_time_sec": round(mamba_tts_time, 3),
            "total_time_sec": round(mamba_total_time, 3),
            "rtf": round(mamba_rtf, 4),
            "peak_vram_mb": round(mamba_peak_vram, 1),
            "sample_rate": user_sr
        },
        "matcha_tts": {
            "file": matcha_path,
            "duration_sec": round(matcha_dur, 2),
            "ode_time_sec": round(matcha_tts_time, 3),
            "total_time_sec": round(matcha_total_time, 3),
            "rtf": round(matcha_rtf, 4),
            "peak_vram_mb": round(matcha_peak_vram, 1),
            "sample_rate": matcha_sr
        }
    }

    with open(os.path.join(output_dir, "comparison_metadata.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "="*70)
    print("SUCCESS: 1-Minute Audio Comparison Completed!")
    print(f"Folder: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
