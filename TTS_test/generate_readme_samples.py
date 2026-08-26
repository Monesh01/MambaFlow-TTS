import os
import sys
import json
import torch
import numpy as np
import soundfile as sf

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
from test_inference_bigvgan import get_latest_checkpoint
import bigvgan
from bigvgan import BigVGAN
from bigvgan import AttrDict
from preprocessing.text import text_to_sequence

@torch.no_grad()
def generate_samples():
    output_dir = "/home/monesh/TTSModel/audio_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating README audio samples on: {device}", flush=True)

    # 1. Load User TTS Model
    ckpt_path = get_latest_checkpoint("/home/monesh/TTSModel/TTS_checkpoints/")
    print(f"Loading TTS checkpoint: {ckpt_path}", flush=True)
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

    samples = [
        {
            "category": "Medium Sentence",
            "filename": "sample_medium.wav",
            "text": "The quick brown fox jumps over the lazy dog, demonstrating clear prosody and natural flow."
        },
        {
            "category": "Longest Sentence",
            "filename": "sample_longest.wav",
            "text": "Continuous flow matching combined with bidirectional state-space models provides a modern, mathematically rigorous foundation for high-fidelity speech synthesis, effortlessly maintaining acoustic consistency across extended paragraphs without quadratic attention bottlenecks."
        },
        {
            "category": "Emotional Expression",
            "filename": "sample_emotional.wav",
            "text": "I simply cannot believe it! We finally made it through against all odds, and it feels absolutely incredible!"
        },
        {
            "category": "Conversational Daily",
            "filename": "sample_conversational.wav",
            "text": "Good morning! How are you doing today? I hope you're ready for an exciting journey ahead."
        }
    ]

    for item in samples:
        print(f"\nSynthesizing: {item['category']} (25 Euler steps)...")
        print(f"Text: '{item['text']}'")
        
        seq = text_to_sequence(item["text"], ["english_cleaners2"])[0]
        x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        
        user_mel_pred_norm, _ = user_model(
            x=x,
            target_latent=None,
            n_timesteps=25,
            temperature=0.667,
        )
        
        user_mel_raw = denormalize_mel(user_mel_pred_norm)
        user_mel_bigvgan = user_mel_raw.transpose(1, 2)
        
        with torch.inference_mode():
            user_audio = bigvgan_model(user_mel_bigvgan)
        
        user_audio_cpu = user_audio.squeeze().detach().cpu().numpy()
        
        # Peak normalization
        peak = np.abs(user_audio_cpu).max()
        if peak > 1e-6:
            user_audio_cpu = user_audio_cpu / peak * 0.95
            
        save_path = os.path.join(output_dir, item["filename"])
        sf.write(save_path, user_audio_cpu, sample_rate)
        duration = len(user_audio_cpu) / sample_rate
        print(f"[+] Saved {item['filename']} ({duration:.2f}s) to {save_path}")

    print("\nAll README audio samples generated successfully!")

if __name__ == "__main__":
    generate_samples()
