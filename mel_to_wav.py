import os
import glob
import json
import torch
import numpy as np
import soundfile as sf

import bigvgan
from bigvgan import BigVGAN
from bigvgan import AttrDict

def mel_to_wav():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running BigVGAN inference on: {device}", flush=True)

    # BigVGAN paths
    config_path = "/home/monesh/bigvgan_model/config_14M.json"
    checkpoint_path = "/home/monesh/bigvgan_model/bigvgan_generator_14M.pt"

    # Load BigVGAN Vocoder
    print("Loading BigVGAN 14M vocoder...", flush=True)
    with open(config_path) as f:
        config = json.load(f)

    h = AttrDict(config)
    
    # Initialize BigVGAN
    bigvgan_model = BigVGAN(h, use_cuda_kernel=False).to(device)

    # Load checkpoint
    bigvgan_checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Safely load the generator state dict
    if "generator" in bigvgan_checkpoint:
        bigvgan_model.load_state_dict(bigvgan_checkpoint["generator"])
    else:
        # Sometimes checkpoints are saved directly as state_dicts
        bigvgan_model.load_state_dict(bigvgan_checkpoint)
        
    bigvgan_model.eval()
    bigvgan_model.remove_weight_norm()
    
    sample_rate = h.sampling_rate
    print(f"BigVGAN Vocoder ready (Sample Rate: {sample_rate} Hz)", flush=True)

    mels_dir = "generated_mels"
    npy_files = glob.glob(os.path.join(mels_dir, "*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {mels_dir}/")
        return

    for npy_path in sorted(npy_files):
        print(f"\nProcessing: {npy_path}", flush=True)
        
        # Load mel-spectrogram [T, 100]
        mel_np = np.load(npy_path)
        
        # Convert to tensor and shape to [1, 100, T] for BigVGAN
        mel_tensor = torch.tensor(mel_np, dtype=torch.float32, device=device)
        if mel_tensor.dim() == 2:
            mel_tensor = mel_tensor.unsqueeze(0) # [1, T, 100]
        
        mel_bigvgan = mel_tensor.transpose(1, 2) # [1, 100, T]

        # Generate audio
        with torch.inference_mode():
            audio = bigvgan_model(mel_bigvgan)

        audio_cpu = audio.squeeze().detach().cpu().numpy()

        # Peak normalization
        peak = np.abs(audio_cpu).max()
        if peak > 1e-6:
            audio_cpu = audio_cpu / peak * 0.95

        # Save audio
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        out_path = os.path.join(mels_dir, f"{base_name}.wav")
        
        sf.write(out_path, audio_cpu, sample_rate)
        print(f"Saved audio to: {out_path}", flush=True)

    print("\nAll Mel-spectrograms successfully converted to audio!")

if __name__ == "__main__":
    mel_to_wav()
