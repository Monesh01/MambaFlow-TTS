import os
import sys
import torch
import pandas as pd
import soundfile as sf

# Monkey patch torch.load to bypass weights_only restriction in PyTorch 2.6+
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

sys.path.append("/home/monesh/Matcha-TTS")
from matcha.models.matcha_tts import MatchaTTS
from matcha.cli import load_vocoder, to_waveform, process_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# 1. Load Matcha Model
print("Loading Matcha-TTS model...", flush=True)
matcha_model = MatchaTTS.load_from_checkpoint(
    "/home/monesh/.local/share/matcha_tts/matcha_ljspeech.ckpt",
    map_location=device
)
matcha_model.eval()
matcha_model.to(device)

# 2. Load Matcha HiFi-GAN Vocoder
print("Loading HiFi-GAN vocoder...", flush=True)
vocoder, denoiser = load_vocoder(
    "hifigan_T2_v1",
    "/home/monesh/.local/share/matcha_tts/hifigan_T2_v1",
    device
)

# 3. Read first row from val.csv
val_csv_path = "/home/monesh/ljspeech/LJSpeech-1.1/val.csv"
df = pd.read_csv(val_csv_path)

first_row = df.iloc[0]
text = str(first_row.get("normalized_text", first_row.get("text", "")))
print(f"\n--- Processing Sample 0 from val.csv ---")
print(f"Text: '{text}'", flush=True)

# Process text with Matcha's official pipeline (including intersperse(..., 0))
text_processed = process_text(0, text, device)

with torch.no_grad():
    # Synthesise mel
    output = matcha_model.synthesise(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=10,  # standard Matcha ODE steps
        temperature=0.667,
        spks=None,
        length_scale=0.95,
    )
    
    # Vocode to waveform
    waveform = to_waveform(output["mel"], vocoder, denoiser, denoiser_strength=0.00025)

audio_numpy = waveform.numpy()

# Save to output locations
os.makedirs("/home/monesh/audio_tts", exist_ok=True)
out_path1 = "/home/monesh/audio_tts/matcha_val_sample_0.wav"
out_path2 = "/home/monesh/TTSModel/TTS_test/matcha_val_sample_0.wav"

sf.write(out_path1, audio_numpy, 22050, "PCM_24")
sf.write(out_path2, audio_numpy, 22050, "PCM_24")

print(f"\n[+] Audio generated successfully!")
print(f"Saved to: {out_path1}")
print(f"Saved to: {out_path2}")
print(f"Audio Duration: {len(audio_numpy)/22050:.2f} seconds")
