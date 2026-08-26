import os
import sys
import torch
import soundfile as sf

# Monkey patch torch.load
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

sys.path.append("/home/monesh/Matcha-TTS")
from matcha.cli import load_matcha, load_vocoder
from preprocessing.text import text_to_sequence

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Matcha TTS...", flush=True)
matcha = load_matcha("matcha_ljspeech", "/home/monesh/.local/share/matcha_tts/matcha_ljspeech.ckpt", device)

print("Loading Matcha Vocoder (HiFi-GAN)...", flush=True)
vocoder, denoiser = load_vocoder("hifigan_T2_v1", "/home/monesh/.local/share/matcha_tts/hifigan_T2_v1", device)

text = "Hello, this is a test audio generated using the native Matcha vocoder, instead of the BigVGAN vocoder."
seq, _ = text_to_sequence(text, ["english_cleaners2"])
x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
x_lengths = torch.tensor([len(seq)], dtype=torch.long, device=device)

print("Synthesising Mel...", flush=True)
with torch.no_grad():
    output = matcha.synthesise(x, x_lengths, n_timesteps=25, temperature=0.667, spks=None)
    mel = output['mel']
    
print("Vocoding...", flush=True)
with torch.no_grad():
    audio = vocoder(mel).squeeze(1)
    
    if denoiser is not None:
        audio = denoiser(audio, 0.00025)
    
audio_numpy = audio.cpu().squeeze().numpy()

output_path = "/home/monesh/audio_tts/matcha test sample 1.wav"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
sf.write(output_path, audio_numpy, 22050)
print(f"Saved to {output_path}")
