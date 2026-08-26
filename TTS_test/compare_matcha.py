import torch
import time
import os
import sys

# Monkey patch torch.load to bypass weights_only restriction in PyTorch 2.6+
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

sys.path.append("/home/monesh/Matcha-TTS")
from matcha.models.matcha_tts import MatchaTTS
from preprocessing.text import text_to_sequence

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Matcha-TTS...", flush=True)
t0 = time.perf_counter()
model = MatchaTTS.load_from_checkpoint("/home/monesh/.local/share/matcha_tts/matcha_ljspeech.ckpt", map_location=device)
model.eval()
print(f"Matcha-TTS Loaded in {time.perf_counter()-t0:.2f}s", flush=True)
print(f"Matcha Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

test_texts = [
    "This is a very short test.",
    "Hello there! This is a quick test of the short sentence.",
    "Generative flow matching models produce fast and high quality speech synthesis, combining the strength of normalizing flows and continuous time diffusion models to achieve unprecedented performance in text to speech applications."
]

print("Warming up...", flush=True)
with torch.no_grad():
    seq, _ = text_to_sequence(test_texts[0], ["english_cleaners2"])
    x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    x_lengths = torch.tensor([len(seq)], dtype=torch.long, device=device)
    _ = model.synthesise(x, x_lengths, n_timesteps=10, temperature=0.667)

print("Running inference test on Matcha...", flush=True)
for i, t in enumerate(test_texts):
    seq, _ = text_to_sequence(t, ["english_cleaners2"])
    x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    x_lengths = torch.tensor([len(seq)], dtype=torch.long, device=device)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output = model.synthesise(x, x_lengths, n_timesteps=25, temperature=0.3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    mel = output['mel']
    frames = mel.shape[-1]
    audio_dur = frames * 256 / 22050
    rtf = (end - start) / audio_dur
    print(f"Matcha Test {i+1} - RTF: {rtf:.4f} (TTS Time: {end-start:.3f}s, Audio Dur: {audio_dur:.3f}s)")
