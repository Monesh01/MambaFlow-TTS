import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.text import text_to_sequence
from TTSDataModule import TTSMODEL
from TTSDatasetModule import denormalize_mel

def get_latest_checkpoint(ckpt_dir="TTS_checkpoints/"):
    if not os.path.exists(ckpt_dir):
        return None
    ckpts = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".ckpt") and not f.endswith(".tmp")
    ]
    if not ckpts:
        return None
    named_ckpts = [f for f in ckpts if not os.path.basename(f).startswith("last")]
    if named_ckpts:
        return max(named_ckpts, key=os.path.getmtime)
    return max(ckpts, key=os.path.getmtime)

@torch.no_grad()
def run_inference():
    out_dir = "generated_mels"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}", flush=True)

    ckpt_path = "/home/monesh/TTSModel/TTS_checkpoints/TTSmodel-cfm-v8-epoch=02-val_loss=0.3363.ckpt"
    if not os.path.exists(ckpt_path):
        ckpt_path = get_latest_checkpoint("TTS_checkpoints/")
        
    print(f"Loading TTS model checkpoint: {ckpt_path}", flush=True)

    lightning_model = TTSMODEL.load_from_checkpoint(
        ckpt_path,
        map_location=device,
    )
    lightning_model.eval()
    lightning_model.to(device)

    model = lightning_model.model
    model.eval()

    test_sentences = [
        "It had established periodic regular review of the status of four hundred individuals.",
        "Generative flow matching models produce fast and high quality speech synthesis.",
        "Hello guys! My name is Monesh. What's your name please?",
        "The quick brown fox jumps over the lazy dog.",
        "This is an example of a generated mel-spectrogram without vocoder.",
        "Machine learning is fascinating and continuously evolving.",
        "We are testing the acoustic prior and decoder outputs today.",
        "This is a longer sentence to check how well the duration predictor expands the latent variables over time.",
        "Let's see if the dilations are causing issues in these generated mel spectrogram frames.",
        "A quiet morning is perfectly suited for testing out deep learning models."
    ]

    for idx, text in enumerate(test_sentences, 1):
        print(f"\n--- Generating Sample {idx} ---", flush=True)
        print(f"Input Text: '{text}'", flush=True)

        # Convert text to sequence
        seq, ipa_text = text_to_sequence(text, ["english_cleaners2"])
        x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

        tts_start = time.time()
        # Decreased temperature to 0.667 for cleaner mels
        mel_pred_norm, latent_expanded = model(
            x=x,
            target_latent=None,
            n_timesteps=40,
            temperature=0.667,
        )
        tts_end = time.time()
        
        mel_raw = denormalize_mel(mel_pred_norm)
        # Expected shape: [B, T, 100], squeeze to [T, 100]
        mel_np = mel_raw.squeeze().cpu().numpy()
        
        print(f"Mel shape: {mel_np.shape} | Time: {tts_end - tts_start:.3f}s")
        
        # Save as .npy
        npy_path = os.path.join(out_dir, f"mel_sample_{idx}.npy")
        np.save(npy_path, mel_np)
        
        # Save as PNG plot
        plt.figure(figsize=(10, 4))
        # Transpose so time is x-axis and freq is y-axis
        plt.imshow(mel_np.T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Generated Mel - Sample {idx}\n{text}")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Bins")
        plt.tight_layout()
        png_path = os.path.join(out_dir, f"mel_sample_{idx}.png")
        plt.savefig(png_path)
        plt.close()
        
        print(f"Saved to {npy_path} and {png_path}")
        
    print("\nAll Mel-spectrograms successfully generated and saved!")

if __name__ == "__main__":
    run_inference()
