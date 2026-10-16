import os
import sys
import json
import numpy as np
import soundfile as sf
import torch
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Protect active training VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.20, 0)

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from TTSDataModule import TTSMODEL
from preprocessing.text import text_to_sequence
from preprocessing.text.cleaners import english_cleaners2
from stats import denormalize_latent
from dacvae import DACVAE


def normalize_audio_to_peak(audio, target_peak=0.85):
    peak = np.max(np.abs(audio))
    if peak > 1e-4:
        audio = audio * (target_peak / peak)
    return np.clip(audio, -1.0, 1.0)


@torch.inference_mode()
def integrate_flow(decoder, cond, mask=None, n_timesteps=30, temperature=0.667, seed=42):
    B, T, _ = cond.size()
    device = cond.device
    dtype = cond.dtype

    torch.manual_seed(seed)
    x = torch.randn(B, T, decoder.d_in, device=device, dtype=dtype) * float(temperature)
    if mask is not None:
        m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
        x = x * m
    else:
        m = None

    dt = 1.0 / float(n_timesteps)

    for step in range(n_timesteps):
        t_curr = step * dt
        t_tensor = torch.full((B,), t_curr, device=device, dtype=dtype)
        v = decoder.predict_velocity(x, t_tensor, cond=cond, mask=mask)
        x = x + dt * v
        if m is not None:
            x = x * m

    return x[0].detach().cpu().float().numpy()


def transcribe_audio(audio_48k, whisper_proc, whisper_model):
    audio_16k = librosa.resample(audio_48k, orig_sr=48000, target_sr=16000)
    feats = whisper_proc(audio_16k, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        ids = whisper_model.generate(feats)
    return whisper_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def main():
    text = "Hey Hi! I am Monesh S, pursuing BTech AIDS in Saveetha University, Chennai."
    print("=" * 80)
    print("  MAMBAFLOW-TTS CUSTOM TEXT SYNTHESIS")
    print(f"  Input Text: \"{text}\"")
    print("=" * 80)

    cleaned_ipa = english_cleaners2(text)
    print(f"[+] Phonemized IPA: {cleaned_ipa}")

    artifact_dir = "/home/monesh/.gemini/antigravity-ide/brain/5d4a0c4f-a414-433e-9d80-eb2abaa626e5"
    os.makedirs(artifact_dir, exist_ok=True)

    # Checkpoint
    best_ckpt = os.path.join(CURRENT_DIR, "TTS_checkpoints/mambaflow_dacae_128d_zscore_epoch_208_loss_0.9434.ckpt")
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(CURRENT_DIR, "TTS_checkpoints/mambaflow_dacae_128d_zscore_last.ckpt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Loading model from: {best_ckpt} (device: {device})")
    pl_model = TTSMODEL.load_from_checkpoint(best_ckpt, map_location=device, strict=False)
    model = pl_model.model.eval().to(device)

    print("[+] Loading DACVAE decoder (cpu)...")
    dacvae = DACVAE.load("facebook/dacvae-watermarked").to("cpu").eval()

    print("[+] Loading Whisper ASR (cpu)...")
    whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").eval().to("cpu")

    # Tokenize input text
    seq = text_to_sequence(text, ["english_cleaners2"])[0]
    text_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    text_mask = torch.ones((1, text_tensor.size(1)), dtype=torch.bool, device=device)

    with torch.no_grad():
        x_in = model.emb_dropout(model.txt_emb(text_tensor))
        enc_out = model.encoder(x_in, mask=text_mask)

        # Predict natural durations
        dur_int, dur_pred, _ = model.duration(enc_out.detach(), mask=text_mask, length_scale=1.0)
        pred_dur = dur_int[0]
        if pred_dur.sum() == 0:
            pred_dur = torch.clamp(pred_dur, min=1)

        total_frames = int(pred_dur.sum().item())
        audio_dur_sec = total_frames * (1920.0 / 48000.0)
        print(f"[+] Predicted Frames: {total_frames} ({audio_dur_sec:.2f}s of speech)")

        text_expanded = torch.repeat_interleave(enc_out[0], pred_dur, dim=0).unsqueeze(0)
        audio_mask = torch.ones((1, text_expanded.size(1)), dtype=torch.bool, device=device)
        if hasattr(model, "mu_smoother") and model.mu_smoother is not None:
            text_expanded = model.mu_smoother(text_expanded, mask=audio_mask)

    # Generate 3 variations with subtle seeds/temperatures so user can pick the best
    variations = [
        {"name": "primary", "temp": 0.667, "seed": 42, "desc": "Standard Optimal (Euler 30, Temp 0.667)"},
        {"name": "natural_v2", "temp": 0.700, "seed": 101, "desc": "Natural Variation (Euler 30, Temp 0.700)"},
        {"name": "crisp_v3", "temp": 0.600, "seed": 2024, "desc": "Crisp Low-Noise (Euler 30, Temp 0.600)"},
    ]

    results = []

    for v in variations:
        print(f"\n[+] Synthesizing Variation: {v['name']} ({v['desc']})...")
        pred_norm = integrate_flow(
            decoder=model.decoder,
            cond=text_expanded,
            mask=audio_mask,
            n_timesteps=30,
            temperature=v["temp"],
            seed=v["seed"],
        )

        pred_denorm = denormalize_latent(pred_norm)
        pred_tensor = torch.tensor(pred_denorm.T, dtype=torch.float32, device="cpu").unsqueeze(0)

        with torch.no_grad():
            raw_audio = dacvae.decode(pred_tensor).squeeze().cpu().numpy()

        audio_loud = normalize_audio_to_peak(raw_audio, target_peak=0.85)
        rms = float(np.sqrt(np.mean(audio_loud**2)))

        # Transcribe with Whisper
        transcription = transcribe_audio(audio_loud, whisper_proc, whisper_model)

        wav_filename = f"monesh_custom_speech_{v['name']}.wav"
        wav_path = os.path.join(artifact_dir, wav_filename)
        sf.write(wav_path, audio_loud, 48000)

        print(f"    Audio saved:  {wav_path}")
        print(f"    Duration:     {len(audio_loud)/48000:.2f}s")
        print(f"    RMS Loudness: {rms:.4f}")
        print(f"    Whisper ASR:  \"{transcription}\"")

        results.append({
            "name": v["name"],
            "desc": v["desc"],
            "temp": v["temp"],
            "seed": v["seed"],
            "wav_filename": wav_filename,
            "wav_path": wav_path,
            "duration": len(audio_loud)/48000,
            "rms": rms,
            "whisper_asr": transcription,
            "audio": audio_loud,
            "latent": pred_norm,
        })

    # Generate a visual plot: Waveforms and Mel Spectrograms
    print("\n[+] Generating visual waveform and spectrogram dashboard...")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 2, figsize=(16, 9), dpi=150, facecolor="#0b0b14")
    fig.patch.set_facecolor("#0b0b14")

    for i, res in enumerate(results):
        ax_wave = axes[i, 0]
        ax_spec = axes[i, 1]
        ax_wave.set_facecolor("#0b0b14")
        ax_spec.set_facecolor("#0b0b14")

        y = res["audio"]
        time_axis = np.linspace(0, len(y) / 48000, len(y))

        # Waveform
        ax_wave.plot(time_axis, y, color="#00e5ff", alpha=0.85, linewidth=0.7)
        ax_wave.set_title(f"{res['name'].capitalize()} Waveform ({res['duration']:.2f}s) | RMS: {res['rms']:.3f}", fontsize=10, fontweight="bold")
        ax_wave.set_ylabel("Amplitude", fontsize=8)
        ax_wave.set_xlabel("Time (seconds)", fontsize=8)
        ax_wave.set_ylim([-1.0, 1.0])
        ax_wave.grid(True, linestyle="--", alpha=0.2)

        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
        img = librosa.display.specshow(D, sr=48000, hop_length=256, x_axis="time", y_axis="hz", ax=ax_spec, cmap="inferno")
        ax_spec.set_title(f"Spectrogram | Whisper: \"{res['whisper_asr'][:45]}...\"", fontsize=10, fontweight="bold")
        ax_spec.set_ylabel("Frequency (Hz)", fontsize=8)
        ax_spec.set_xlabel("Time (seconds)", fontsize=8)

    plt.suptitle(f"MambaFlow-TTS Speech Synthesis: \"{text[:50]}...\"\n3 Generated Audio Variations (48 kHz DACVAE Neural Audio)", fontsize=12, fontweight="bold", color="#ffffff")
    plt.tight_layout()

    plot_path = os.path.join(artifact_dir, "monesh_custom_speech_analysis.png")
    plt.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[+] Saved visualization: {plot_path}")

    # Save summary JSON
    summary_path = os.path.join(artifact_dir, "monesh_custom_speech_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "text": text,
            "phonemes": cleaned_ipa,
            "total_frames": total_frames,
            "duration_sec": audio_dur_sec,
            "variations": [
                {
                    "name": r["name"],
                    "desc": r["desc"],
                    "temp": r["temp"],
                    "seed": r["seed"],
                    "wav_filename": r["wav_filename"],
                    "duration": r["duration"],
                    "rms": r["rms"],
                    "whisper_asr": r["whisper_asr"],
                }
                for r in results
            ]
        }, f, indent=2)

    print("\n[+] Synthesis complete! All audio files and spectrograms saved.")


if __name__ == "__main__":
    main()
