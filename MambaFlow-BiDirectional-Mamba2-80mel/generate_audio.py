import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

# Ensure local codebase and HiFi-GAN are on Python path
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(CURRENT_DIR)

from TTSDataModule import TTSMODEL
from TTSDatasetModule import load_normalization_stats
from preprocessing.text import text_to_sequence


def setup_hifigan(hifigan_dir, hifigan_config_path, hifigan_ckpt_path, device):
    """
    Loads HiFi-GAN v1 vocoder from repository.
    """
    # Try alternate hifi-gan path if specified dir does not exist
    if not os.path.isdir(hifigan_dir):
        alt_dir = "/home/monesh/hifi-gan"
        if os.path.isdir(alt_dir):
            hifigan_dir = alt_dir

    if hifigan_dir not in sys.path:
        sys.path.insert(0, hifigan_dir)

    from models import Generator
    from types import SimpleNamespace

    # Locate config
    if not os.path.exists(hifigan_config_path):
        for candidate in [
            os.path.join(hifigan_dir, "config_v1.json"),
            "/home/monesh/hifi-gan/config_v1.json",
            "/home/monesh/hifigan/config_v1.json",
        ]:
            if os.path.exists(candidate):
                hifigan_config_path = candidate
                break

    print(f"[+] Loading HiFi-GAN config from: {hifigan_config_path}")
    with open(hifigan_config_path, "r") as f:
        h = SimpleNamespace(**json.load(f))

    generator = Generator(h).to(device)

    # Locate checkpoint
    if not os.path.exists(hifigan_ckpt_path):
        for candidate in [
            "/home/monesh/ljspeech/LJSpeech-1.1/generator_v1",
            os.path.join(hifigan_dir, "LJ_V1"),
            "/home/monesh/hifi-gan/LJ_V1",
        ]:
            if os.path.exists(candidate):
                hifigan_ckpt_path = candidate
                break

    print(f"[+] Loading HiFi-GAN weights from: {hifigan_ckpt_path}")
    checkpoint = torch.load(hifigan_ckpt_path, map_location=device)
    if "generator" in checkpoint:
        generator.load_state_dict(checkpoint["generator"])
    else:
        generator.load_state_dict(checkpoint)

    generator.eval()
    try:
        generator.remove_weight_norm()
    except Exception:
        pass

    return generator, h


def synthesize_text(
    model,
    text,
    vocoder,
    mean_t,
    std_t,
    device,
    timesteps=30,
    temperature=0.667,
    length_scale=1.0,
    solver="midpoint",
    cfg_scale=1.0,
):
    """
    Synthesizes speech waveform directly from an input text string.
    """
    cleaner_names = ["english_cleaners2"]
    seq = text_to_sequence(text, cleaner_names)[0]
    text_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    text_mask = torch.ones((1, text_tensor.size(1)), dtype=torch.bool, device=device)

    with torch.no_grad():
        # Generate normalized mel spectrogram [1, T, 80]
        pred_mel_norm, _ = model(
            x=text_tensor,
            target_latent=None,
            mask=text_mask,
            n_timesteps=timesteps,
            temperature=temperature,
            length_scale=length_scale,
            solver=solver,
            #cfg_scale=cfg_scale,
        )

        # Denormalize: mel = norm * std + mean
        pred_mel_denorm = pred_mel_norm * std_t + mean_t  # [1, T, 80]
        # Transpose to [1, 80, T] for HiFi-GAN
        mel_hifigan = pred_mel_denorm.transpose(1, 2)

        denoising_mel = torch.zeros_like(mel_hifigan)
        # Vocode to waveform [1, 1, T_audio]
        wav = vocoder(mel_hifigan).squeeze().cpu().numpy()
        denoise = vocoder(denoising_mel).squeeze().cpu().numpy()

        wav = wav - (denoise * 0.00025)
    wav = np.clip(wav, -1.0, 1.0)
    return wav, pred_mel_norm.size(1)


def main():
    parser = argparse.ArgumentParser(
        description="MambaFlow-TTS HiFi-GAN v1 Audio Generation (80 mel bins, 22.05 kHz)"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="TTS_checkpoints/mambaflow_hifigan_val_epoch_118_loss_0.3313.ckpt",
        help="Path to MambaFlow model checkpoint",
    )
    parser.add_argument(
        "--hifigan_dir",
        type=str,
        default="/home/monesh/hifigan",
        help="Path to HiFi-GAN repository directory",
    )
    parser.add_argument(
        "--hifigan_ckpt",
        type=str,
        default="/home/monesh/ljspeech/LJSpeech-1.1/generator_v1",
        help="Path to HiFi-GAN generator checkpoint",
    )
    parser.add_argument(
        "--hifigan_config",
        type=str,
        default="/home/monesh/hifigan/config_v1.json",
        help="Path to HiFi-GAN config_v1.json",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="/home/monesh/ljspeech/LJSpeech-1.1/stats_hifigan.py",
        help="Path to normalization stats file (stats_hifigan.py)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="/home/monesh/ljspeech/LJSpeech-1.1/val_hifigan.csv",
        help="Path to validation CSV for batch evaluation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of evaluation samples to generate from CSV",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optional single custom text prompt to synthesize",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=30,
        help="Number of ODE solver timesteps (default: 30)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.667,
        help="Noise temperature (default: 0.667)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Length scale for duration predictor (default: 1.0)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="midpoint",
        choices=["midpoint", "euler"],
        help="ODE numerical integration solver (default: midpoint)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="Classifier-Free Guidance scale (1.0 = standard single pass; >1.0 = guided)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_audio_hifigan",
        help="Output directory for generated audio files",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        # Safe VRAM cap so generation does not trigger OOM while background training is active
        try:
            torch.cuda.set_per_process_memory_fraction(0.20, 0)
        except Exception:
            pass
        torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Check and locate MambaFlow checkpoint
    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        candidates = [
            os.path.join("TTS_checkpoints", f)
            for f in os.listdir("TTS_checkpoints")
            if f.endswith(".ckpt") and "hifigan" in f
        ]
        if candidates:
            candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            ckpt_path = candidates[0]
            print(f"[!] Target checkpoint not found, defaulting to newest: {ckpt_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    print("=" * 70)
    print("  MambaFlow-TTS HiFi-GAN v1 Audio Generation")
    print("=" * 70)
    print(f"  Model Checkpoint:     {ckpt_path}")
    print(f"  ODE Solver:           {args.solver.upper()}")
    print(f"  ODE Steps:            {args.timesteps}")
    print(f"  Noise Temperature:    {args.temp}")
    print(f"  Length Scale:         {args.scale}")
    print(f"  CFG Scale:            {args.cfg_scale}")
    print(f"  Device:               {device}")
    print(f"  Output Directory:     {args.output_dir}/")
    print("=" * 70)

    # 2. Load MambaFlow-TTS Model
    print(f"[+] Loading MambaFlow model from checkpoint...")
    pl_module = TTSMODEL.load_from_checkpoint(ckpt_path).to(device)
    eval_model = pl_module.model
    eval_model.eval()
    eval_model.to(device)

    # 3. Load HiFi-GAN Vocoder
    generator, h = setup_hifigan(
        args.hifigan_dir, args.hifigan_config, args.hifigan_ckpt, device
    )
    sr = h.sampling_rate

    # 4. Load 80-mel normalization statistics
    mean, std = load_normalization_stats(args.stats_path, dim=80)
    mean_t = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 1, 80)
    std_t = torch.tensor(std, device=device, dtype=torch.float32).view(1, 1, 80)

    # 5. Handle single text synthesis if provided
    if args.text is not None:
        print(f"\n[+] Synthesizing custom text: \"{args.text}\"")
        wav, n_frames = synthesize_text(
            model=eval_model,
            text=args.text,
            vocoder=generator,
            mean_t=mean_t,
            std_t=std_t,
            device=device,
            timesteps=args.timesteps,
            temperature=args.temp,
            length_scale=args.scale,
            solver=args.solver,
            #cfg_scale=args.cfg_scale,
        )
        out_wav_path = os.path.join(args.output_dir, "custom_text_pred.wav")
        sf.write(out_wav_path, wav, sr)
        print(f"[+] Successfully generated custom audio ({len(wav)/sr:.2f}s, {n_frames} frames): {out_wav_path}")
        return

    # 6. Synthesize samples from validation CSV
    if not os.path.exists(args.csv):
        print(f"[!] Validation CSV not found at {args.csv}. Exiting.")
        return

    df = pd.read_csv(args.csv)
    print(f"\n[+] Synthesizing first {args.num_samples} evaluation samples from {args.csv}...")

    results = []
    for idx, row in df.head(args.num_samples).iterrows():
        audio_id = row.get("audio_id", f"sample_{idx+1}")
        text = str(row.get("normalized_text", row.get("text", "")))
        mel_path = row.get("mel_path", None)

        print(f"\n--- [{idx+1}/{args.num_samples}] Sample: {audio_id} ---")
        print(f"Text: \"{text}\"")

        # Synthesize audio with MambaFlow
        pred_wav, pred_frames = synthesize_text(
            model=eval_model,
            text=text,
            vocoder=generator,
            mean_t=mean_t,
            std_t=std_t,
            device=device,
            timesteps=args.timesteps,
            temperature=args.temp,
            length_scale=args.scale,
            solver=args.solver,
            #cfg_scale=args.cfg_scale,
        )

        pred_wav_path = os.path.join(args.output_dir, f"{idx+1:02d}_{audio_id}_pred.wav")
        sf.write(pred_wav_path, pred_wav, sr)

        gt_frames = None
        dur_ratio = None
        gt_wav_path = None

        # If GT mel exists, generate GT vocoded audio for side-by-side comparison
        if mel_path and os.path.exists(mel_path):
            raw_mel = np.squeeze(np.load(mel_path))
            if raw_mel.ndim == 2:
                # Transpose if [T, 80] instead of [80, T]
                if raw_mel.shape[0] != 80 and raw_mel.shape[1] == 80:
                    raw_mel = raw_mel.T
                gt_mel_tensor = torch.tensor(raw_mel, dtype=torch.float32, device=device).unsqueeze(0)
                gt_frames = int(raw_mel.shape[1])
                dur_ratio = float(pred_frames / gt_frames)

                with torch.no_grad():
                    gt_wav = generator(gt_mel_tensor).squeeze().cpu().numpy()
                gt_wav = np.clip(gt_wav, -1.0, 1.0)
                gt_wav_path = os.path.join(args.output_dir, f"{idx+1:02d}_{audio_id}_gt.wav")
                sf.write(gt_wav_path, gt_wav, sr)

        results.append({
            "sample_index": idx + 1,
            "audio_id": audio_id,
            "text": text,
            "pred_frames": pred_frames,
            "gt_frames": gt_frames,
            "duration_ratio": round(dur_ratio, 3) if dur_ratio is not None else None,
            "pred_wav": pred_wav_path,
            "gt_wav": gt_wav_path,
        })

        dur_info = f"DurRatio={dur_ratio:.2f} ({pred_frames} vs GT {gt_frames})" if dur_ratio else f"Frames={pred_frames}"
        print(f"Generated ({len(pred_wav)/sr:.2f}s | {dur_info}) -> {pred_wav_path}")

    summary_file = os.path.join(args.output_dir, "generation_manifest.json")
    with open(summary_file, "w") as f:
        json.dump({"settings": vars(args), "results": results}, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[+] All audio samples successfully written to: {args.output_dir}/")
    print(f"[+] Manifest summary saved to: {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
