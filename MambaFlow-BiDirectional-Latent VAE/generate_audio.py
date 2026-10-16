import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import inspect

def enable_mamba_cpu_fallback():
    """Enables pure PyTorch CPU fallbacks for Mamba2, CausalConv1d, and SSD kernels when CUDA is unavailable."""
    try:
        import causal_conv1d
        import causal_conv1d.causal_conv1d_interface as cci
        import mamba_ssm.ops.selective_scan_interface as ssi
        import mamba_ssm.ops.triton.layernorm_gated as lng
        import mamba_ssm.ops.triton.ssd_combined as ssd
        import mamba_ssm.modules.mamba2 as m2

        causal_conv1d.causal_conv1d_fn = cci.causal_conv1d_ref
        cci.causal_conv1d_fn = cci.causal_conv1d_ref
        ssd.causal_conv1d_fn = cci.causal_conv1d_ref
        ssi.selective_scan_fn = ssi.selective_scan_ref
        lng.rmsnorm_fn = lng.rms_norm_ref
        ssd.rmsnorm_fn = lng.rms_norm_ref

        ref_keys = set(inspect.signature(ssd.mamba_split_conv1d_scan_ref).parameters.keys())
        def cpu_scan(*args, **kwargs):
            filtered = {k: v for k, v in kwargs.items() if k in ref_keys}
            return ssd.mamba_split_conv1d_scan_ref(*args, **filtered)
        m2.mamba_split_conv1d_scan_combined = cpu_scan
    except Exception:
        pass

if not torch.cuda.is_available():
    enable_mamba_cpu_fallback()

from TTSDataModule import TTSMODEL
from preprocessing.text import text_to_sequence
from stats import denormalize_latent


def find_latest_checkpoint(ckpt_dir="TTS_checkpoints"):
    """
    Auto-detects the newest checkpoint in the checkpoint directory.
    Prioritizes the most recently updated *last.ckpt or latest modified .ckpt file.
    """
    if not os.path.exists(ckpt_dir):
        return None

    candidates = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".ckpt") and not f.endswith(".tmp")
    ]
    if not candidates:
        return None

    # Prioritize active *last.ckpt files
    last_ckpts = [f for f in candidates if f.endswith("last.ckpt")]
    if last_ckpts:
        last_ckpts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return last_ckpts[0]

    # Otherwise sort all .ckpt files by modification time descending
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


def setup_dacvae(model_name="facebook/dacvae-watermarked", device="cuda"):
    """
    Loads DACVAE (128-dim continuous latent codec) decoder.
    When device is 'cuda', loads in float16 to conserve VRAM (~450MB footprint)
    so it runs concurrently with active training without causing OOM.
    """
    from dacvae import DACVAE
    print(f"[+] Loading DACVAE decoder ({model_name}) on device: {device}")
    
    if device == "cuda" and torch.cuda.is_available():
        try:
            model = DACVAE.load(model_name).to("cuda").half().eval()
            print("[+] DACVAE loaded on CUDA with FP16 precision (memory efficient).")
            return model, 48000, "cuda"
        except Exception as e:
            print(f"[!] Warning: Failed loading DACVAE on CUDA ({e}). Falling back to CPU.")
            model = DACVAE.load(model_name).to("cpu").eval()
            return model, 48000, "cpu"
    else:
        model = DACVAE.load(model_name).to("cpu").eval()
        return model, 48000, "cpu"


def decode_latent_to_audio(dacvae_decoder, latent_tensor, target_device="cuda"):
    """
    Decodes [1, 128, T] latent tensor into 48kHz audio waveform.
    Includes automatic fallback to CPU if CUDA memory spikes.
    """
    model_device = next(dacvae_decoder.parameters()).device
    model_dtype = next(dacvae_decoder.parameters()).dtype

    latent_in = latent_tensor.to(device=model_device, dtype=model_dtype)

    with torch.inference_mode():
        try:
            out_audio = dacvae_decoder.decode(latent_in)
            wav = out_audio.squeeze().float().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            print("[!] CUDA OOM during DACVAE decode. Gracefully falling back to CPU decoding...")
            torch.cuda.empty_cache()
            from dacvae import DACVAE
            cpu_decoder = DACVAE.load("facebook/dacvae-watermarked").to("cpu").eval()
            out_audio = cpu_decoder.decode(latent_tensor.to("cpu", dtype=torch.float32))
            wav = out_audio.squeeze().float().cpu().numpy()

    # Peak normalization to prevent clipping and ensure clean audio
    peak = np.max(np.abs(wav))
    if peak > 1e-4:
        wav = wav * (0.90 / max(peak, 0.90))
    wav = np.clip(wav, -1.0, 1.0)
    return wav


def synthesize_text(
    model,
    text,
    dacvae_decoder,
    device="cuda",
    timesteps=30,
    temperature=0.667,
    length_scale=1.0,
    solver="euler",
    velocity_scale=1.0,
    cfg_scale=1.0,
):
    """
    Synthesizes speech waveform directly from an input text string using:
    1. Pure Transformer text encoder + Duration predictor
    2. Continuous flow direct Euler ODE solver predicting 128D DACAE latent with CFG support
    3. Stats-based Z-score denormalization back to raw latent space
    4. 48kHz neural audio codec decoding via DACVAE
    """
    cleaner_names = ["english_cleaners2"]
    seq = text_to_sequence(text, cleaner_names)[0]
    text_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    text_mask = torch.ones((1, text_tensor.size(1)), dtype=torch.bool, device=device)

    t_start = time.time()
    with torch.inference_mode():
        # Predict normalized 128D DACAE latent [1, T, 128] via direct Euler ODE with CFG
        pred_latent, _ = model(
            x=text_tensor,
            target_latent=None,
            mask=text_mask,
            n_timesteps=timesteps,
            temperature=temperature,
            length_scale=length_scale,
            solver=solver,
            velocity_scale=velocity_scale,
            cfg_scale=cfg_scale,
        )

        # Denormalize from standardized space back to raw DACAE latent space
        pred_latent_denorm = denormalize_latent(pred_latent)

        # Transpose to [1, 128, T] for DACVAE decoder
        latent_128_t = pred_latent_denorm.transpose(1, 2)
        
        # Decode to 48kHz audio waveform
        wav = decode_latent_to_audio(dacvae_decoder, latent_128_t, target_device=device)

    t_total = time.time() - t_start
    n_frames = pred_latent.size(1)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return wav, n_frames, t_total


def main():
    parser = argparse.ArgumentParser(
        description="MambaFlow-TTS 128D DACAE Audio Generation (48 kHz) - Fast & VRAM-Safe"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="TTS_checkpoints/mambaflow_ca_bimamba_256d_epoch_141_loss_0.9500.ckpt",
        help="Path to model checkpoint (default: auto-detect latest checkpoint in TTS_checkpoints/)",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="TTS_checkpoints",
        help="Directory to search for latest checkpoint (default: TTS_checkpoints)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="/home/monesh/ljspeech/LJSpeech-1.1/val.csv",
        help="Path to validation CSV for evaluation (default: val.csv)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of evaluation samples to generate from CSV (default: 5)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Custom text prompt from user to synthesize into audio",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive prompt loop to input custom text",
    )
    parser.add_argument(
        "--timesteps",
        "--steps",
        dest="timesteps",
        type=int,
        default=30,
        help="Number of ODE solver timesteps (default: 30)",
    )
    parser.add_argument(
        "--temp",
        "--temperature",
        dest="temp",
        type=float,
        default=0.667,
        help="Noise temperature (default: 0.667)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="euler",
        choices=["euler", "midpoint"],
        help="ODE numerical integration solver: 'euler' (default) or 'midpoint'",
    )
    parser.add_argument(
        "--scale",
        "--length_scale",
        dest="scale",
        type=float,
        default=1.0,
        help="Length scale factor for duration predictor (default: 1.0)",
    )
    parser.add_argument(
        "--velocity_scale",
        type=float,
        default=1.0,
        help="Inference velocity scaling factor (default: 1.0)",
    )
    parser.add_argument(
        "--cfg",
        "--cfg_scale",
        dest="cfg_scale",
        type=float,
        default=1.0,
        help="Classifier-Free Guidance (CFG) scale (e.g. 1.5 - 2.5; default: 1.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run MambaFlow model on (default: cuda)",
    )
    parser.add_argument(
        "--dacvae_device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run DACVAE decoder on (default: cuda in fp16)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_audio",
        help="Output directory for generated audio files (default: generated_audio/)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Resolve Checkpoint
    ckpt_path = args.ckpt
    if ckpt_path is None or not os.path.exists(ckpt_path):
        detected_ckpt = find_latest_checkpoint(args.ckpt_dir)
        if detected_ckpt is None:
            raise FileNotFoundError(
                f"No checkpoint found in '{args.ckpt_dir}'. Please provide a valid checkpoint path via --ckpt."
            )
        ckpt_path = detected_ckpt
        print(f"[+] Auto-detected latest checkpoint: {ckpt_path}")
    else:
        print(f"[+] Using specified checkpoint: {ckpt_path}")

    # Checkpoint timestamp
    mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(ckpt_path)))

    print("=" * 75)
    print("  MAMBAFLOW-TTS AUDIO GENERATION (48 kHz DACAE CONTINUOUS LATENT)")
    print("=" * 75)
    print(f"  Checkpoint Path:     {ckpt_path}")
    print(f"  Checkpoint Date:     {mtime}")
    print(f"  MambaFlow Device:    {args.device}")
    print(f"  DACVAE Device:       {args.dacvae_device} (FP16 VRAM-Safe)")
    print(f"  ODE Solver:          {args.solver.upper()}")
    print(f"  ODE Steps:           {args.timesteps}")
    print(f"  Noise Temperature:   {args.temp}")
    print(f"  Length Scale:        {args.scale}")
    print(f"  CFG Scale:           {args.cfg_scale}")
    print(f"  Output Directory:    {args.output_dir}/")
    print("=" * 75)

    # 2. Load Model
    print(f"[+] Loading MambaFlow model from checkpoint...")
    pl_module = TTSMODEL.load_from_checkpoint(ckpt_path, map_location=args.device)
    eval_model = pl_module.model.to(args.device)
    eval_model.eval()

    # 3. Load DACVAE Audio Decoder
    dacvae_decoder, sr, actual_dacvae_device = setup_dacvae(
        model_name="facebook/dacvae-watermarked",
        device=args.dacvae_device,
    )

    # 4. Handle Custom User Text if provided
    if args.text is not None:
        print(f"\n[+] Synthesizing custom user text: \"{args.text}\"")
        wav, n_frames, gen_time = synthesize_text(
            model=eval_model,
            text=args.text,
            dacvae_decoder=dacvae_decoder,
            device=args.device,
            timesteps=args.timesteps,
            temperature=args.temp,
            length_scale=args.scale,
            solver=args.solver,
            velocity_scale=args.velocity_scale,
            cfg_scale=args.cfg_scale,
        )
        audio_dur = len(wav) / sr
        rtf = gen_time / audio_dur if audio_dur > 0 else 0
        out_wav_path = os.path.join(args.output_dir, "custom_user_prompt.wav")
        sf.write(out_wav_path, wav, sr)
        print(f"[+] Successfully saved: {out_wav_path}")
        print(f"    Duration: {audio_dur:.2f}s | Latent Frames: {n_frames} | Gen Time: {gen_time:.2f}s (RTF: {rtf:.3f})")
        return

    # 5. Handle Interactive Mode
    if args.interactive:
        print("\n" + "=" * 75)
        print("  INTERACTIVE TEXT-TO-SPEECH GENERATION MODE")
        print("  (Type your text and press Enter. Type 'exit' or 'quit' to finish)")
        print("=" * 75)
        idx = 1
        while True:
            try:
                user_input = input(f"\nPrompt [{idx}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[+] Exiting interactive mode.")
                break
            if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                print("[+] Exiting interactive mode.")
                break

            wav, n_frames, gen_time = synthesize_text(
                model=eval_model,
                text=user_input,
                dacvae_decoder=dacvae_decoder,
                device=args.device,
                timesteps=args.timesteps,
                temperature=args.temp,
                length_scale=args.scale,
                solver=args.solver,
                velocity_scale=args.velocity_scale,
                cfg_scale=args.cfg_scale,
            )
            audio_dur = len(wav) / sr
            rtf = gen_time / audio_dur if audio_dur > 0 else 0
            out_wav_path = os.path.join(args.output_dir, f"interactive_{idx:03d}.wav")
            sf.write(out_wav_path, wav, sr)
            print(f"    Saved -> {out_wav_path} ({audio_dur:.2f}s | Gen Time: {gen_time:.2f}s | RTF: {rtf:.3f})")
            idx += 1
        return

    # 6. Default: Synthesize 5 Samples from val.csv
    if not os.path.exists(args.csv):
        print(f"[!] Validation CSV not found at: {args.csv}")
        return

    df = pd.read_csv(args.csv)
    print(f"\n[+] Synthesizing default first {args.num_samples} evaluation samples from: {args.csv}")

    results = []
    for idx, row in df.head(args.num_samples).iterrows():
        audio_id = row.get("audio_id", f"sample_{idx+1}")
        text = str(row.get("normalized_text", row.get("text", "")))
        latent_path = row.get("latent", row.get("mel_path", None))

        print(f"\n--- [{idx+1}/{args.num_samples}] Sample: {audio_id} ---")
        print(f"Text: \"{text}\"")

        # Synthesize audio with MambaFlow + DACVAE
        pred_wav, pred_frames, gen_time = synthesize_text(
            model=eval_model,
            text=text,
            dacvae_decoder=dacvae_decoder,
            device=args.device,
            timesteps=args.timesteps,
            temperature=args.temp,
            length_scale=args.scale,
            solver=args.solver,
            velocity_scale=args.velocity_scale,
            cfg_scale=args.cfg_scale,
        )

        pred_dur = len(pred_wav) / sr
        rtf = gen_time / pred_dur if pred_dur > 0 else 0
        pred_wav_path = os.path.join(args.output_dir, f"{idx+1:02d}_{audio_id}_pred.wav")
        sf.write(pred_wav_path, pred_wav, sr)

        gt_frames = None
        dur_ratio = None
        gt_wav_path = None

        # Decode Ground-Truth latent if available for direct comparison
        if latent_path and os.path.exists(latent_path):
            try:
                raw_latent = np.squeeze(np.load(latent_path))
                if raw_latent.ndim == 2:
                    if raw_latent.shape[0] != 128 and raw_latent.shape[1] == 128:
                        raw_latent = raw_latent.T
                    gt_frames = int(raw_latent.shape[1])
                    dur_ratio = float(pred_frames / gt_frames)

                    gt_tensor = torch.tensor(raw_latent, dtype=torch.float32).unsqueeze(0)
                    gt_wav = decode_latent_to_audio(dacvae_decoder, gt_tensor, target_device=args.device)
                    gt_wav_path = os.path.join(args.output_dir, f"{idx+1:02d}_{audio_id}_gt.wav")
                    sf.write(gt_wav_path, gt_wav, sr)
            except Exception as e:
                print(f"    [!] Note: Could not decode GT latent ({e})")

        dur_str = f"DurRatio={dur_ratio:.2f} ({pred_frames} vs GT {gt_frames})" if dur_ratio else f"Frames={pred_frames}"
        print(f"    -> Generated: {pred_wav_path} ({pred_dur:.2f}s | {dur_str} | RTF: {rtf:.3f})")
        if gt_wav_path:
            print(f"    -> Ground-Truth: {gt_wav_path}")

        results.append({
            "sample_index": idx + 1,
            "audio_id": audio_id,
            "text": text,
            "pred_frames": pred_frames,
            "gt_frames": gt_frames,
            "pred_duration_sec": round(pred_dur, 2),
            "generation_time_sec": round(gen_time, 2),
            "rtf": round(rtf, 3),
            "duration_ratio": round(dur_ratio, 3) if dur_ratio is not None else None,
            "pred_wav": pred_wav_path,
            "gt_wav": gt_wav_path,
        })

    # Save manifest summary
    manifest_path = os.path.join(args.output_dir, "generation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "checkpoint": ckpt_path,
                "checkpoint_date": mtime,
                "settings": {
                    "solver": args.solver,
                    "timesteps": args.timesteps,
                    "temperature": args.temp,
                    "length_scale": args.scale,
                    "device": args.device,
                    "dacvae_device": actual_dacvae_device,
                    "csv": args.csv,
                    "num_samples": args.num_samples,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 75)
    print(f"[+] All {args.num_samples} samples generated successfully in: {args.output_dir}/")
    print(f"[+] Manifest summary written to: {manifest_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
