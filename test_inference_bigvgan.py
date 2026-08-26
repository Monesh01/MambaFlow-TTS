import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import json
from einops import rearrange
from mamba_ssm import Mamba2
from preprocessing.text import text_to_sequence
from TTSDataModule import TTSMODEL
from TTSDatasetModule import denormalize_mel

import bigvgan
from bigvgan import BigVGAN
from bigvgan import AttrDict

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
    output_dir = "/home/monesh/audio_samples-ttts"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}", flush=True)

    # 1. Load TTS Model Checkpoint
    ckpt_path = "/home/monesh/TTSModel/TTS_checkpoints/TTSmodel-cfm-v13-continue-epoch=99-val_loss=0.3275.ckpt"
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
    print(f"BigVGAN Vocoder ready (Sample Rate: {sample_rate} Hz)", flush=True)
    print(f"Output directory: {output_dir}\n", flush=True)

    # 3. Process Text and Generate Audio for each sentence across multiple steps
    test_sentences = [
        "Hello there! This is a quick test of the short sentence.",
        "that not more than one bottle of wine or one quart of beer could be issued at one time. No account was taken of the amount of liquors admitted in one day, ",
        "It had established periodic regular review of the status of four hundred individuals; ",
        "Hey! Hello, I am Monesh. Welcome to my TTS model test. Ready! 1, 2, 3 Lets start",
        "Generative flow matching models produce fast and high quality speech synthesis, combining the strength of normalizing flows and continuous time diffusion models to achieve unprecedented performance in text to speech applications. This enables real-time synthesis of human-like voice with various accents and emotions without any issues."
    ]

    steps_list = [25]
    benchmark_results = []

    # Warmup GPU
    print("Performing GPU warmup...", flush=True)
    dummy_seq, _ = text_to_sequence("Warmup run", ["english_cleaners2"])
    dummy_x = torch.tensor(dummy_seq, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        dummy_mel, _ = model(x=dummy_x, target_latent=None, n_timesteps=5, temperature=0.667)
        dummy_raw = denormalize_mel(dummy_mel).transpose(1, 2)
        _ = bigvgan_model(dummy_raw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Warmup complete.\n", flush=True)

    for idx, text in enumerate(test_sentences, 1):
        print(f"================================================================================", flush=True)
        print(f"Sentence {idx}/{len(test_sentences)}: '{text}'", flush=True)
        print(f"================================================================================", flush=True)

        # Convert text to IPA sequence
        seq, ipa_text = text_to_sequence(text, ["english_cleaners2"])
        print(f"IPA Output: {ipa_text}", flush=True)
        x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

        for steps in steps_list:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 1. Flow Matching ODE
            tts_start = time.perf_counter()
            mel_pred_norm, latent_expanded = model(
                x=x,
                target_latent=None,
                n_timesteps=steps,
                temperature=0.3,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            tts_end = time.perf_counter()

            # 2. Denormalize mel
            mel_raw = denormalize_mel(mel_pred_norm)
            mel_bigvgan = mel_raw.transpose(1, 2)

            # 3. BigVGAN Vocoder
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            vocoder_start = time.perf_counter()
            with torch.inference_mode():
                audio = bigvgan_model(mel_bigvgan)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            vocoder_end = time.perf_counter()

            audio_cpu = audio.squeeze().detach().cpu().numpy()
            audio_duration = len(audio_cpu) / sample_rate
            tts_time = tts_end - tts_start
            vocoder_time = vocoder_end - vocoder_start
            total_time = tts_time + vocoder_time

            tts_rtf = tts_time / audio_duration
            vocoder_rtf = vocoder_time / audio_duration
            total_rtf = total_time / audio_duration
            speed_factor = audio_duration / total_time if total_time > 0 else 0

            # Peak normalization
            peak = np.abs(audio_cpu).max()
            if peak > 1e-6:
                audio_cpu = audio_cpu / peak * 0.95

            out_filename = f"sentence_{idx}_steps_{steps}.wav"
            out_path = os.path.join(output_dir, out_filename)
            sf.write(out_path, audio_cpu, sample_rate)

            res = {
                "sentence_idx": idx,
                "steps": steps,
                "text": text,
                "audio_duration_sec": audio_duration,
                "tts_time_sec": tts_time,
                "vocoder_time_sec": vocoder_time,
                "total_time_sec": total_time,
                "tts_rtf": tts_rtf,
                "vocoder_rtf": vocoder_rtf,
                "total_rtf": total_rtf,
                "speed_factor_x": speed_factor,
                "file_path": out_path
            }
            benchmark_results.append(res)

            print(
                f"[Steps: {steps:2d}] Audio: {audio_duration:5.2f}s | "
                f"TTS: {tts_time*1000:6.1f}ms (RTF: {tts_rtf:0.4f}) | "
                f"Vocoder: {vocoder_time*1000:6.1f}ms (RTF: {vocoder_rtf:0.4f}) | "
                f"Total: {total_time*1000:6.1f}ms (Total RTF: {total_rtf:0.4f}) | "
                f"Speed: {speed_factor:5.1f}x RT | Saved: {out_filename}",
                flush=True
            )

        print()

    # Save detailed JSON results
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    # Print Consolidated Benchmark Table
    print("\n" + "="*100)
    print("                              TTS BENCHMARKING SUMMARY TABLE")
    print("="*100)
    print(f"{'Sent':<5} {'Steps':<6} {'Audio(s)':<9} {'TTS Time(ms)':<13} {'Voc Time(ms)':<13} {'Tot Time(ms)':<13} {'TTS RTF':<10} {'Tot RTF':<10} {'Speed Factor':<12}")
    print("-"*100)
    for r in benchmark_results:
        print(
            f"{r['sentence_idx']:<5} "
            f"{r['steps']:<6} "
            f"{r['audio_duration_sec']:<9.2f} "
            f"{r['tts_time_sec']*1000:<13.1f} "
            f"{r['vocoder_time_sec']*1000:<13.1f} "
            f"{r['total_time_sec']*1000:<13.1f} "
            f"{r['tts_rtf']:<10.4f} "
            f"{r['total_rtf']:<10.4f} "
            f"{r['speed_factor_x']:<12.1f}x"
        )
    print("="*100)

    # Step-wise Averages
    print("\n" + "="*80)
    print("                        STEP-WISE AVERAGE BENCHMARK")
    print("="*80)
    print(f"{'Steps':<8} {'Avg Audio(s)':<14} {'Avg TTS(ms)':<14} {'Avg Total(ms)':<14} {'Avg RTF':<12} {'Avg Speed':<12}")
    print("-"*80)
    for s in steps_list:
        step_runs = [r for r in benchmark_results if r['steps'] == s]
        avg_dur = np.mean([r['audio_duration_sec'] for r in step_runs])
        avg_tts = np.mean([r['tts_time_sec']*1000 for r in step_runs])
        avg_tot = np.mean([r['total_time_sec']*1000 for r in step_runs])
        avg_rtf = np.mean([r['total_rtf'] for r in step_runs])
        avg_speed = np.mean([r['speed_factor_x'] for r in step_runs])
        print(f"{s:<8} {avg_dur:<14.2f} {avg_tts:<14.1f} {avg_tot:<14.1f} {avg_rtf:<12.4f} {avg_speed:<12.1f}x")
    print("="*80)
    print(f"\nAll audio files and benchmark results saved to: {output_dir}\n")

if __name__ == "__main__":
    run_inference()
