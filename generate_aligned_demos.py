#!/usr/bin/env python3
"""
Generate properly aligned audio demonstration samples (sample_01 to sample_04)
for all models so they match the Ground Truth reference sentences displayed on the site.
"""

import os
import sys
import json
import shutil
import time
import numpy as np
import soundfile as sf
import torch
from types import SimpleNamespace

ROOT_DIR = "/home/monesh/TTSModel"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from preprocessing.text import text_to_sequence

DEMO_SAMPLES = [
    {
        "id": "sample_01",
        "audio_id": "LJ006-0208",
        "text": "There was also a good supply of Bibles and prayers,",
    },
    {
        "id": "sample_02",
        "audio_id": "LJ003-0014",
        "text": "Quarreling among the debtors was not unfrequent. Blows were struck, and fights often ensued.",
    },
    {
        "id": "sample_03",
        "audio_id": "LJ033-0014",
        "text": "Lee Harvey Oswald lived in a roominghouse in Dallas while his wife and children lived in Irving, at the home of Ruth Paine,",
    },
    {
        "id": "sample_04",
        "audio_id": "LJ029-0194",
        "text": "Any thought that the President might cancel his visit to Dallas was ended",
    },
]

def purge_model_modules():
    for mod in list(sys.modules.keys()):
        if any(mod.startswith(p) for p in [
            'mambaflow', 'TTSDataModule', 'TTSDatasetModule', 'TTS_model',
            'trans_encoder', 'duration_predictor', 'mamba_decoder',
            'mambaflow_tetra_decoder', 'mambaflow_mamba_convnext_decoder'
        ]):
            del sys.modules[mod]

def generate_tetra(device="cuda"):
    print("\n[+] Generating Sequential-Tetra demo samples...")
    purge_model_modules()
    tetra_dir = os.path.join(ROOT_DIR, "MambaFlow-Sequential-Tetra")
    sys.path.insert(0, tetra_dir)
    from TTSDataModule import TTSMODEL
    from TTSDatasetModule import denormalize_mel
    from bigvgan import BigVGAN, AttrDict

    ckpt_tetra = os.path.join(ROOT_DIR, "Checkpoints/MambaFlow-Sequential-Tetra.ckpt")
    lightning_tetra = TTSMODEL()
    sd = torch.load(ckpt_tetra, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("model.", ""): v for k, v in sd.items()}
    lightning_tetra.model.load_state_dict(sd, strict=True)
    model = lightning_tetra.model.to(device).eval()

    # BigVGAN Vocoder
    voc_cfg = "/home/monesh/bigvgan_model/config_14M.json"
    voc_pt = "/home/monesh/bigvgan_model/bigvgan_generator_14M.pt"
    with open(voc_cfg) as f:
        h = AttrDict(json.load(f))
    bigvgan = BigVGAN(h, use_cuda_kernel=False).to(device).eval()
    ckpt_voc = torch.load(voc_pt, map_location="cpu")
    bigvgan.load_state_dict(ckpt_voc.get("generator", ckpt_voc))
    bigvgan.remove_weight_norm()

    out_dirs = [os.path.join(ROOT_DIR, "audio/seq_tetra_100mel"), os.path.join(ROOT_DIR, "docs/audio/seq_tetra_100mel")]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

    for s in DEMO_SAMPLES:
        text = s["text"]
        filename = f"{s['id']}.wav"
        print(f"  -> Tetra synthesizing {filename}: \"{text}\"")
        
        cleaner_names = ["english_cleaners2"]
        seq = text_to_sequence(text, cleaner_names)[0]
        x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            x_enc = model.encoder(model.emb_dropout(model.txt_emb(x)))
            mu_text_base = model.to_latent(x_enc)
            dur_int, _, _ = model.duration(mu_text_base, latent=None, length_scale=1.0)
            durations = torch.clamp(dur_int, min=1)
            mu_expanded = torch.repeat_interleave(mu_text_base[0], durations[0], dim=0).unsqueeze(0)

            n_steps = 30
            dt = 1.0 / float(n_steps)
            x_curr = torch.randn_like(mu_expanded) * 0.667
            for step in range(n_steps):
                t_tensor = torch.full((1,), step * dt, device=device, dtype=torch.float32)
                v = model.decoder.predict_velocity(x_curr, t_tensor, mu_expanded)
                x_curr = x_curr + dt * v

            mel_raw = denormalize_mel(x_curr).transpose(1, 2)
            with torch.inference_mode():
                audio = bigvgan(mel_raw)[0, 0].detach().cpu().numpy()

            peak = np.abs(audio).max()
            if peak > 1e-6:
                audio = (audio / peak) * 0.95

            for d in out_dirs:
                sf.write(os.path.join(d, filename), audio, 24000)

    del model, lightning_tetra, bigvgan
    torch.cuda.empty_cache()


def generate_80mel(model_type="bi", device="cuda"):
    name = "BiDirectional-80mel" if model_type == "bi" else "UniDirectional-80mel"
    folder = "bi_80mel" if model_type == "bi" else "uni_80mel"
    ckpt_file = f"Checkpoints/MambaFlow-{ 'BiDirectional' if model_type == 'bi' else 'UniDirectional' }-Mamba2-80mel.ckpt"
    model_dir = f"MambaFlow-{ 'BiDirectional' if model_type == 'bi' else 'UniDirectional' }-Mamba2-80mel"

    print(f"\n[+] Generating {name} demo samples...")
    purge_model_modules()
    sys.path.insert(0, os.path.join(ROOT_DIR, model_dir))
    from TTSDataModule import TTSMODEL

    # HiFi-GAN Vocoder
    hifi_dir = "/home/monesh/hifigan"
    hifi_ckpt = "/home/monesh/ljspeech/LJSpeech-1.1/generator_v1"
    hifi_cfg = os.path.join(hifi_dir, "config_v1.json")
    if hifi_dir not in sys.path:
        sys.path.insert(0, hifi_dir)
    from models import Generator
    with open(hifi_cfg) as f:
        h = SimpleNamespace(**json.load(f))
    hifigan = Generator(h).to(device).eval()
    ckpt_hifi = torch.load(hifi_ckpt, map_location="cpu")
    hifigan.load_state_dict(ckpt_hifi.get("generator", ckpt_hifi))
    try:
        hifigan.remove_weight_norm()
    except Exception:
        pass

    import importlib.util
    spec = importlib.util.spec_from_file_location("stats_hifigan", "/home/monesh/ljspeech/LJSpeech-1.1/stats_hifigan.py")
    stats_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats_mod)
    mean_t = torch.tensor(stats_mod.MEL_MEAN, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    std_t = torch.tensor(stats_mod.MEL_STD, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    ckpt_path = os.path.join(ROOT_DIR, ckpt_file)
    lightning = TTSMODEL()
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("model.", ""): v for k, v in sd.items()}
    lightning.model.load_state_dict(sd, strict=True)
    model = lightning.model.to(device).eval()

    out_dirs = [os.path.join(ROOT_DIR, f"audio/{folder}"), os.path.join(ROOT_DIR, f"docs/audio/{folder}")]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

    for s in DEMO_SAMPLES:
        text = s["text"]
        filename = f"{s['id']}.wav"
        print(f"  -> {name} synthesizing {filename}: \"{text}\"")

        cleaner_names = ["english_cleaners2"]
        seq = text_to_sequence(text, cleaner_names)[0]
        x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            x_enc = model.encoder(model.emb_dropout(model.txt_emb(x)))
            mu_text_base = model.to_latent(x_enc)
            dur_int, _, _ = model.duration(x_enc.detach(), mask=None, length_scale=1.0)
            durations = torch.clamp(dur_int, min=1)
            mu_expanded = torch.repeat_interleave(mu_text_base[0], durations[0], dim=0).unsqueeze(0)

            n_steps = 30
            dt = 1.0 / float(n_steps)
            x_curr = torch.randn_like(mu_expanded) * 0.667
            for step in range(n_steps):
                t_tensor = torch.full((1,), step * dt, device=device, dtype=torch.float32)
                v = model.decoder.predict_velocity(x_curr, t_tensor, mu_expanded)
                x_curr = x_curr + dt * v

            mel_denorm = (x_curr * std_t + mean_t).transpose(1, 2)
            with torch.inference_mode():
                audio = hifigan(mel_denorm).squeeze().detach().cpu().numpy()

            peak = np.abs(audio).max()
            if peak > 1e-6:
                audio = (audio / peak) * 0.95

            for d in out_dirs:
                sf.write(os.path.join(d, filename), audio, 22050)

    del model, lightning, hifigan
    torch.cuda.empty_cache()


def generate_448d(ckpt_path, device="cuda"):
    print(f"\n[+] Generating 448D Latent demo samples using: {os.path.basename(ckpt_path)}...")
    purge_model_modules()
    latent_dir = os.path.join(ROOT_DIR, "MambaFlow-BiDirectional-Latent VAE")
    sys.path.insert(0, latent_dir)
    from TTSDataModule import TTSMODEL
    from generate_audio import setup_dacvae
    from evaluate_val_health import synthesize_and_record_trajectory

    pl_module = TTSMODEL.load_from_checkpoint(ckpt_path, map_location=device)
    model = pl_module.model.to(device).eval()
    dacvae_decoder, _, _ = setup_dacvae(device=device)

    out_dirs = [os.path.join(ROOT_DIR, "audio/latent_448d"), os.path.join(ROOT_DIR, "docs/audio/latent_448d")]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

    for idx, s in enumerate(DEMO_SAMPLES):
        text = s["text"]
        filename = f"{s['id']}.wav"
        print(f"  -> 448D synthesizing {filename}: \"{text}\"")

        wav_48k, straightness, _ = synthesize_and_record_trajectory(
            model=model,
            dacvae_decoder=dacvae_decoder,
            text=text,
            gt_durations=None,
            use_gt_durations=False,
            n_steps=30,
            temperature=0.667,
            solver="euler",
            cfg_scale=2.0,
            device=device,
            seed=42 + idx,
        )

        peak = np.abs(wav_48k).max()
        if peak > 1e-6:
            wav_48k = (wav_48k / peak) * 0.95

        for d in out_dirs:
            sf.write(os.path.join(d, filename), wav_48k, 48000)

    del model, pl_module, dacvae_decoder
    torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_448d", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Running audio generation on device: {device}")

    # 1. Tetra (100-mel BigVGAN)
    generate_tetra(device=device)

    # 2. BiDirectional 80-mel (HiFi-GAN)
    generate_80mel("bi", device=device)

    # 3. UniDirectional 80-mel (HiFi-GAN)
    generate_80mel("uni", device=device)

    # 4. 448D Latent
    if args.ckpt_448d and os.path.exists(args.ckpt_448d):
        generate_448d(args.ckpt_448d, device=device)
    else:
        # Default to latest in Checkpoints or TTS_checkpoints
        default_ckpt = os.path.join(ROOT_DIR, "Checkpoints/MambaFlow-CA-BiMamba-448D.ckpt")
        generate_448d(default_ckpt, device=device)

    print("\n[+] All audio demonstrations successfully generated and synchronized!")

if __name__ == "__main__":
    main()
