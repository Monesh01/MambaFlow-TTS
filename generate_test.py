import torch
import sys
import argparse
import os
import json
import soundfile as sf
import numpy as np

MODEL_REGISTRY = {
    'sequential_tetra': {
        'name': 'MambaFlow-Sequential-Tetra',
        'dir': 'MambaFlow-Sequential-Tetra',
        'ckpt': 'Checkpoints/MambaFlow-Sequential-Tetra.ckpt',
        'mel_bins': 100,
        'sr': 24000,
        'vocoder': 'bigvgan',
    },
    'staircase_xtencoder': {
        'name': 'MambaFlow-Staircase-XTEncoder',
        'dir': 'MambaFlow-Staircase-XTEncoder',
        'ckpt': 'Checkpoints/MambaFlow-Staircase-XTEncoder.ckpt',
        'mel_bins': 100,
        'sr': 24000,
        'vocoder': 'bigvgan',
    },
    'staircase_mamba2': {
        'name': 'MambaFlow-Staircase-Mamba2',
        'dir': 'MambaFlow-Staircase-Mamba2',
        'ckpt': 'Checkpoints/MambaFlow-Staircase-Mamba2.ckpt',
        'mel_bins': 100,
        'sr': 24000,
        'vocoder': 'bigvgan',
    },
    'twostage_mamba2': {
        'name': 'MambaFlow-TwoStage-Mamba2',
        'dir': 'MambaFlow-TwoStage-Mamba2',
        'ckpt': 'Checkpoints/MambaFlow-TwoStage-Mamba2.ckpt',
        'mel_bins': 100,
        'sr': 24000,
        'vocoder': 'bigvgan',
    },
    'unet_onebottleneck': {
        'name': 'MambaFlow-UNet-OneBottleneck',
        'dir': 'MambaFlow-UNet-OneBottleneck',
        'ckpt': 'Checkpoints/MambaFlow-UNet-OneBottleneck.ckpt',
        'mel_bins': 100,
        'sr': 24000,
        'vocoder': 'bigvgan',
    },
    'bimamba2_80mel': {
        'name': 'MambaFlow-BiDirectional-Mamba2-80mel',
        'dir': 'MambaFlow-BiDirectional-Mamba2-80mel',
        'ckpt': 'Checkpoints/MambaFlow-BiDirectional-Mamba2-80mel.ckpt',
        'mel_bins': 80,
        'sr': 22050,
        'vocoder': 'hifigan',
    },
    'unimamba2_80mel': {
        'name': 'MambaFlow-UniDirectional-Mamba2-80mel',
        'dir': 'MambaFlow-UniDirectional-Mamba2-80mel',
        'ckpt': 'Checkpoints/MambaFlow-UniDirectional-Mamba2-80mel.ckpt',
        'mel_bins': 80,
        'sr': 22050,
        'vocoder': 'hifigan',
    },
}

def load_isolated_model(model_key, ckpt_path=None, device="cpu"):
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Choices: {list(MODEL_REGISTRY.keys())}")
    
    cfg = MODEL_REGISTRY[model_key]
    model_dir = cfg['dir']
    ckpt_path = ckpt_path or cfg['ckpt']

    # Purge cached modules from sys.modules to prevent cross-model namespace collisions
    for mod_name in list(sys.modules.keys()):
        if any(mod_name.startswith(p) for p in ['mambaflow', 'TTS', 'trans_encoder', 'duration_predictor']):
            del sys.modules[mod_name]

    abs_dir = os.path.abspath(model_dir)
    sys.path.insert(0, abs_dir)
    sys.path.insert(0, os.path.abspath("."))

    from TTSDataModule import TTSMODEL
    from TTSDatasetModule import denormalize_mel

    print(f"Loading {cfg['name']} from {model_dir}...")
    lightning_model = TTSMODEL()

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
        try:
            lightning_model.model.load_state_dict(state_dict, strict=True)
            print(">>> Checkpoint loaded with 100% strict parameter match!")
        except Exception as e:
            print(f">>> Strict load failed ({e}), falling back to non-strict:")
            missing, unexpected = lightning_model.model.load_state_dict(state_dict, strict=False)
            print(f"    Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model = lightning_model.model.to(device).eval()
    sys.path.remove(abs_dir)
    return model, denormalize_mel, cfg


def generate_audio(model_key, ckpt_path=None, text=None, out_wav=None, n_timesteps=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, denormalize_fn, cfg = load_isolated_model(model_key, ckpt_path, device=device)

    test_text = text or "Welcome to the MambaFlow Text to Speech demonstration. Our models eliminate multiscale aliasing."
    print(f"Synthesizing text: '{test_text}'")

    from preprocessing.text import text_to_sequence
    seq, _ = text_to_sequence(test_text, ["english_cleaners2"])
    x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

    print(f"Generating audio with {cfg['name']} (n_timesteps={n_timesteps}, solver=euler)...")
    with torch.no_grad():
        mel_pred_norm, _ = model(
            x=x,
            n_timesteps=n_timesteps,
            temperature=0.667,
            length_scale=1.0,
            solver="euler",
        )

    print(f"Generated Mel-Spectrogram shape: {mel_pred_norm.shape} ({cfg['mel_bins']} mel bins)")

    audio = None
    sr = cfg['sr']

    # 1. BigVGAN Vocoding (for 100-mel models)
    if cfg['vocoder'] == 'bigvgan':
        voc_cfg = "/home/monesh/bigvgan_model/config_14M.json"
        voc_pt = "/home/monesh/bigvgan_model/bigvgan_generator_14M.pt"
        if os.path.exists(voc_cfg) and os.path.exists(voc_pt):
            from bigvgan import BigVGAN, AttrDict
            with open(voc_cfg) as f:
                h = AttrDict(json.load(f))
            bigvgan = BigVGAN(h, use_cuda_kernel=False).to(device).eval()
            ckpt_voc = torch.load(voc_pt, map_location=device)
            bigvgan.load_state_dict(ckpt_voc.get("generator", ckpt_voc))
            bigvgan.remove_weight_norm()

            mel_raw = denormalize_fn(mel_pred_norm).transpose(1, 2)
            with torch.inference_mode():
                audio = bigvgan(mel_raw)[0, 0].detach().cpu().numpy()
        else:
            print("Warning: BigVGAN weights not found; audio not rendered.")

    # 2. HiFi-GAN Vocoding (for 80-mel models)
    elif cfg['vocoder'] == 'hifigan':
        hifi_dir = "/home/monesh/hifigan"
        hifi_ckpt = "/home/monesh/ljspeech/LJSpeech-1.1/generator_v1"
        hifi_cfg = os.path.join(hifi_dir, "config_v1.json")
        if os.path.exists(hifi_ckpt) and os.path.exists(hifi_cfg):
            if hifi_dir not in sys.path:
                sys.path.insert(0, hifi_dir)
            from models import Generator
            from types import SimpleNamespace
            with open(hifi_cfg) as f:
                h = SimpleNamespace(**json.load(f))
            hifigan = Generator(h).to(device).eval()
            ckpt_hifi = torch.load(hifi_ckpt, map_location=device)
            hifigan.load_state_dict(ckpt_hifi.get("generator", ckpt_hifi))
            try:
                hifigan.remove_weight_norm()
            except Exception:
                pass

            # Load 80-mel stats
            import importlib.util
            spec = importlib.util.spec_from_file_location("stats_hifigan", "/home/monesh/ljspeech/LJSpeech-1.1/stats_hifigan.py")
            stats_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(stats_mod)
            mean_t = torch.tensor(stats_mod.MEL_MEAN, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            std_t = torch.tensor(stats_mod.MEL_STD, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            mel_denorm = (mel_pred_norm * std_t + mean_t).transpose(1, 2)
            with torch.inference_mode():
                audio = hifigan(mel_denorm).squeeze().detach().cpu().numpy()
        else:
            print("Warning: HiFi-GAN weights not found; audio not rendered.")

    if audio is not None:
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = (audio / peak) * 0.95
        out_path = out_wav or f"test_{model_key}_sample.wav"
        sf.write(out_path, audio, sr)
        print(f"Audio successfully saved to: {out_path} (SR: {sr}Hz, Duration: {len(audio)/sr:.2f}s, Peak: {peak:.4f})")
    else:
        print("Audio vocoding skipped.")

    return mel_pred_norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MambaFlow-TTS Multi-Model Audio Generator")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_REGISTRY.keys(), help="Model identifier")
    parser.add_argument("--ckpt", type=str, default=None, help="Custom checkpoint path")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--out", type=str, default=None, help="Output WAV path")
    parser.add_argument("--steps", type=int, default=30, help="Flow matching ODE integration steps")
    
    args = parser.parse_args()
    generate_audio(args.model, args.ckpt, text=args.text, out_wav=args.out, n_timesteps=args.steps)
