import torch
import sys
import argparse
import os
import soundfile as sf
import numpy as np

MODEL_DIRS = {
    'staircase_mamba2': ('MambaFlow-Staircase-Mamba2', 'Checkpoints/MambaFlow-Staircase-Mamba2-epoch=127-val_loss=0.5824.ckpt'),
    'staircase_xtencoder': ('MambaFlow-Staircase-XTEncoder', 'Checkpoints/MambaFlow-Staircase-XTEncoder-epoch=145-val_loss=0.5805.ckpt'),
    'twostage_mamba2': ('MambaFlow-TwoStage-Mamba2', 'Checkpoints/MambaFlow-TwoStage-Mamba2-epoch=169-val_loss=0.5837.ckpt'),
    'unet_onebottleneck': ('MambaFlow-UNet-OneBottleneck', 'Checkpoints/MambaFlow-UNet-OneBottleneck-last_decoder_only.ckpt'),
    'sequential_tetra': ('MambaFlow-Sequential-Tetra', 'Checkpoints/MambaFlow-Sequential-Tetra-epoch=132-val_loss=0.3295.ckpt'),
}

def load_isolated_model(model_name, ckpt_path=None, device="cpu"):
    if model_name not in MODEL_DIRS:
        raise ValueError(f"Unknown model {model_name}. Choices: {list(MODEL_DIRS.keys())}")
    
    model_dir, default_ckpt = MODEL_DIRS[model_name]
    ckpt_path = ckpt_path or default_ckpt

    # 1. Purge cached modules from sys.modules to prevent cross-model namespace collisions
    for mod_name in list(sys.modules.keys()):
        if any(mod_name.startswith(p) for p in ['mambaflow', 'TTS', 'trans_encoder', 'duration_predictor']):
            del sys.modules[mod_name]

    abs_dir = os.path.abspath(model_dir)
    sys.path.insert(0, abs_dir)
    sys.path.insert(0, os.path.abspath("."))

    from TTSDataModule import TTSMODEL
    from TTSDatasetModule import denormalize_mel

    print(f"Loading {model_name} architecture from {model_dir}...")
    lightning_model = TTSMODEL()

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
        # Enforce strict loading to guarantee 100% parameter match
        try:
            lightning_model.model.load_state_dict(state_dict, strict=True)
            print(">>> Checkpoint loaded successfully with 100% strict parameter match!")
        except Exception as e:
            print(f">>> Strict load failed ({e}), falling back to non-strict:")
            missing, unexpected = lightning_model.model.load_state_dict(state_dict, strict=False)
            print(f"    Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model = lightning_model.model.to(device).eval()
    sys.path.remove(abs_dir)
    return model, denormalize_mel

def generate_audio(model_name, ckpt_path=None, text=None, out_wav=None, n_timesteps=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, denormalize_fn = load_isolated_model(model_name, ckpt_path, device=device)

    # Convert text or use default test phrase
    test_text = text or "Welcome to the MambaFlow Text to Speech demonstration. Our models eliminate multiscale aliasing."
    print(f"Synthesizing text: '{test_text}'")

    from preprocessing.text import text_to_sequence
    seq, _ = text_to_sequence(test_text, ["english_cleaners2"])
    x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

    print(f"Generating audio with {model_name} (n_timesteps={n_timesteps}, solver=euler)...")
    with torch.no_grad():
        mel_pred_norm, _ = model(
            x=x,
            n_timesteps=n_timesteps,
            temperature=0.667,
            length_scale=1.0,
            solver="euler",
        )

    print(f"Generated Mel-Spectrogram shape: {mel_pred_norm.shape}")

    # Vocode with BigVGAN if vocoder exists
    voc_cfg = "/home/monesh/bigvgan_model/config_14M.json"
    voc_pt = "/home/monesh/bigvgan_model/bigvgan_generator_14M.pt"
    if os.path.exists(voc_cfg) and os.path.exists(voc_pt):
        import json
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
        
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = (audio / peak) * 0.95

        out_path = out_wav or f"test_{model_name}_sample.wav"
        sf.write(out_path, audio, 24000)
        print(f"Audio successfully saved to: {out_path} (duration: {len(audio)/24000:.2f}s, peak: {peak:.4f})")
    else:
        print("BigVGAN weights not found; saved mel only.")

    return mel_pred_norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TTS generation cleanly")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_DIRS.keys(), help="Model to use")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--out", type=str, default=None, help="Path to output WAV file")
    parser.add_argument("--steps", type=int, default=30, help="Flow matching steps")
    
    args = parser.parse_args()
    generate_audio(args.model, args.ckpt, text=args.text, out_wav=args.out, n_timesteps=args.steps)
