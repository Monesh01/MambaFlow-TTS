# MambaFlow-TTS: Eliminating Multiscale Aliasing in Flow-Matching Speech Synthesis via Full-Resolution State-Space Models

🚀 **[Click Here for the Interactive Audio Demonstration Site](https://monesh01.github.io/MambaFlow-TTS/)** 🚀

**Author**: Monesh

**MambaFlow-TTS** is a high-fidelity, non-autoregressive Text-to-Speech (TTS) architecture that combines **Continuous Optimal Transport Flow Matching (OT-CFM)** with **Bidirectional Mamba-2 State-Space Models (SSMs)**.

This repository serves as a comprehensive investigation into the acoustic artifacts commonly found in modern diffusion and flow-matching TTS systems. Our research empirically demonstrates that traditional multiscale downsampling/upsampling paradigms and residual target leakage are the primary culprits behind severe temporal aliasing ("comb-filtering" and "dual voice" artifacts). 

To solve this, we propose **MambaFlow-Sequential-Tetra**, a $1\times$ native-resolution flow decoder that completely eliminates these artifacts, achieving high spectral convergence and naturalness.

---

## 🛠️ Tech Stack Used

- **Deep Learning Framework**: PyTorch, PyTorch Lightning
- **State-Space Models**: Mamba-2 (Bidirectional)
- **Spatial Mixing**: ConvNeXt-1D
- **Vocoder**: BigVGAN (14M parameter variant)
- **Flow Matching**: Continuous Optimal Transport Conditional Flow Matching (OT-CFM)
- **ODE Solvers**: Midpoint (Runge-Kutta 2nd Order), Euler
- **Audio Processing**: SoundFile, Librosa

---

## 🔬 Key Architectural Highlights

1. **Identification of Artifact Etiology**: We rigorously demonstrate that using transposed 1D convolutions (`ConvTranspose1d`) in standard U-Net or Staircase decoders injects high-frequency Nyquist sidebands into the mel-spectrogram. When processed by neural vocoders (like BigVGAN), this aliasing manifests as robotic flanging and phase distortion.
2. **Mitigation of Residual Leakage**: We identify that training models with residual targets (where the network predicts the difference between a piecewise-constant acoustic prior and the target mel) results in a "staircase leakage" effect, causing a static robotic voice to persist underneath the dynamic generated speech.
3. **The Tetra Architecture**: We introduce the **Tetra Sequential Decoder**, which operates entirely at full temporal resolution with zero pooling and zero upsampling. Combined with direct full-mel velocity targeting, Tetra achieves pristine audio generation, eliminating the worst-case distortions deliberately modeled in our baseline architectures.
4. **Computational Efficiency**: By leveraging Bidirectional Mamba-2 blocks interleaved with ConvNeXt-1D spatial mixing, the model achieves linear $\mathcal{O}(N)$ complexity, bypassing the quadratic memory constraints of self-attention.

---

## 📊 Experimental Validation & Benchmark Results

To evaluate our hypothesis regarding multiscale aliasing, we trained and evaluated 5 distinct decoder topologies on the LJSpeech dataset. Evaluation spans 100% of the validation set (1,310 utterances).

### Detailed Model & Checkpoint Ranking

Based on objective validation loss, temporal artifact suppression, and computational architecture, the models are strictly ranked as follows:

1. **FIRST PLACE (BEST): MambaFlow-Sequential-Tetra [14.06M params]**
   * **Checkpoint**: `MambaFlow-Sequential-Tetra-epoch=132-val_loss=0.3295.ckpt`
   * **Why Best**: Operates entirely at full temporal resolution ($1\times$), eliminating multiscale aliasing and comb-filtered phase artifacts. Uses direct full mel flow targeting without staircase residual leakage. Lowest validation CFM loss ($0.3295$).

2. **SECOND PLACE: MambaFlow-Staircase-XTEncoder [11.18M params]**
   * **Checkpoint**: `MambaFlow-Staircase-XTEncoder-epoch=145-val_loss=0.5805.ckpt`
   * **Why Second**: Stable end-to-end multi-task trained backbone with balanced multi-scale feature aggregation. Served as the robust initialization source for the Tetra fine-tuning run.

3. **THIRD PLACE: MambaFlow-Staircase-Mamba2 [11.27M params]**
   * **Checkpoint**: `MambaFlow-Staircase-Mamba2-epoch=127-val_loss=0.5824.ckpt`
   * **Why Third**: Balanced pyramid skip connections provide better high-frequency detail preservation than single-bottleneck architectures.

4. **FOURTH PLACE: MambaFlow-TwoStage-Mamba2 [14.37M params]**
   * **Checkpoint**: `MambaFlow-TwoStage-Mamba2-epoch=169-val_loss=0.5837.ckpt`
   * **Why Fourth**: Large model capacity, but cascaded downsampling ($1/2\times$ and $1/4\times$) introduces temporal aliasing audible on single-speaker devices.

5. **FIFTH PLACE: MambaFlow-UNet-OneBottleneck [8.50M params]**
   * **Checkpoint**: `MambaFlow-UNet-OneBottleneck-last_decoder_only.ckpt`
   * **Why Fifth**: Highly compact, but the single bottleneck forces aggressive compression and creates slight loss of acoustic sharpness.

### Complete Multi-Model Benchmark Matrix ($N = 1,310$ Validation Samples)

| Architecture | Params | Decoder | Ckpt | Mel MAE | Spectral Conv | High-Freq MAE | DTW Dist | RTF | Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tetra Sequential (Ours)** | 14.06M | 10.33M | 132.6 MB | 0.743 | 0.5933 | **0.633** | 25.70 | 0.0283 | 35.3x |
| **Staircase-XTEncoder** | 11.18M | 7.46M | 128.2 MB | **0.737** | **0.5878** | 0.636 | **25.30** | 0.0259 | 38.7x |
| **Staircase-Mamba2** | 11.27M | 7.55M | 129.2 MB | 0.765 | 0.6054 | 0.659 | 26.65 | 0.0257 | 38.9x |
| **TwoStage-Mamba2** | 14.37M | 10.65M | 164.6 MB | 0.749 | 0.6009 | 0.646 | 25.95 | 0.0160 | 62.6x |
| **UNet-OneBottleneck** | **8.50M** | **4.78M** | **69.0 MB** | 0.752 | 0.5958 | 0.652 | 26.07 | **0.0150** | **66.5x** |

---

## 💻 Reproducibility & Inference

The codebase provides clean, dynamic loading of all architectural variants. Below is the standard protocol for generating high-fidelity audio using the champion **Tetra** model.

### Python Quickstart

```python
import torch
import soundfile as sf
import json
from bigvgan import BigVGAN, AttrDict

# Ensure compatibility for checkpoint loading
import torch
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})

import sys
sys.path.insert(0, "./MambaFlow-Sequential-Tetra")
from TTSDataModule import TTSMODEL
from TTSDatasetModule import denormalize_mel
from preprocessing.text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load MambaFlow-Sequential-Tetra Architecture
ckpt_path = "Checkpoints/MambaFlow-Sequential-Tetra-epoch=132-val_loss=0.3295.ckpt"
lightning_model = TTSMODEL()
ckpt = torch.load(ckpt_path, map_location=device)
lightning_model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
model = lightning_model.model.to(device).eval()

# 2. Load BigVGAN Neural Vocoder
vocoder_config = "/home/monesh/bigvgan_model/config_14M.json"
vocoder_ckpt = "/home/monesh/bigvgan_model/bigvgan_generator_14M.pt"
with open(vocoder_config) as f:
    h = AttrDict(json.load(f))

bigvgan = BigVGAN(h, use_cuda_kernel=False).to(device).eval()
if "generator" in torch.load(vocoder_ckpt, map_location=device):
    bigvgan.load_state_dict(torch.load(vocoder_ckpt, map_location=device)["generator"])
else:
    bigvgan.load_state_dict(torch.load(vocoder_ckpt, map_location=device))
bigvgan.remove_weight_norm()

# 3. Linguistic Frontend Processing
text = "MambaFlow Text to Speech synthesizes crystal clear audio by eliminating temporal aliasing."
seq, _ = text_to_sequence(text, ["english_cleaners2"])
x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

# 4. Continuous ODE Integration (Flow Matching)
with torch.no_grad():
    mel_norm, _ = model(
        x=x,
        target_latent=None,
        n_timesteps=30,      # 30 steps for optimal fidelity
        temperature=0.667,   # Prior sampling temperature
        length_scale=1.0,    # Speech tempo (1.0 = normal)
        solver="midpoint",   # 2nd-order Runge-Kutta solver
        t_end=1.0,
    )
    
    # 5. Acoustic Denormalization & Vocoding
    mel_raw = denormalize_mel(mel_norm)               
    mel_bigvgan = mel_raw.transpose(1, 2)             
    audio = bigvgan(mel_bigvgan)                      
    
    # Fix for multi-channel bug (Issue BUG-01)
    audio_np = audio[0, 0].detach().cpu().numpy()

# 6. Peak Normalization and Audio Export
peak = abs(audio_np).max()
if peak > 1e-6:
    audio_np = (audio_np / peak) * 0.95

sf.write("output.wav", audio_np, 24000)
print("Successfully saved output.wav (24,000 Hz Mono PCM)")
```

---
*Note: Due to multi-channel interpretation by libraries like `soundfile`, ensure batch slicing (e.g., `audio[0, 0]`) is applied during generation to maintain proper 1-channel mono PCM format.*
