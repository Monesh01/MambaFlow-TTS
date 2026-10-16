# MambaFlow-TTS: Eliminating Multiscale Aliasing via Full-Resolution State Space Flow Matching

[![Live Audio Demonstration](https://img.shields.io/badge/GitHub%20Pages-Live%20Showcase-indigo?style=for-the-badge&logo=github)](https://monesh01.github.io/MambaFlow-TTS/)
[![Report](https://img.shields.io/badge/Technical%20Report-BENCHMARK__REPORT.md-blue?style=for-the-badge&logo=markdown)](BENCHMARK_REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg?style=for-the-badge)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

**Author**: Monesh S ([@Monesh01](https://github.com/Monesh01))  
**Institution**: Saveetha University, Chennai  
**Interactive Audio Showcase**: [https://monesh01.github.io/MambaFlow-TTS/](https://monesh01.github.io/MambaFlow-TTS/)  

---

## Overview

**MambaFlow-TTS** is an experimental research Text-to-Speech (TTS) architecture uniting **Continuous Optimal Transport Conditional Flow Matching (OT-CFM)** with **Bidirectional Mamba-2 State Space Models (SSMs)**.

### Problem: Destructive Multiscale Aliasing
Diffusion and flow-matching speech generators commonly employ multiscale temporal downsamplers (e.g., U-Net pyramids pooling by 2x, 4x, 8x) to avoid the quadratic O(T^2) memory footprint of self-attention. However, multiscale temporal downsampling followed by transposed convolutions creates periodic spectral replicas (Nyquist sidebands). When passed to neural vocoders, these cause phase flanging and comb-filtering artifacts in high-frequency spectral bands (6-12 kHz).

### Solution: Native 1x Resolution State Space Flow Matching
By replacing self-attention with bidirectional Mamba-2 SSMs, MambaFlow operates at **strict 1x temporal resolution** throughout all decoder layers with linear O(T) memory scaling:
- **Zero temporal downsampling**: No strided pooling and no transposed convolutions.
- **Direct Target Vector Fields**: Eliminates additive residual staircase leakage.
- **Multi-Domain Acoustic Exploration**: Evaluated across 48 kHz continuous latents (DACVAE), 24 kHz 100-mel spectrograms (BigVGAN), and 22.05 kHz 80-mel spectrograms (HiFi-GAN).

---

## Research Architectures

The repository includes five experimental model configurations:

1. **MambaFlow-CA-BiMamba-448D** (42.2M params): 128D continuous latent space with Meta DACVAE at 48 kHz, combining cross-attention and bidirectional Mamba-2.
2. **MambaFlow-CA-BiMamba-256D** (18.1M params): Compact 128D continuous latent model for resource-constrained evaluation.
3. **MambaFlow-Sequential-Tetra** (14.1M params): Native 1x resolution 100-mel spectrogram model paired with BigVGAN 14M (24 kHz), bypassing multiscale pooling.
4. **MambaFlow-BiDirectional-80mel** (15.1M params): Dual-channel non-causal SSM with full lookahead context paired with HiFi-GAN (22.05 kHz).
5. **MambaFlow-UniDirectional-80mel** (11.1M params): Forward-only causal SSM designed for low-latency streaming speech synthesis (22.05 kHz).

---

## System Architecture Pipeline

The synthesis pipeline operates in three sequential stages:

```
[Text Input (IPA)] 
       │
       ▼
1. Linguistic Frontend: 6-Layer BiMamba-2 Text Encoder (192D) + Duration Predictor (MAS)
       │
       ▼
2. Flow Matching Engine: 30-step Euler ODE Solver with Full 1x Resolution BiMamba-2 Decoders
       │
       ▼
3. Neural Vocoder Backend:
   ├─ 48 kHz: Meta DACVAE (128D Continuous Latent)
   ├─ 24 kHz: BigVGAN 14M (100-Mel Native 1x)
   └─ 22.05 kHz: HiFi-GAN V1 (80-Mel Causal / Non-Causal)
```

---

## Repository Structure

```
TTSModel/
├── Checkpoints/                               # Model Checkpoints
│   ├── MambaFlow-CA-BiMamba-448D.ckpt         # 128D DACVAE (42.20M params)
│   ├── MambaFlow-CA-BiMamba-256D.ckpt         # Compact 128D DACVAE (18.07M params)
│   ├── MambaFlow-Sequential-Tetra.ckpt        # Native 1x 100-Mel Model (14.06M params)
│   ├── MambaFlow-BiDirectional-Mamba2-80mel.ckpt # 80-Mel HiFi-GAN Model (15.09M params)
│   └── MambaFlow-UniDirectional-Mamba2-80mel.ckpt # Causal HiFi-GAN Model (11.14M params)
│
├── docs/                                      # GitHub Pages Audio Demonstration Site
│   ├── index.html                             # Showcase Web Application
│   └── audio/                                 # Evaluation Audio Samples
│       ├── gt/                                # LJSpeech Reference Ground Truth
│       ├── latent_448d/                       # MambaFlow-CA-BiMamba-448D (48 kHz DACVAE)
│       ├── latent_256d/                       # MambaFlow-CA-BiMamba-256D (48 kHz DACVAE)
│       ├── seq_tetra_100mel/                  # MambaFlow-Sequential-Tetra (24 kHz BigVGAN)
│       ├── bi_80mel/                          # MambaFlow-BiDirectional-80mel (22 kHz HiFi-GAN)
│       └── uni_80mel/                         # MambaFlow-UniDirectional-80mel (22 kHz HiFi-GAN)
│
├── MambaFlow-BiDirectional-Latent VAE/        # 48 kHz DACVAE Architecture Code
├── MambaFlow-Sequential-Tetra/                # Native 1x Resolution BigVGAN Architecture Code
├── MambaFlow-BiDirectional-Mamba2-80mel/      # Non-Causal 80-Mel Architecture Code
├── MambaFlow-UniDirectional-Mamba2-80mel/      # Causal 80-Mel Architecture Code
│
├── run_unified_50_eval.py                     # 50-Sample Evaluation Suite
├── BENCHMARK_REPORT.md                        # Empirical Technical Report
└── README.md                                  # Research Overview & Quickstart Guide
```

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/Monesh01/MambaFlow-TTS.git
cd MambaFlow-TTS
pip install torch torchaudio mamba-ssm causal-conv1d soundfile librosa dac-vae faster-whisper
```

### 2. Audio Synthesis
```python
import torch
import soundfile as sf
from MambaFlow_BiDirectional_Latent_VAE.TTSDataModule import TTSMODEL
from MambaFlow_BiDirectional_Latent_VAE.generate_audio import setup_dacvae, decode_latent_to_audio

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load 448D Latent Checkpoint (42.20M params)
ckpt_path = "Checkpoints/MambaFlow-CA-BiMamba-448D.ckpt"
model = TTSMODEL.load_from_checkpoint(ckpt_path, map_location=device).model.to(device).eval()
dacvae_decoder, _, _ = setup_dacvae(device="cpu")

# 2. Synthesize 48 kHz Speech
text = "MambaFlow-TTS eliminates multiscale aliasing through full-resolution state space flow matching."
with torch.inference_mode():
    audio_48k = model.synthesize(text, n_timesteps=30, temperature=0.667, cfg_scale=2.0, dacvae=dacvae_decoder)

sf.write("output_mambaflow_48k.wav", audio_48k, 48000)
print("Speech synthesized at 48,000 Hz.")
```

---

## Citation

```bibtex
@misc{monesh2026mambaflow,
  author = {Monesh, S.},
  title = {MambaFlow-TTS: Eliminating Multiscale Aliasing and Comb-Filtering via Full-Resolution Bidirectional State Space Flow Matching},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Monesh01/MambaFlow-TTS}},
  note = {Interactive Demo: \url{https://monesh01.github.io/MambaFlow-TTS/}}
}
```
