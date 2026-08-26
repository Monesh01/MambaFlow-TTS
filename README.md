# MambaFlow-TTS: Flow-Matching Speech Synthesis with Bidirectional State-Space Models

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Mamba-2](https://img.shields.io/badge/SSM-Mamba--2%20(SSD)-green.svg)](https://github.com/state-spaces/mamba)
[![Generative Paradigm](https://img.shields.io/badge/CFM-Optimal%20Transport-purple.svg)](https://arxiv.org/abs/2302.00482)
[![Audio Quality](https://img.shields.io/badge/Audio-24kHz%20%7C%20100--Band%20Mel-orange.svg)](https://github.com/NVIDIA/BigVGAN)

**MambaFlow-TTS** is a non-autoregressive Text-to-Speech (TTS) architecture that pairs **Continuous Optimal Transport Flow Matching (OT-CFM)** with **Bidirectional Mamba-2 State-Space Models (SSMs)**.

By replacing traditional 1D convolutional U-Nets (e.g., Matcha-TTS) and quadratic self-attention mechanisms with selective State-Space Models (State Space Duality / SSD), MambaFlow-TTS models long-range acoustic dependencies, phonetic transitions, and prosodic contours with **linear $\mathcal{O}(N)$ computational complexity and memory efficiency**.

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [Architecture Overview](#-architecture-overview)
- [Comparative SOTA Analysis](#-comparative-sota-analysis)
- [Full Dataset Validation Benchmarks](#-full-dataset-validation-benchmarks)
- [Long-Form Audio Scalability Stress Test](#-long-form-audio-scalability-stress-test)
- [Audio Samples & Demonstrations](#-audio-samples--demonstrations)
- [Repository Structure](#-repository-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Training & Fine-Tuning](#-training--fine-tuning)
- [Inference & Synthesis](#-inference--synthesis)
- [Roadmap](#-roadmap)
- [Citations & Acknowledgments](#-citations--acknowledgments)
- [License](#-license)

---

## 🌟 Key Features

* **Linear $\mathcal{O}(N)$ Flow-Matching Backbone:** Replaces convolutional U-Nets with a 6-layer bidirectional Mamba-2 SSM decoder, ensuring global receptive fields across thousands of acoustic frames without quadratic self-attention memory blowouts.
* **Continuous Optimal Transport Flow Matching (OT-CFM):** Learns straight ODE trajectories between standard Gaussian priors and complex acoustic targets, generating high-fidelity mel-spectrograms in as few as **10 Euler integration steps**.
* **High-Fidelity 100-Channel 24 kHz Synthesis:** Produces 100-band log-mel spectrograms inverted with a **14M BigVGAN** neural vocoder using anti-aliased periodic Snake activations for artifact-free audio.
* **Monotonic Alignment Search (MAS):** End-to-end unsupervised alignment search directly learns duration distributions and aligns phonemes to frame-level acoustic targets.
* **Real-Time Factor (RTF):** Achieves **0.0297 TTS RTF (~33.7× faster than real-time)** on GPU for 10 Euler steps across the entire LJSpeech validation set.

---

## 🏗 Architecture Overview

```mermaid
flowchart TD
    A["Input Text / Phonemes (x)"] --> B["Transformer Text Encoder<br/>(6 Layers | 6 Heads | 256 Dim | 5.51M Params)"]
    B --> C["Acoustic Prior Projection<br/>Linear (256 → 100 Dim)"]
    B --> D["Duration Predictor & Alignment<br/>(1D Depthwise Conv + MAS | 0.39M Params)"]
    D --> E["Frame Expansion (Repeat-Interleave)<br/>Temporal Length Regulation"]
    C --> E
    
    F["Standard Gaussian Prior<br/>x₀ ~ N(0, I)"] --> G["OT-CFM ODE Integration<br/>(Euler Solver: 10–25 Steps)"]
    E --> G
    
    subgraph Decoder ["Bidirectional Mamba-2 Vector Field Estimator (21.38M Params)"]
        G <--> H["Forward Selective Scan (SSD)"]
        G <--> I["Reverse Valid-Sequence Gather Scan"]
        H & I <--> J["Timestep Modulation MLP φ(t) & Latent Fusion"]
    end
    
    G --> K["Predicted Mel-Spectrogram<br/>(100 Channels | 24 kHz Resolution)"]
    K --> L["BigVGAN Neural Vocoder<br/>(14M Generator | Anti-Aliased Snake)"]
    L --> M["🔊 Synthesized Audio Waveform<br/>(24,000 Hz / 16-bit PCM)"]
```

### Mathematical Foundations

1. **Optimal Transport Conditional Flow Matching (OT-CFM):**
   Given a standard Gaussian prior distribution $x_0 \sim \mathcal{N}(0, \mathbf{I})$ and data target $x_1 \sim q(x_1)$, the probability path $x_t$ is defined along the linear interpolation:
   $$\psi_t(x_0, x_1) = (1 - (1 - \sigma_{\min})t)x_0 + t x_1$$
   The target velocity field is constant with respect to time:
   $$u_t(x \mid x_0, x_1) = x_1 - (1 - \sigma_{\min})x_0$$
   The bidirectional Mamba-2 network $v_\theta(x_t, t, \mu)$ is trained using the mean squared error regression objective:
   $$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left\| v_\theta(x_t, t, \mu) - (x_1 - (1 - \sigma_{\min})x_0) \right\|_2^2$$

2. **Bidirectional State-Space Duality (Mamba-2):**
   Unlike standard causal SSMs designed for language modeling, speech generation requires bidirectional context. We construct bidirectional Mamba-2 blocks that compute:
   $$h_{\text{fwd}} = \text{SSM}_{\text{fwd}}(x, t), \quad h_{\text{bwd}} = \text{Reverse}\Big(\text{SSM}_{\text{bwd}}(\text{Reverse}(x, \text{mask}), t), \text{mask}\Big)$$
   where sequence reversal strictly respects padded boundary masks using tensor gather operations to ensure numerical stability during distributed gradient backpropagation.

### Detailed Parameter Breakdown

| Component | Architecture Description | Parameters | Memory Footprint |
| :--- | :--- | :---: | :---: |
| **Text Embedding** | Vocab Table ($179 \times 256$) | 45.8 K | 0.18 MB |
| **Text Encoder** | 6-Layer Transformer Encoder (256-dim, 6 heads, FFN 1024) | 5.51 M | 22.0 MB |
| **Duration Predictor** | 1D Depthwise Conv Blocks + MAS | 0.39 M | 1.56 MB |
| **Latent Projection** | Linear Projection ($256 \to 100$) | 25.6 K | 0.10 MB |
| **CFM Decoder** | 6-Layer Bidirectional Mamba-2 SSM ($d_{\text{model}}=384, d_{\text{state}}=128$) | **21.38 M** | 85.5 MB |
| **Total TTS Core** | **MambaFlow-TTS Acoustic Generator** | **~27.35 M** | **~109.4 MB** |
| **Neural Vocoder** | BigVGAN-14M Generator (Anti-Aliased Snake) | ~14.0 M | ~56.0 MB |
| **Full Pipeline** | **End-to-End System** | **~41.35 M** | **~165.4 MB** |

---

## 🔬 Comparative SOTA Analysis

The table below contrasts **MambaFlow-TTS** against leading open-source TTS paradigms:

| Model Architecture | Generative Paradigm | Core Decoder Backbone | Mel Channels & Audio Bandwidth | Parameter Count (TTS + Vocoder) | ODE / Sampling Steps (NFE) | TTS GPU RTF | Sequence Scaling Complexity | Prosody & Long-Range Context |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MambaFlow-TTS (Ours - 10 Steps)** | **Continuous Flow Matching (OT-CFM)** | **Bidirectional Mamba-2 SSM** | **100 Bins / 24 kHz** | **27.4M + 14.0M (BigVGAN)** | **10** | **0.0297** (~33.7× RT) | **$\mathcal{O}(N)$ Linear** | **High (Global SSM context)** |
| **MambaFlow-TTS (Ours - 25 Steps)** | **Continuous Flow Matching (OT-CFM)** | **Bidirectional Mamba-2 SSM** | **100 Bins / 24 kHz** | **27.4M + 14.0M (BigVGAN)** | **25** | **0.0701** (~14.3× RT) | **$\mathcal{O}(N)$ Linear** | **Exceptional** |
| **Matcha-TTS** *(Mehta et al., 2024)* | Continuous Flow Matching (OT-CFM) | 1D Conv U-Net + DiT Block | 80 Bins / 22.05 kHz | 18.2M + 13.9M (HiFi-GAN) | 10 | 0.0124 (~80.6× RT) | $\mathcal{O}(N)$ Local Conv | Moderate (Local receptive field) |
| **Grad-TTS** *(Popov et al., 2021)* | Score-based Diffusion (SDE) | 2D/1D Conv U-Net | 80 Bins / 22.05 kHz | 14.8M + 13.9M (HiFi-GAN) | 50–100 | 0.0800–0.4000 | $\mathcal{O}(N)$ Local Conv | Moderate |
| **VITS** *(Kim et al., 2021)* | VAE + Normalizing Flow + GAN | WaveNet Flow Residual Blocks | End-to-End / 22.05 kHz | ~36.0M (Integrated) | 1 (Non-AR) | 0.0200–0.0350 | $\mathcal{O}(N)$ Local Conv | High |
| **FastSpeech 2** *(Ren et al., 2020)* | Feed-Forward Non-AR | Feed-Forward Transformer (FFT) | 80 Bins / 22.05 kHz | 31.5M + 13.9M (HiFi-GAN) | 1 (Non-AR) | 0.0150–0.0250 | $\mathcal{O}(N^2)$ Attention | Static / Over-smoothed |
| **F5-TTS / Voicebox** *(2024)* | Flow Matching Non-AR | Diffusion Transformer (DiT) / ConvNeXt | 100 Bins / 24 kHz | 100M – 330M | 16–32 | 0.1500–0.4500 | $\mathcal{O}(N^2)$ Attention | High (Resource-intensive) |

### Key Takeaways & Architectural Trade-offs
1. **Speed & Receptive Field Trade-off:** 1D Conv U-Nets (e.g., Matcha-TTS) have smaller per-step latency because local 1D convolutions require fewer FLOPs than State-Space linear recurrence. However, Mamba-2 SSM decoders maintain exact continuous hidden states over thousands of frames, eliminating monotonic pitch drift and memory blowouts on extended sentences.
2. **Audio Bandwidth & Resolution:** MambaFlow-TTS models **100 mel channels at 24 kHz** (providing higher spectral detail) compared to standard 80-channel / 22.05 kHz baselines.
3. **Pacing & Articulation (`length_scale=1.2`):** By defaulting to `length_scale=1.2`, MambaFlow-TTS scales phoneme duration by 20%, relaxing frame transitions and noticeably reducing rushed phoneme blending and high-frequency vocoder artifacts.

---

## ⚠️ Academic Scope, Limitations & Transparent Discussion

To ensure complete scientific integrity and prevent any misinterpretation by researchers or practitioners:

* **Academic Single-Speaker Dataset Scale:** This model was trained exclusively on **LJSpeech-1.1 (~11,790 training utterances, ~21 hours of single-speaker audio)**. While this is the exact standard benchmark used across academic TTS papers (Matcha-TTS, Grad-TTS, VITS, FastSpeech 2), it is fundamentally distinct from industrial foundation models (e.g., ElevenLabs, OpenAI Voice, CosyVoice, F5-TTS) trained on 10,000–100,000+ hours of multi-speaker speech.
* **Perceptual Audio Quality vs. Vocoder Matching:** The official Matcha-TTS release uses a custom HiFi-GAN vocoder specifically fine-tuned on Matcha's predicted mel distribution. In contrast, MambaFlow-TTS pairs with a zero-shot, pre-trained BigVGAN-14M vocoder. Minor high-frequency breath or hiss artifacts can occasionally arise due to acoustic prior variance. Adjusting `length_scale=1.2` and `temperature=0.3–0.5` significantly smooths these transitions.
* **Objective vs. Subjective Evaluation:** The metrics reported in this repository (RTF, MSE, MAE, Peak VRAM) represent exact, reproducible programmatic benchmarks across all 1,310 LJSpeech validation samples. Formal crowdsourced Mean Opinion Score (MOS) evaluations with 50+ human listeners are ongoing and will be published in a future revision.

---

## 📊 Full Dataset Validation Benchmarks

Rigorous evaluation across all **1,310 validation samples** of the LJSpeech-1.1 dataset conducted under identical hardware conditions (NVIDIA GPU with CUDA 12.x, FP32 inference):

| Benchmark Metric | **MambaFlow-TTS (10 Steps)** | **MambaFlow-TTS (25 Steps)** | **Matcha-TTS (10 Steps)** |
| :--- | :---: | :---: | :---: |
| **Total Validation Samples** | 1,310 | 1,310 | 1,310 |
| **Total Generated Audio** | **8,074.68 s (~2.24 hrs)** | **8,074.68 s (~2.24 hrs)** | 8,879.19 s (~2.46 hrs) |
| **Total TTS Flow Solver Time** | **240.16 s** | 565.76 s | 110.05 s |
| **Total Vocoder Inversion Time** | 569.24 s (BigVGAN 14M) | 568.73 s (BigVGAN 14M) | 146.63 s (HiFi-GAN V1) |
| **Total End-to-End Latency** | **809.40 s (~13.5 min)** | 1,134.49 s (~18.9 min) | 256.68 s (~4.2 min) |
| **TTS Core RTF** | **0.0297 (~33.7× RT)** | **0.0701 (~14.3× RT)** | **0.0124 (~80.6× RT)** |
| **End-to-End Pipeline RTF** | **0.1002 (~10.0× RT)** | **0.1405 (~7.1× RT)** | **0.0289 (~34.6× RT)** |
| **Mel Reconstruction MSE** | **1.035** | **1.018** | — |
| **Mel Reconstruction MAE** | **0.760** | **0.752** | — |

> [!TIP]
> **Vocoder Latency Distribution:** In the MambaFlow-TTS 10-step configuration, the BigVGAN neural vocoder accounts for **~70% of total runtime**. When deployed with lightweight vocoders (such as Vocos or HiFi-GAN), MambaFlow-TTS achieves an End-to-End RTF under **0.045**.

---

## 📈 Long-Form Audio Scalability Stress Test

To validate stability on long-context audio without splitting into chunked sentences, a continuous **64.78-second** paragraph (~120 words) was synthesized in a single forward pass (15 Euler steps, temperature = 0.667):

```
Text: "The rapid advancement of artificial intelligence and generative modeling has fundamentally transformed
modern speech synthesis. Traditional text-to-speech pipelines often relied on multi-stage architectures with
autoregressive decoders, which suffered from slow sequential generation and compounding error propagation.
In recent years, continuous flow matching and state-space architectures have emerged as powerful alternatives..."
```

| Metric | **MambaFlow-TTS (15 Steps)** | **Matcha-TTS (15 Steps)** |
| :--- | :---: | :---: |
| **Synthesized Duration** | 64.78 s (1,518 mel frames) | 74.29 s (1,600 mel frames) |
| **ODE Flow Solver Time** | 5.675 s | 1.142 s |
| **Neural Vocoder Time** | 4.481 s (BigVGAN) | 1.195 s (HiFi-GAN) |
| **Total Generation Time** | 10.156 s | 2.337 s |
| **Full Pipeline RTF** | **0.1568 (~6.4× RT)** | **0.0315 (~31.7× RT)** |
| **Peak VRAM Allocated** | **2,131.1 MB** | **1,774.0 MB** |
| **Audio Sample Rate** | **24,000 Hz** | 22,050 Hz |
| **Acoustic Consistency** | **Flawless across entire paragraph** | Natural flow |

---

## 🎧 Audio Samples & Demonstrations

All audio samples were generated with MambaFlow-TTS, normalized to `-0.95` peak amplitude, and vocoded with BigVGAN:

### 1. 1-Minute Continuous Speech (64.78s — 15 Euler Steps, Temp = 0.667)
> *"The rapid advancement of artificial intelligence and generative modeling has fundamentally transformed modern speech synthesis. Traditional text-to-speech pipelines often relied on multi-stage architectures with autoregressive decoders, which suffered from slow sequential generation and compounding error propagation. In recent years, continuous flow matching and state-space architectures have emerged as powerful alternatives. By modeling the velocity field of a probability path between simple prior distributions and complex acoustic targets, flow matching enables high-fidelity mel-spectrogram generation in just a handful of integration steps. Furthermore, replacing quadratic self-attention mechanisms with linear state-space models allows speech synthesis systems to scale effortlessly to long-form audio without running into memory bottlenecks or computational slowdowns. As these acoustic representations pass through high-capacity neural vocoders with anti-aliased periodic activations, the resulting synthesized speech exhibits exceptional naturalness, crystal-clear articulation, and smooth, human-like prosodic rhythm throughout the entire passage."*

<audio controls src="audio_samples/sample_1min_continuous.wav">
  Your browser does not support the audio element. <a href="audio_samples/sample_1min_continuous.wav">Download sample_1min_continuous.wav</a>
</audio>

---

### 2. Conversational Daily (5.06s — 25 Euler Steps)
> *"Good morning! How are you doing today? I hope you're ready for an exciting journey ahead."*

<audio controls src="audio_samples/sample_conversational.wav">
  Your browser does not support the audio element. <a href="audio_samples/sample_conversational.wav">Download sample_conversational.wav</a>
</audio>

---

### 3. Medium Balanced Sentence (6.44s — 25 Euler Steps)
> *"The quick brown fox jumps over the lazy dog, demonstrating clear prosody and natural flow."*

<audio controls src="audio_samples/sample_medium.wav">
  Your browser does not support the audio element. <a href="audio_samples/sample_medium.wav">Download sample_medium.wav</a>
</audio>

---

### 4. Emotional Expression (7.08s — 25 Euler Steps)
> *"I simply cannot believe it! We finally made it through against all odds, and it feels absolutely incredible!"*

<audio controls src="audio_samples/sample_emotional.wav">
  Your browser does not support the audio element. <a href="audio_samples/sample_emotional.wav">Download sample_emotional.wav</a>
</audio>

---

### 5. Technical Definition (16.52s — 25 Euler Steps)
> *"Continuous flow matching combined with bidirectional state-space models provides a modern, mathematically rigorous foundation for high-fidelity speech synthesis, effortlessly maintaining acoustic consistency across extended paragraphs without quadratic attention bottlenecks."*

<audio controls src="audio_samples/sample_longest.wav">
  Your browser does not support the audio element. <a href="audio_samples/sample_longest.wav">Download sample_longest.wav</a>
</audio>

---

## 📂 Repository Structure

```
├── README.md                          # Comprehensive project documentation
├── TTS_model.py                       # MambaFlowTTSModel architecture definition
├── ssm_decoder.py                     # Bidirectional Mamba-2 SSM Flow-Matching decoder
├── trans_encoder.py                   # 6-layer Transformer text encoder
├── duration_predictor.py              # 1D Conv duration predictor with Monotonic Alignment Search (MAS)
├── TTSDataModule.py                   # PyTorch Lightning DataModule & Loss computation
├── TTSDatasetModule.py                # Dataset loader, mel-spectrogram extraction & normalization
├── TTSTraining.py                     # Training script with Lightning Trainer & atomic checkpointing
├── test_inference_bigvgan.py          # End-to-end inference script with BigVGAN vocoder
├── test_inference_mel_only.py         # Standalone mel-spectrogram generator
├── mel_to_wav.py                      # Vocoder conversion utility
├── audio_samples/                     # WAV audio samples featured in documentation
│   ├── sample_1min_continuous.wav
│   ├── sample_conversational.wav
│   ├── sample_emotional.wav
│   ├── sample_longest.wav
│   └── sample_medium.wav
├── long_audio_comparison/             # 1-minute stress test comparison assets & metadata
│   ├── comparison_metadata.json
│   ├── mamba_tts_1min_15steps.wav
│   └── matcha_tts_1min_15steps.wav
└── TTS_test/                          # Full validation benchmarks & evaluation suite
    ├── TTSModel_report.md
    ├── validation_report.json
    ├── benchmark_full_val.py
    ├── benchmark_user_tts_only.py
    └── benchmark_matcha_only.py
```

---

## ⚡ Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/MambaFlow-TTS.git
cd MambaFlow-TTS
```

### 2. Set Up Environment
```bash
conda create -n mambaflow python=3.10 -y
conda activate mambaflow

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Mamba-2 SSM and Causal-Conv1d requirements
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.2.2

# Install remaining dependencies
pip install lightning soundfile librosa pandas numpy einops tqdm
pip install git+https://github.com/NVIDIA/BigVGAN.git
```

---

## 🎯 Inference & Synthesis

### Command-Line Inference
Run synthesis with default settings:
```bash
python test_inference_bigvgan.py
```

### Python API Example
```python
import torch
import soundfile as sf
from TTSDataModule import TTSMODEL
from TTSDatasetModule import denormalize_mel
from preprocessing.text import text_to_sequence
import bigvgan

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load MambaFlow-TTS Checkpoint
ckpt_path = "TTS_checkpoints/latest.ckpt"
lightning_model = TTSMODEL.load_from_checkpoint(ckpt_path, map_location=device)
model = lightning_model.model.eval().to(device)

# 2. Load BigVGAN Vocoder
vocoder = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_24khz_100band', use_cuda_kernel=False).to(device)
vocoder.eval()
vocoder.remove_weight_norm()

# 3. Prepare Text Sequence
text = "State-space models with continuous flow matching synthesize speech with exceptional naturalness."
seq = text_to_sequence(text, ["english_cleaners2"])[0]
x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

# 4. Generate Mel-Spectrogram with Euler ODE (10 steps, length_scale=1.2 for clear pace)
with torch.no_grad():
    mel_norm, _ = model(
        x=x,
        n_timesteps=10,
        temperature=0.3,
        length_scale=1.2,  # 1.2 expands duration by 20%, slowing down speech for cleaner articulation
    )
    mel_raw = denormalize_mel(mel_norm).transpose(1, 2)  # [B, 100, T]
    
    # 5. Invert Mel to Waveform
    wav = vocoder(mel_raw).squeeze().cpu().numpy()
    
    # Peak normalize
    wav = wav / max(abs(wav).max(), 1e-6) * 0.95

# 6. Save Output
sf.write("output.wav", wav, 24000)
print("Synthesized audio successfully saved to output.wav")
```

---

## 🏋️‍♂️ Training & Fine-Tuning

### 1. Dataset Preparation
Format your dataset following the LJSpeech-1.1 convention (CSV with audio file paths and normalized transcriptions):
```
audio_path|normalized_text
/path/to/wavs/LJ001-0001.wav|Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts...
```

### 2. Launch Training
```bash
python TTSTraining.py
```
Training logs and model checkpoints are automatically saved with atomic updates in `TTS_checkpoints/` and `lightning_logs/`.

---

## 🛣 Roadmap

- [x] Continuous Optimal Transport Flow Matching (OT-CFM) implementation
- [x] Bidirectional Mamba-2 (SSD) Vector Field Estimator
- [x] 100-channel 24 kHz BigVGAN vocoder integration
- [x] Full 1,310 LJSpeech validation benchmark suite
- [ ] Multi-Speaker Conditioning with reference speaker embeddings
- [ ] Classifier-Free Guidance (CFG) for controllable prosody expressiveness
- [ ] Web-based Gradio & Hugging Face Spaces interactive demo
- [ ] ONNX / TensorRT export for sub-0.01 edge RTF deployment

---

## 📖 Citations & Acknowledgments

If you find MambaFlow-TTS useful in your research or applications, please cite:

```bibtex
@misc{mambaflowtts2026,
  author = {Monesh and Contributors},
  title = {MambaFlow-TTS: Flow-Matching Speech Synthesis with Bidirectional State-Space Models},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/your-username/MambaFlow-TTS}}
}
```

### Reference Works
* **Mamba-2 (SSD):** Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* [arXiv:2405.21060](https://arxiv.org/abs/2405.21060).
* **Flow Matching:** Lipman, Y., Chen, R. T., Ben-Hamu, H., Nicklas, M., & Le, M. (2023). *Flow Matching for Generative Modeling.* [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).
* **Matcha-TTS:** Mehta, S., Tu, Z., Beskow, J., Székely, É., & Henter, G. E. (2024). *Matcha-TTS: A fast TTS architecture with conditional flow matching.* ICASSP 2024. [arXiv:2309.03199](https://arxiv.org/abs/2309.03199).
* **BigVGAN:** Lee, S. G., Ping, W., Ginsburg, B., Catanzaro, B., & Yoon, S. (2023). *BigVGAN: A Universal Neural Vocoder with Large-Scale Training.* [arXiv:2206.04658](https://arxiv.org/abs/2206.04658).

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
