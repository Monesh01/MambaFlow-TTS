# MambaFlow-TTS: Comprehensive Empirical Benchmark & Architectural Diagnosis Report

**Author**: Monesh S  
**Date**: October 2026  
**Repository**: [https://github.com/Monesh01/MambaFlow-TTS](https://github.com/Monesh01/MambaFlow-TTS)  
**Interactive Audio Showcase**: [https://monesh01.github.io/MambaFlow-TTS/](https://monesh01.github.io/MambaFlow-TTS/)

---

## 1. Executive Summary & Core Research Questions

This technical report delivers the complete empirical diagnosis, numerical evaluation, and acoustic analysis of **MambaFlow-TTS**, focusing exclusively on the **Top 5 Distinct Architectures** representing our research progression across three diverse acoustic representations:
1. **Continuous Latent Representation (48 kHz Meta DACVAE)**: 128-dimensional continuous latent space supervised via flow matching, decoupling acoustic temporal modeling from waveform generation.
2. **Native $1\times$ Mel-Spectrogram (24 kHz BigVGAN 14M)**: 100-mel bins at native frame rate, resolving the multi-scale aliasing bottleneck.
3. **Dual-Channel Mel-Spectrogram (22.05 kHz HiFi-GAN V1)**: 80-mel bins comparing non-causal bidirectional state space modeling against causal forward-only modeling.

### Core Research Questions Investigated
- **Q1 (Temporal Aliasing)**: Why do standard multiscale temporal pyramids (U-Nets, feature pyramids) suffer from comb-filtering in high-frequency bands (6–12 kHz), and how does native $1\times$ resolution eliminate it?
- **Q2 (Sequence Directionality)**: What is the quantitative benefit of bidirectional future phonetic context in non-autoregressive flow matching compared to forward-only causal modeling?
- **Q3 (Continuous Latent Flow Matching)**: How does Classifier-Free Guidance (CFG) and sampling temperature ($T$) behave in continuous 128D VAE latent space compared to mel-spectrogram spaces?

---

## 2. Experimental Methodology & Evaluation Protocol

### A. Dataset & Linguistic Frontend
- **Corpus**: LJSpeech-1.1 (single English female speaker, 13,100 total utterances, ~24 hours of speech).
- **Phonetic Representation**: International Phonetic Alphabet (IPA) sequences extracted via CMUDict and `english_cleaners2`.
- **Duration Alignment**: Monotonic Alignment Search (MAS) trained end-to-end to align linguistic tokens with acoustic latent frames, paired with a convolutional duration predictor.

### B. Flow Matching Configuration
- **Generative Framework**: Continuous Optimal Transport Conditional Flow Matching (OT-CFM).
- **Target Flow Vector**: Direct full acoustic target velocity $v_t(x_t, t) = x_1 - (1 - \sigma_{\text{min}})x_0$, with prior $x_0 \sim \mathcal{N}(0, \sigma^2 I)$ and target $x_1 = x_{\text{acoustic}}$.
- **Numerical ODE Solver**: Euler and 2nd-order Runge-Kutta Midpoint solvers with 30 integration steps ($t \in [0, 1]$).
- **Sampling Temperature**: $\tau \in [0.600, 1.000]$ (Optimal: $\tau = 0.667$).
- **Classifier-Free Guidance (CFG)**: Explored $\text{CFG} \in [1.0, 3.0]$ (Optimal: $\text{CFG} = 2.0$).

### C. Unified Metrics Suite
1. **Speech Intelligibility & Pronunciation**:
   - **Whisper Word Error Rate (WER %)**: Automatic Speech Recognition transcript accuracy using OpenAI Faster-Whisper (`tiny.en`).
   - **Phoneme Error Rate (PER %)**: Phoneme edit distance between recognized and reference IPA sequences.
   - **Character Error Rate (CER %)**: Levenshtein character distance.
2. **Perceptual Audio Quality & Naturalness**:
   - **NISQA MOS**: Deep learning speech quality predictor (1.0 to 5.0 scale) measuring overall naturalness, coloration, and discontinuity.
   - **PESQ (Wideband, ITU-T P.862.2)**: Perceptual evaluation of speech quality.
   - **STOI**: Short-Time Objective Intelligibility score.
3. **Speaker Fidelity & Acoustic Alignment**:
   - **WavLM SECS**: Speaker Embedding Cosine Similarity using pretrained `WavLM-SV` comparing synthesized utterances to ground-truth reference audio.
   - **Voiced $F_0$ RMSE (Hz)**: Pitch trajectory root mean square error computed on mutually voiced frames aligned via Dynamic Time Warping (DTW).
   - **Spectral Roll-Off (85% Energy)**: Upper-frequency harmonic retention cutoff in Hz.
4. **ODE Trajectory & Generative Efficiency**:
   - **Trajectory Straightness ($S$)**: Ratio of Euclidean end-to-end displacement to total path length:
     $$S = \frac{\|x_1 - x_0\|_2}{\sum_{k=0}^{K-1} \|x_{t_{k+1}} - x_{t_k}\|_2} \in (0, 1]$$
   - **Real-Time Factor (RTF)** & **Speed (xRT)**: Evaluated on an NVIDIA GPU:
     $$\text{RTF} = \frac{\text{ODE Integration Time (s)}}{\text{Audio Duration (s)}}, \quad \text{Speed} = \frac{1}{\text{RTF}}$$

---

## 3. Master 5-Model Empirical Benchmark Matrix

The table below presents the verified performance across our Top 5 Distinct Architectures:

| Model Architecture | Acoustic Domain | Vocoder & Sample Rate | Total Params | Dec Params | Whisper WER (%) | Whisper PER (%) | NISQA MOS | WavLM SECS | ODE Straightness ($S$) | $F_0$ RMSE (Hz) | RTF | Speed (xRT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MambaFlow-CA-BiMamba-448D** | 128D DACVAE | Meta DACVAE (48 kHz) | **42.20M** | 31.86M | **3.84%** | **1.29%** | **4.62** | 0.973 | 0.8742 | 54.3 | 0.0577 | 17.3× |
| **MambaFlow-CA-BiMamba-256D** | 128D DACVAE | Meta DACVAE (48 kHz) | **18.07M** | 11.23M | 8.50% | 2.90% | 3.95 | 0.954 | 0.8660 | 62.1 | 0.0950 | 10.5× |
| **MambaFlow-Sequential-Tetra** | 100-Mel Native 1× | BigVGAN 14M (24 kHz) | **14.06M** | 10.33M | 4.90% | **0.55%** | 4.30 | **0.982** | 0.9410 | 48.7 | 0.0283 | 35.3× |
| **MambaFlow-BiDirectional-80mel** | 80-Mel Non-Causal | HiFi-GAN V1 (22.05 kHz) | **15.09M** | 10.60M | 4.40% | 0.65% | 4.39 | 0.985 | **0.9810** | 42.1 | 0.0463 | 21.6× |
| **MambaFlow-UniDirectional-80mel** | 80-Mel Causal | HiFi-GAN V1 (22.05 kHz) | **11.14M** | 6.65M | **3.88%** | 0.58% | 4.29 | 0.968 | 0.9800 | 45.3 | **0.0229** | **43.6×** |

---

## 4. In-Depth Comparative Analysis

### A. Continuous Latent Space: MambaFlow-CA-BiMamba-448D (448D Latent Model)
Operating in Meta's 128D DACVAE continuous latent space at 48,000 Hz represents our most expressive acoustic domain:
- **Studio-Quality Naturalness**: Achieves a NISQA MOS of **4.62**, outperforming all mel-spectrogram representations.
- **Intelligibility**: Achieves a low Word Error Rate of **3.84%** and Phoneme Error Rate of **1.29%** on 50 unseen validation sentences.
- **High Trajectory Straightness ($S = 0.8742$)**: Confirms that continuous optimal transport paths remain nearly linear in high-dimensional latent space.

### B. Native $1\times$ Resolution Champion: MambaFlow-Sequential-Tetra
Tetra operates strictly at full $1\times$ temporal resolution without downsampling or pooling:
- **Eradication of Comb-Filtering**: Multiscale temporal pooling introduces Nyquist sidebands during transposed convolution upsampling. When processed by neural vocoders (e.g., BigVGAN's periodic Snake activation), these sidebands manifest as high-frequency harmonic flutter and phase flanging. Tetra completely bypasses this mechanism.
- **Lowest Phoneme Error Rate (PER 0.55%)**: Delivers the highest phonetic clarity among all mel decoders.
- **High Throughput (35.3× Real-Time)**: Fast 4-stage BiMamba-2 decoder processes full-resolution speech with $O(T)$ linear memory complexity.

### C. 80-Mel Domain: Non-Causal Bidirectional vs. Causal Mamba-2
Comparing dual-channel BiMamba-2 against forward-only causal Mamba-2:
- **Bidirectional Advantage**: Future phonetic context reduces validation flow loss from 0.3917 to 0.3313, improves STOI from 0.697 to 0.711, and lowers mel-cepstral distortion from 32.65 dB to 32.11 dB.
- **Causal Advantage**: Unidirectional Mamba-2 requires no future lookahead buffer, doubling synthesis speed to **43.6× Real-Time** (RTF: 0.0229) for ultra-low-latency streaming applications.

---

## 5. Architectural Failure Modes in Deprecated Multiscale Baselines

Our experiments with four multiscale architectures (`Staircase-XTEncoder`, `Staircase-Mamba2`, `TwoStage-Mamba2`, and `UNet-OneBottleneck`, now archived in `archive/multiscale_baselines/`) identified two fundamental failure modes in diffusion and flow-matching speech synthesis:

1. **Nyquist Spectral Sideband Leakage**: Transposed convolution upsampling inserts $S-1$ zeros between consecutive frames. Finite impulse response kernels cannot act as ideal brick-wall filters, leaking aliased energy into the 6–12 kHz bands.
2. **Piecewise-Constant Residual Leakage**: When models predict a residual flow target ($v_t = x_1 - \mu_{\text{prior}}$), step discontinuities in duration-expanded priors leak into the synthesized speech, producing an audible robotic monotone hum. Direct velocity targeting ($v_t = x_1 - x_0$) completely cures this problem.

---

## 6. Conclusion & Recommendation

| Use Case | Recommended Architecture | Key Rationale |
| :--- | :--- | :--- |
| **High-Fidelity Studio Speech (Highest Naturalness)** | `MambaFlow-CA-BiMamba-448D` | NISQA MOS 4.62, 48 kHz DACVAE fidelity, rich prosody. |
| **High-Fidelity Offline Synthesis (Zero Aliasing)** | `MambaFlow-Sequential-Tetra` | Native 1× resolution, zero comb-filtering, PER 0.55%. |
| **Edge & Low-Memory Deployment** | `MambaFlow-CA-BiMamba-256D` | 18M params, 48 kHz DACVAE latent representation. |
| **Ultra-Low Latency Streaming Speech** | `MambaFlow-UniDirectional-80mel` | 43.6× Real-Time throughput, causal forward SSM. |
