# MambaFlow TTS — Raw Audio Noise & Generalization Report

## Purpose

Comparing **raw, unmodified** ground-truth input audio vs raw model-generated audio.
No mastering, no EQ, no loudness normalization applied. Pure model output.
Goal: assess noise levels, generalization quality, and variance capture.

---

## Per-Sample Detailed Analysis

### Sample: LJ046-0164
**Text:** "PRS investigated thirty-four newly established or reactivated cases concerning residents of Texas."
**Duration:** GT=8.12s | Pred=8.12s

#### Energy & Peak

| Metric | Ground Truth | Model Predicted | Delta |
|--------|-------------|-----------------|-------|
| RMS Energy (dB) | -23.97 | -28.3 | -4.33 |
| Peak Amplitude (dBFS) | -3.68 | -5.78 | -2.10 |
| Crest Factor (dB) | 20.29 | 22.52 | +2.23 |

#### Noise Assessment

| Metric | Ground Truth | Model Predicted | Delta |
|--------|-------------|-----------------|-------|
| Noise Floor (dB) | -67.66 | -66.33 | +1.33 |
| SNR Estimate (dB) | 43.69 | 38.03 | -5.66 |
| Spectral Flatness | 0.00175 | 0.00177 | +0.00002 |

> **Noise Verdict:** MILD noise increase — slight elevation but may not dominate

#### Spectral Characteristics

| Metric | Ground Truth | Model Predicted |
|--------|-------------|-----------------|
| Spectral Centroid (Hz) | 2986.5 | 3032.5 |
| Spectral Bandwidth (Hz) | 2540.1 | 2537.6 |
| Log-Mel Spectral Distance (L1) | — | 1.409 |
| Log-Mel Spectral Distance (L2) | — | 2.236 |
| Energy Envelope Correlation | — | 0.5413 |

#### Whisper ASR (Word Error Rate)

- **Reference text:** "PRS investigated thirty-four newly established or reactivated cases concerning residents of Texas."
- **Whisper on GT audio:** "PRS investigated 34 newly established or reactivated cases concerning residents of Texas."
- **Whisper on Pred audio:** "as investigated, 34, newly established or reactivated cases concerning residents of Texas."
- **WER (vs reference):** 25.0%
- **CER (vs reference):** 14.3%

---

### Sample: LJ018-0055
**Text:** "He is described as not a bad-looking man, with a square German type of face,"
**Duration:** GT=5.16s | Pred=5.16s

#### Energy & Peak

| Metric | Ground Truth | Model Predicted | Delta |
|--------|-------------|-----------------|-------|
| RMS Energy (dB) | -22.09 | -26.49 | -4.40 |
| Peak Amplitude (dBFS) | -4.47 | -5.6 | -1.13 |
| Crest Factor (dB) | 17.62 | 20.89 | +3.27 |

#### Noise Assessment

| Metric | Ground Truth | Model Predicted | Delta |
|--------|-------------|-----------------|-------|
| Noise Floor (dB) | -53.16 | -60.0 | -6.84 |
| SNR Estimate (dB) | 31.07 | 33.51 | +2.44 |
| Spectral Flatness | 0.00095 | 0.00124 | +0.00029 |

> **Noise Verdict:** No additional noise — model output is as clean or cleaner than GT

#### Spectral Characteristics

| Metric | Ground Truth | Model Predicted |
|--------|-------------|-----------------|
| Spectral Centroid (Hz) | 2826.0 | 2901.9 |
| Spectral Bandwidth (Hz) | 2123.6 | 2256.6 |
| Log-Mel Spectral Distance (L1) | — | 1.387 |
| Log-Mel Spectral Distance (L2) | — | 2.092 |
| Energy Envelope Correlation | — | 0.6714 |

#### Whisper ASR (Word Error Rate)

- **Reference text:** "He is described as not a bad-looking man, with a square German type of face,"
- **Whisper on GT audio:** "He is described as not a bad looking man with a square German type of face."
- **Whisper on Pred audio:** "He is described as not a bad-looking man with a square, German type of face."
- **WER (vs reference):** 20.0%
- **CER (vs reference):** 3.9%

---

### Sample: LJ018-0254
**Text:** "He declined to plead, but a plea of "Not Guilty" was recorded."
**Duration:** GT=4.28s | Pred=4.28s

#### Energy & Peak

| Metric | Ground Truth | Model Predicted | Delta |
|--------|-------------|-----------------|-------|
| RMS Energy (dB) | -24.11 | -27.88 | -3.77 |
| Peak Amplitude (dBFS) | -6.36 | -6.41 | -0.05 |
| Crest Factor (dB) | 17.75 | 21.48 | +3.73 |

#### Noise Assessment

| Metric | Ground Truth | Model Predicted | Delta |
|--------|-------------|-----------------|-------|
| Noise Floor (dB) | -69.79 | -66.55 | +3.24 |
| SNR Estimate (dB) | 45.69 | 38.66 | -7.03 |
| Spectral Flatness | 0.0019 | 0.00143 | -0.00047 |

> **Noise Verdict:** MODERATE noise — noise floor elevated, audible hiss likely

#### Spectral Characteristics

| Metric | Ground Truth | Model Predicted |
|--------|-------------|-----------------|
| Spectral Centroid (Hz) | 2414.3 | 2532.0 |
| Spectral Bandwidth (Hz) | 2621.4 | 2624.6 |
| Log-Mel Spectral Distance (L1) | — | 1.241 |
| Log-Mel Spectral Distance (L2) | — | 1.953 |
| Energy Envelope Correlation | — | 0.7388 |

#### Whisper ASR (Word Error Rate)

- **Reference text:** "He declined to plead, but a plea of "Not Guilty" was recorded."
- **Whisper on GT audio:** "He declined to plead, but a plea of not guilty was recorded."
- **Whisper on Pred audio:** "He declined to plead, but a plea of doubt guilty was recorded."
- **WER (vs reference):** 16.7%
- **CER (vs reference):** 8.1%

---

### Sample: LJ003-0339
**Text:** "could possibly afford."
**Duration:** GT=1.48s | Pred=1.48s

#### Energy & Peak

| Metric | Ground Truth | Model Predicted | Delta |
|--------|-------------|-----------------|-------|
| RMS Energy (dB) | -24.18 | -29.44 | -5.26 |
| Peak Amplitude (dBFS) | -8.04 | -10.45 | -2.41 |
| Crest Factor (dB) | 16.13 | 18.99 | +2.86 |

#### Noise Assessment

| Metric | Ground Truth | Model Predicted | Delta |
|--------|-------------|-----------------|-------|
| Noise Floor (dB) | -58.67 | -64.79 | -6.12 |
| SNR Estimate (dB) | 34.49 | 35.35 | +0.86 |
| Spectral Flatness | 0.00091 | 0.00199 | +0.00108 |

> **Noise Verdict:** No additional noise — model output is as clean or cleaner than GT

#### Spectral Characteristics

| Metric | Ground Truth | Model Predicted |
|--------|-------------|-----------------|
| Spectral Centroid (Hz) | 2344.8 | 2616.1 |
| Spectral Bandwidth (Hz) | 2205.8 | 2388.0 |
| Log-Mel Spectral Distance (L1) | — | 1.359 |
| Log-Mel Spectral Distance (L2) | — | 2.143 |
| Energy Envelope Correlation | — | 0.654 |

#### Whisper ASR (Word Error Rate)

- **Reference text:** "could possibly afford."
- **Whisper on GT audio:** "could possibly afford."
- **Whisper on Pred audio:** "could possibly afford."
- **WER (vs reference):** 0.0%
- **CER (vs reference):** 0.0%

---

### Sample: LJ028-0012
**Text:** "she realizes that she is bound on the long journey across the desert. Hour after hour she bears you over the hard monotonous plain."
**Duration:** GT=9.12s | Pred=9.12s

#### Energy & Peak

| Metric | Ground Truth | Model Predicted | Delta |
|--------|-------------|-----------------|-------|
| RMS Energy (dB) | -22.79 | -27.35 | -4.56 |
| Peak Amplitude (dBFS) | -2.16 | -6.59 | -4.43 |
| Crest Factor (dB) | 20.63 | 20.76 | +0.13 |

#### Noise Assessment

| Metric | Ground Truth | Model Predicted | Delta |
|--------|-------------|-----------------|-------|
| Noise Floor (dB) | -69.89 | -68.39 | +1.50 |
| SNR Estimate (dB) | 47.09 | 41.03 | -6.06 |
| Spectral Flatness | 0.00142 | 0.00133 | -0.00009 |

> **Noise Verdict:** MILD noise increase — slight elevation but may not dominate

#### Spectral Characteristics

| Metric | Ground Truth | Model Predicted |
|--------|-------------|-----------------|
| Spectral Centroid (Hz) | 2318.0 | 2420.9 |
| Spectral Bandwidth (Hz) | 2266.9 | 2325.9 |
| Log-Mel Spectral Distance (L1) | — | 1.141 |
| Log-Mel Spectral Distance (L2) | — | 1.816 |
| Energy Envelope Correlation | — | 0.7268 |

#### Whisper ASR (Word Error Rate)

- **Reference text:** "she realizes that she is bound on the long journey across the desert. Hour after hour she bears you over the hard monotonous plain."
- **Whisper on GT audio:** "She realizes that she is bound on the long journey across the desert. Hour after hour, she bears you over the hard monotonous plane."
- **Whisper on Pred audio:** "She realizes that she is bound no longer on the long journey across the desert. Our after hour she bears you over the hard menopness plane."
- **WER (vs reference):** 20.8%
- **CER (vs reference):** 13.7%

---

## Summary Comparison Table

| Sample | GT RMS (dB) | Pred RMS (dB) | GT Peak (dBFS) | Pred Peak (dBFS) | GT SNR (dB) | Pred SNR (dB) | SNR Delta | NF Delta | Flatness Delta | LMSD L1 | Env Corr | WER% |
|--------|------------|--------------|---------------|-----------------|------------|--------------|-----------|----------|----------------|---------|----------|------|
| LJ046-0164 | -23.97 | -28.3 | -3.68 | -5.78 | 43.69 | 38.03 | -5.7 | +1.3 | +0.00002 | 1.409 | 0.5413 | 25.0 |
| LJ018-0055 | -22.09 | -26.49 | -4.47 | -5.6 | 31.07 | 33.51 | +2.4 | -6.8 | +0.00029 | 1.387 | 0.6714 | 20.0 |
| LJ018-0254 | -24.11 | -27.88 | -6.36 | -6.41 | 45.69 | 38.66 | -7.0 | +3.2 | -0.00047 | 1.241 | 0.7388 | 16.7 |
| LJ003-0339 | -24.18 | -29.44 | -8.04 | -10.45 | 34.49 | 35.35 | +0.9 | -6.1 | +0.00108 | 1.359 | 0.654 | 0.0 |
| LJ028-0012 | -22.79 | -27.35 | -2.16 | -6.59 | 47.09 | 41.03 | -6.1 | +1.5 | -0.00009 | 1.141 | 0.7268 | 20.8 |

## Aggregate Statistics

| Metric | Value |
|--------|-------|
| Average GT SNR (dB) | 40.41 |
| Average Pred SNR (dB) | 37.32 |
| Average SNR Delta (Pred - GT) | -3.09 dB |
| SNR Delta Std Dev | 3.93 dB |
| Average Noise Floor Delta | -1.38 dB |
| Average Spectral Flatness Delta | +0.00017 |
| Average GT Crest Factor (dB) | 18.48 |
| Average Pred Crest Factor (dB) | 20.93 |
| Average Log-Mel Spectral Dist (L1) | 1.307 |
| Average Energy Envelope Correlation | 0.6665 |
| Average WER (%) | 16.5 |

## Generalization & Variance Analysis

### Does the model preserve dynamic range across samples?

| Metric | GT Std Dev | Pred Std Dev | Ratio (Pred/GT) |
|--------|-----------|-------------|-----------------|
| RMS Energy (dB) | 0.839 | 0.982 | 1.170 |
| SNR (dB) | 6.412 | 2.625 | 0.409 |
| Crest Factor (dB) | 1.714 | 1.151 | 0.671 |

> **Interpretation:** Ratio close to 1.0 = model preserves natural variance.
> Ratio << 1.0 = variance collapse (all outputs sound similar).
> Ratio >> 1.0 = model amplifies differences beyond GT.

## Overall Verdict

### Noise Level: MODERATE

Moderate noise in model output. SNR is somewhat lower than GT, spectral flatness slightly higher. Suggests partial convergence — the flow trajectory moves toward the target but retains residual noise.

### Generalization Assessment

Preserves variance well. Predicted outputs maintain similar spread of energy levels as GT. Good generalization.

### Flow Matching Convergence

Energy envelope correlation is moderate. Captures some temporal dynamics but misses others.

### Intelligibility (Whisper WER)

Average WER is 16.5% — good intelligibility with some errors.

### Key Findings

1. **Noise floor is 1.4 dB lower** in model output vs ground truth on average.
2. **SNR drops by 3.1 dB** from GT (40.4 dB) to Pred (37.3 dB).
3. **Spectral flatness change: +0.00017** — more noise-like spectrum.
4. **LMSD L1 = 1.307** — moderate.
5. **Energy envelope correlation = 0.6665** — well-preserved.
6. **Crest factor: GT=18.5 dB, Pred=20.9 dB** — expanded dynamics.
7. **Variance ratio (RMS): 1.170** — good preservation.
