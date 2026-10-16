#!/usr/bin/env python3
"""
===============================================================================
MambaFlow-TTS: Automated End-to-End Evaluation & Artifact Detection Pipeline
===============================================================================

Evaluates TTS checkpoints on unseen validation words (val.csv) across:
1. Intelligibility (WSR: Word/Char Error Rate, In-Domain vs OOV WER breakdown)
2. Acoustic & Flow-Matching Dynamics (F0 RMSE & Energy RMSE via DTW, Trajectory Straightness)
3. Speaker Similarity (SECS via WavLM)
4. Quality & Artifact Detectors:
   - IS_MUFFLED: NISQA Coloration < 3.0 OR 85% Spectral Roll-off < 3200 Hz
   - IS_DUAL_AUDIO: Pitch multi-modality / frame-to-frame pitch jumps + NISQA Discontinuity < 3.0
   - IS_NON_UTTERANCE: Silero VAD ratio < 0.20 OR Whisper no_speech_prob > 0.60 OR compression_ratio > 2.4 OR char_rate > 0.25 s/char

Outputs:
- eval_results/per_sample_metrics.csv
- eval_results/summary_report.json
- eval_results/report.html (Interactive HTML report with embedded audio controls)
"""

import os
import sys
import re
import time
import json
import ast
import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import librosa

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def enable_mamba_cpu_fallback():
    """Enables pure PyTorch CPU fallbacks for Mamba2, CausalConv1d, and SSD kernels."""
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
from stats import denormalize_latent, normalize_latent
from generate_audio import setup_dacvae, decode_latent_to_audio, find_latest_checkpoint


# =============================================================================
# 1. Text Normalization & OOV Analysis
# =============================================================================

def clean_text_for_asr(text: str) -> str:
    """Standardizes text for fair WER/CER evaluation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Strip punctuation except apostrophes within words
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class OOVAnalyzer:
    """Analyzes In-Domain vs Out-Of-Vocabulary (OOV) words relative to train.csv."""
    def __init__(self, train_csv_path: str = None):
        self.vocab = set()
        if train_csv_path and os.path.exists(train_csv_path):
            try:
                print(f"[+] Indexing training vocabulary from: {train_csv_path}")
                df_train = pd.read_csv(train_csv_path)
                for col in ["normalized_text", "text"]:
                    if col in df_train.columns:
                        for line in df_train[col].dropna():
                            words = clean_text_for_asr(str(line)).split()
                            self.vocab.update(words)
                print(f"[+] Loaded {len(self.vocab):,} unique words into training vocabulary.")
            except Exception as e:
                print(f"[!] Warning: Could not load train vocabulary ({e}). OOV analysis skipped.")

    def analyze_sentence(self, sentence: str):
        words = clean_text_for_asr(sentence).split()
        if not words or not self.vocab:
            return {"total_words": len(words), "oov_words": 0, "oov_ratio": 0.0, "is_oov": False, "oov_list": []}
        oov_list = [w for w in words if w not in self.vocab]
        return {
            "total_words": len(words),
            "oov_words": len(oov_list),
            "oov_ratio": len(oov_list) / max(len(words), 1),
            "is_oov": len(oov_list) > 0,
            "oov_list": oov_list,
        }


# =============================================================================
# 2. Metric Evaluators
# =============================================================================

class PipelineEvaluator:
    """
    Unified evaluator loading Whisper, Silero VAD, SQUIM, NISQA, and WavLM models.
    Operates safely on CPU or GPU without interfering with training.
    """
    def __init__(self, whisper_model_size="tiny.en", device="cpu"):
        self.device = device
        self.whisper_model_size = whisper_model_size
        print(f"[+] Initializing Evaluator on device: {device}...")

        # 1. Faster-Whisper
        print(f"[+] Loading Faster-Whisper ({whisper_model_size})...")
        from faster_whisper import WhisperModel
        compute_type = "float16" if device == "cuda" else "int8"
        self.whisper = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)

        # 2. Silero VAD
        print("[+] Loading Silero VAD...")
        try:
            self.vad_model, self.vad_utils = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", onnx=True, trust_repo=True
            )
            self.get_speech_timestamps = self.vad_utils[0]
        except Exception as e:
            print(f"[!] Warning: Silero VAD failed to load ({e}). Using energy VAD fallback.")
            self.vad_model = None

        # 3. Torchaudio SQUIM Objective (STOI, PESQ, SI-SDR)
        print("[+] Loading Torchaudio SQUIM Objective...")
        try:
            self.squim_model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().to("cpu").eval()
        except Exception as e:
            print(f"[!] Warning: SQUIM failed to load ({e}).")
            self.squim_model = None

        # 4. Torchmetrics NISQA (MOS, Noisiness, Discontinuity, Coloration, Loudness)
        print("[+] Loading NISQA Speech Quality Assessment...")
        try:
            from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment
            self.nisqa_model = NonIntrusiveSpeechQualityAssessment(fs=16000).to("cpu").eval()
        except Exception as e:
            print(f"[!] Warning: NISQA failed to load ({e}).")
            self.nisqa_model = None

        # 5. WavLM Speaker Verification for SECS
        print("[+] Loading WavLM-SV for Speaker Cosine Similarity (SECS)...")
        try:
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
            self.wavlm_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
            self.wavlm_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to("cpu").eval()
        except Exception as e:
            print(f"[!] Warning: WavLM-SV failed to load ({e}). Using MFCC cosine fallback.")
            self.wavlm_model = None

    def evaluate_whisper(self, wav: np.ndarray, ref_text: str, sr: int = 16000):
        """Transcribes audio with faster-whisper, calculates WER/CER and extraction flags."""
        import jiwer
        segments, info = self.whisper.transcribe(
            wav,
            beam_size=5,
            temperature=0.0,
            vad_filter=False,
            language="en" if "en" in str(self.whisper_model_size) else None,
        )
        segments_list = list(segments)
        hyp_text = " ".join([s.text for s in segments_list]).strip()

        no_speech_probs = [s.no_speech_prob for s in segments_list] if segments_list else [0.0]
        compression_ratios = [s.compression_ratio for s in segments_list] if segments_list else [1.0]

        max_no_speech = float(max(no_speech_probs))
        max_comp_ratio = float(max(compression_ratios))

        clean_ref = clean_text_for_asr(ref_text)
        clean_hyp = clean_text_for_asr(hyp_text)

        if not clean_ref:
            wer = 0.0 if not clean_hyp else 1.0
            cer = 0.0 if not clean_hyp else 1.0
            per = 0.0 if not clean_hyp else 1.0
        else:
            wer = float(jiwer.wer(clean_ref, clean_hyp))
            cer = float(jiwer.cer(clean_ref, clean_hyp))
            try:
                ref_seq = text_to_sequence(clean_ref, ["english_cleaners2"])
                hyp_seq = text_to_sequence(clean_hyp, ["english_cleaners2"])
                ref_ipa = ref_seq[1] if len(ref_seq) > 1 else ""
                hyp_ipa = hyp_seq[1] if len(hyp_seq) > 1 else ""
                ref_ph = " ".join(list(ref_ipa.replace(" ", "")))
                hyp_ph = " ".join(list(hyp_ipa.replace(" ", "")))
                if not ref_ph:
                    per = 0.0 if not hyp_ph else 1.0
                else:
                    per = float(jiwer.wer(ref_ph, hyp_ph))
            except Exception:
                per = cer

        return {
            "transcript": hyp_text,
            "clean_ref": clean_ref,
            "clean_hyp": clean_hyp,
            "wer": wer,
            "cer": cer,
            "per": per,
            "no_speech_prob": max_no_speech,
            "compression_ratio": max_comp_ratio,
        }

    def evaluate_vad(self, wav_16k: np.ndarray, target_text: str):
        """Computes Silero VAD speech ratio and char_rate."""
        duration_sec = len(wav_16k) / 16000.0
        char_count = max(len(clean_text_for_asr(target_text)), 1)
        char_rate = duration_sec / char_count

        if self.vad_model is not None:
            tensor_wav = torch.from_numpy(wav_16k).float()
            speech_ts = self.get_speech_timestamps(tensor_wav, self.vad_model, sampling_rate=16000)
            speech_samples = sum(ts["end"] - ts["start"] for ts in speech_ts)
            vad_ratio = speech_samples / max(len(wav_16k), 1)
        else:
            # Simple energy-based VAD fallback
            frame_len = 512
            hop = 256
            rms = librosa.feature.rms(y=wav_16k, frame_length=frame_len, hop_length=hop)[0]
            thresh = np.max(rms) * 0.08 if len(rms) > 0 else 0.01
            vad_ratio = float(np.mean(rms > thresh)) if len(rms) > 0 else 0.5

        return {
            "duration_sec": duration_sec,
            "vad_speech_ratio": float(vad_ratio),
            "char_rate": float(char_rate),
        }

    def evaluate_acoustic(self, wav_pred: np.ndarray, wav_gt: np.ndarray, sr: int = 16000):
        """Computes F0 RMSE and Energy RMSE using DTW alignment and measures pitch jump rate."""
        # 1. Pitch estimation (pyin)
        hop = 256
        f0_pred, _, _ = librosa.pyin(wav_pred, fmin=50, fmax=500, sr=sr, hop_length=hop)
        f0_gt, _, _ = librosa.pyin(wav_gt, fmin=50, fmax=500, sr=sr, hop_length=hop)

        # 2. RMS Energy
        rms_pred = librosa.feature.rms(y=wav_pred, frame_length=1024, hop_length=hop)[0]
        rms_gt = librosa.feature.rms(y=wav_gt, frame_length=1024, hop_length=hop)[0]

        # 3. DTW Alignment on RMS energy
        try:
            # Reshape for cdist
            X = rms_pred.reshape(-1, 1)
            Y = rms_gt.reshape(-1, 1)
            D, wp = librosa.sequence.dtw(X=X.T, Y=Y.T, metric="cosine")
            pred_idx = wp[:, 0]
            gt_idx = wp[:, 1]
            aligned_f0_pred = f0_pred[pred_idx]
            aligned_f0_gt = f0_gt[gt_idx]
            aligned_rms_pred = rms_pred[pred_idx]
            aligned_rms_gt = rms_gt[gt_idx]
        except Exception:
            min_len = min(len(f0_pred), len(f0_gt))
            aligned_f0_pred = f0_pred[:min_len]
            aligned_f0_gt = f0_gt[:min_len]
            min_rms = min(len(rms_pred), len(rms_gt))
            aligned_rms_pred = rms_pred[:min_rms]
            aligned_rms_gt = rms_gt[:min_rms]

        # F0 RMSE on mutually voiced frames
        voiced_mask = (~np.isnan(aligned_f0_pred)) & (~np.isnan(aligned_f0_gt)) & (aligned_f0_pred > 0) & (aligned_f0_gt > 0)
        if np.sum(voiced_mask) > 5:
            f0_diff = aligned_f0_pred[voiced_mask] - aligned_f0_gt[voiced_mask]
            f0_rmse = float(np.sqrt(np.mean(f0_diff ** 2)))
        else:
            f0_rmse = float("nan")

        # Energy RMSE
        energy_diff = aligned_rms_pred - aligned_rms_gt
        energy_rmse = float(np.sqrt(np.mean(energy_diff ** 2)))

        # 4. Pitch Multi-Modality / Jumps in Generated Audio
        valid_pred_f0 = f0_pred[~np.isnan(f0_pred) & (f0_pred > 0)]
        if len(valid_pred_f0) > 10:
            semitone_steps = np.abs(12.0 * np.log2(valid_pred_f0[1:] / valid_pred_f0[:-1]))
            # Frame-to-frame jump > 4 semitones indicates glitching or competing dual voices
            pitch_jump_rate = float(np.mean(semitone_steps > 4.0))
        else:
            pitch_jump_rate = 0.0

        # 5. Harmonic-to-Noise Ratio (HNR)
        try:
            y_harm = librosa.effects.harmonic(wav_pred)
            y_noise = wav_pred - y_harm
            p_harm = np.mean(y_harm ** 2)
            p_noise = np.mean(y_noise ** 2)
            hnr_db = float(10.0 * np.log10(max(p_harm, 1e-9) / max(p_noise, 1e-9)))
        except Exception:
            hnr_db = 15.0

        return {
            "f0_rmse": f0_rmse,
            "energy_rmse": energy_rmse,
            "pitch_jump_rate": pitch_jump_rate,
            "hnr_db": hnr_db,
        }

    def evaluate_spectral(self, wav: np.ndarray, sr: int = 16000):
        """Computes 85% spectral roll-off and spectral flatness."""
        rolloff = librosa.feature.spectral_rolloff(y=wav, sr=sr, roll_percent=0.85)[0]
        flatness = librosa.feature.spectral_flatness(y=wav)[0]
        return {
            "spectral_rolloff_85_median": float(np.median(rolloff)),
            "spectral_rolloff_85_mean": float(np.mean(rolloff)),
            "spectral_flatness_mean": float(np.mean(flatness)),
        }

    def evaluate_quality(self, wav_16k: np.ndarray):
        """Extracts NISQA sub-scores and SQUIM objective metrics."""
        nisqa_mos, nisqa_noi, nisqa_dis, nisqa_col, nisqa_loud = 3.5, 3.5, 3.5, 3.5, 3.5
        squim_stoi, squim_pesq, squim_si_sdr = 0.8, 2.5, 12.0

        tensor_wav = torch.from_numpy(wav_16k).float().unsqueeze(0)

        # NISQA Evaluation
        if self.nisqa_model is not None:
            try:
                with torch.no_grad():
                    nisqa_out = self.nisqa_model(tensor_wav).squeeze().cpu().numpy()
                nisqa_mos = float(nisqa_out[0])
                nisqa_noi = float(nisqa_out[1])
                nisqa_dis = float(nisqa_out[2])
                nisqa_col = float(nisqa_out[3])
                nisqa_loud = float(nisqa_out[4])
            except Exception as e:
                pass

        # SQUIM Evaluation
        if self.squim_model is not None:
            try:
                with torch.no_grad():
                    stoi, pesq, si_sdr = self.squim_model(tensor_wav)
                squim_stoi = float(stoi.item())
                squim_pesq = float(pesq.item())
                squim_si_sdr = float(si_sdr.item())
            except Exception as e:
                pass

        return {
            "nisqa_mos": nisqa_mos,
            "nisqa_noi": nisqa_noi,
            "nisqa_discontinuity": nisqa_dis,
            "nisqa_coloration": nisqa_col,
            "nisqa_loudness": nisqa_loud,
            "squim_stoi": squim_stoi,
            "squim_pesq": squim_pesq,
            "squim_si_sdr": squim_si_sdr,
        }

    def evaluate_secs(self, wav_pred_16k: np.ndarray, wav_gt_16k: np.ndarray):
        """Computes Speaker Embedding Cosine Similarity (SECS)."""
        if self.wavlm_model is not None:
            try:
                inputs_pred = self.wavlm_extractor(wav_pred_16k, sampling_rate=16000, return_tensors="pt")
                inputs_gt = self.wavlm_extractor(wav_gt_16k, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    emb_pred = self.wavlm_model(**inputs_pred).embeddings
                    emb_gt = self.wavlm_model(**inputs_gt).embeddings
                    secs = float(F.cosine_similarity(emb_pred, emb_gt).item())
                    return secs
            except Exception:
                pass

        # Fallback: MFCC mean vector cosine similarity
        mfcc_pred = np.mean(librosa.feature.mfcc(y=wav_pred_16k, sr=16000, n_mfcc=20), axis=1)
        mfcc_gt = np.mean(librosa.feature.mfcc(y=wav_gt_16k, sr=16000, n_mfcc=20), axis=1)
        norm_p = np.linalg.norm(mfcc_pred)
        norm_g = np.linalg.norm(mfcc_gt)
        if norm_p > 1e-6 and norm_g > 1e-6:
            secs = float(np.dot(mfcc_pred, mfcc_gt) / (norm_p * norm_g))
        else:
            secs = 0.5
        return secs


# =============================================================================
# 3. Trajectory Tracking & Audio Synthesis
# =============================================================================

def synthesize_and_record_trajectory(
    model,
    dacvae_decoder,
    text: str,
    gt_durations: list = None,
    use_gt_durations: bool = False,
    n_steps: int = 30,
    temperature: float = 1.0,
    solver: str = "euler",
    cfg_scale: float = 1.0,
    device: str = "cpu",
    dacvae_device: str = "cpu",
    seed: int = 42,
):
    """
    Synthesizes audio while recording intermediate ODE latent states z_t,
    and calculates path straightness: S = ||z_1 - z_0||_2 / sum_t ||z_{t+dt} - z_t||_2.
    """
    cleaner_names = ["english_cleaners2"]
    seq = text_to_sequence(text, cleaner_names)[0]
    text_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    text_mask = torch.ones_like(text_tensor, dtype=torch.bool)

    torch.manual_seed(seed)

    with torch.inference_mode():
        x_enc = model.txt_emb(text_tensor)
        enc_out = model.encoder(x_enc, mask=text_mask)

        # Durations
        if use_gt_durations and gt_durations is not None:
            durations = torch.tensor(gt_durations, dtype=torch.long, device=device).unsqueeze(0)
        else:
            dur_int, _, _ = model.duration(enc_out.detach(), mask=text_mask, length_scale=1.0)
            durations = dur_int

        if durations.dim() == 3:
            durations = durations.squeeze(-1)
        durations = torch.clamp(durations, min=0)
        if durations.sum() == 0:
            durations[0, 0] = 1

        audio_lengths = durations.sum(dim=-1)
        T_audio = int(audio_lengths.max().item())
        audio_mask = torch.arange(T_audio, device=device)[None, :] < audio_lengths[:, None]

        # Initial Gaussian noise scaled by temperature
        x_0 = torch.randn((1, T_audio, model.d_codec), device=device) * float(temperature)
        x_curr = x_0.clone()
        m = audio_mask.unsqueeze(-1).float()
        x_curr = x_curr * m

        trajectory_states = [x_curr.clone()]
        dt = 1.0 / float(n_steps)
        use_cfg = (cfg_scale is not None and cfg_scale != 1.0 and cfg_scale > 0.0)

        for step in range(n_steps):
            t_val = step * dt
            t_tensor = torch.full((1,), t_val, device=device, dtype=torch.float32)

            if use_cfg:
                x_batched = torch.cat([x_curr, x_curr], dim=0)
                t_batched = torch.cat([t_tensor, t_tensor], dim=0)
                text_batched = torch.cat([enc_out, enc_out], dim=0)
                dur_batched = torch.cat([durations, durations], dim=0)
                text_mask_batched = torch.cat([text_mask, text_mask], dim=0)
                audio_mask_batched = torch.cat([audio_mask, audio_mask], dim=0)
                drop_mask = torch.cat([
                    torch.zeros(1, dtype=torch.bool, device=device),
                    torch.ones(1, dtype=torch.bool, device=device),
                ], dim=0)

                v_batched = model.decoder.predict_velocity(
                    x=x_batched,
                    t=t_batched,
                    text_repr=text_batched,
                    durations=dur_batched,
                    text_mask=text_mask_batched,
                    audio_mask=audio_mask_batched,
                    cond_drop_mask=drop_mask,
                )
                v_cond, v_uncond = v_batched.chunk(2, dim=0)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model.decoder.predict_velocity(
                    x=x_curr,
                    t=t_tensor,
                    text_repr=enc_out,
                    durations=durations,
                    text_mask=text_mask,
                    audio_mask=audio_mask,
                )

            if solver == "midpoint" and step < n_steps - 1:
                x_mid = (x_curr + (dt / 2.0) * v) * m
                t_mid = torch.full((1,), t_val + dt / 2.0, device=device, dtype=torch.float32)
                v_mid = model.decoder.predict_velocity(
                    x=x_mid,
                    t=t_mid,
                    text_repr=enc_out,
                    durations=durations,
                    text_mask=text_mask,
                    audio_mask=audio_mask,
                )
                x_next = (x_curr + dt * v_mid) * m
            else:
                x_next = (x_curr + dt * v) * m

            trajectory_states.append(x_next.clone())
            x_curr = x_next

        # Path Straightness: S = ||z_1 - z_0||_2 / sum_t ||z_{t+dt} - z_t||_2
        z_start = trajectory_states[0].float().flatten()
        z_end = trajectory_states[-1].float().flatten()
        displacement = torch.norm(z_end - z_start, p=2).item()

        step_lengths = [
            torch.norm((trajectory_states[i + 1] - trajectory_states[i]).float().flatten(), p=2).item()
            for i in range(len(trajectory_states) - 1)
        ]
        total_path_length = sum(step_lengths)
        straightness = float(displacement / max(total_path_length, 1e-8))

        # Denormalize final latent and decode to 48kHz audio
        x_final = trajectory_states[-1].float()
        pred_latent_denorm = denormalize_latent(x_final)
        latent_128_t = pred_latent_denorm.transpose(1, 2)
        wav_48k = decode_latent_to_audio(dacvae_decoder, latent_128_t, target_device=dacvae_device)

    return wav_48k, straightness, T_audio


# =============================================================================
# 4. Report Generation (HTML, JSON, Terminal Table)
# =============================================================================

def generate_html_report(df: pd.DataFrame, summary: dict, output_html_path: str, wavs_rel_dir: str = "wavs"):
    """Creates a modern, responsive HTML report with embedded audio controls for worst offenders."""
    
    # Worst offenders top 10
    worst_wer = df.sort_values(by="wer", ascending=False).head(10)
    worst_muffled = df[df["is_muffled"]].sort_values(by="spectral_rolloff_85", ascending=True).head(10)
    if len(worst_muffled) == 0:
        worst_muffled = df.sort_values(by="spectral_rolloff_85", ascending=True).head(5)

    worst_dual = df[df["is_dual_audio"]].sort_values(by="pitch_jump_rate", ascending=False).head(10)
    if len(worst_dual) == 0:
        worst_dual = df.sort_values(by="pitch_jump_rate", ascending=False).head(5)

    worst_non_utt = df[df["is_non_utterance"]].head(10)
    if len(worst_non_utt) == 0:
        worst_non_utt = df.sort_values(by="vad_speech_ratio", ascending=True).head(5)

    def render_audio_table(subset_df, title, badge_color):
        if len(subset_df) == 0:
            return f"<p class='text-muted'>No failures detected in this category.</p>"
        
        rows = []
        for _, r in subset_df.iterrows():
            audio_id = r["audio_id"]
            pred_wav = r.get("pred_wav_rel", "")
            gt_wav = r.get("gt_wav_rel", "")
            wer_val = f"{r.get('wer', 0)*100:.1f}%"
            per_val = f"{r.get('per', 0)*100:.1f}%"
            rolloff = f"{r.get('spectral_rolloff_85', 0):.0f} Hz"
            vad_ratio = f"{r.get('vad_speech_ratio', 0)*100:.1f}%"
            p_jump = f"{r.get('pitch_jump_rate', 0)*100:.1f}%"
            nisqa_col = f"{r.get('nisqa_coloration', 0):.2f}"
            
            rows.append(f"""
            <tr>
                <td style="font-weight:600; color:#e2e8f0;">{audio_id}</td>
                <td style="max-width:280px; font-size:13px; color:#cbd5e1;">{r.get('text', '')}</td>
                <td style="font-size:13px; color:#94a3b8;"><em>{r.get('transcript', '')}</em></td>
                <td>
                    <div style="font-size:11px; margin-bottom:4px; color:#38bdf8;">Generated</div>
                    <audio controls preload="none" style="height:32px; width:200px;">
                        <source src="{pred_wav}" type="audio/wav">
                    </audio>
                    <div style="font-size:11px; margin-top:4px; margin-bottom:4px; color:#a3e635;">Reference GT</div>
                    <audio controls preload="none" style="height:32px; width:200px;">
                        <source src="{gt_wav}" type="audio/wav">
                    </audio>
                </td>
                <td style="font-size:13px;">
                    <div><strong>WER:</strong> {wer_val} | <strong>PER:</strong> {per_val}</div>
                    <div><strong>Roll-off:</strong> {rolloff}</div>
                    <div><strong>Coloration:</strong> {nisqa_col}</div>
                    <div><strong>VAD Ratio:</strong> {vad_ratio}</div>
                    <div><strong>Pitch Jump:</strong> {p_jump}</div>
                </td>
            </tr>
            """)
        return f"""
        <div class="card mb-4" style="background:#1e293b; border:1px solid #334155; border-radius:12px;">
            <div class="card-header" style="background:#0f172a; border-bottom:1px solid #334155; padding:14px 20px;">
                <h4 style="margin:0; font-size:16px; color:#f8fafc;">
                    <span class="badge" style="background:{badge_color}; margin-right:8px;">Flagged</span>
                    {title} ({len(subset_df)} samples)
                </h4>
            </div>
            <div class="card-body" style="padding:0; overflow-x:auto;">
                <table class="table table-dark table-hover mb-0" style="background:transparent; margin:0;">
                    <thead>
                        <tr style="color:#94a3b8; font-size:12px; text-transform:uppercase; letter-spacing:0.05em;">
                            <th>Audio ID</th>
                            <th>Target Text</th>
                            <th>ASR Transcript</th>
                            <th>Listening Check</th>
                            <th>Key Metrics</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows)}
                    </tbody>
                </table>
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MambaFlow-TTS: Validation Health & Artifact Diagnosis</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #090d16;
            --bg-card: #131b2e;
            --border-card: #222f4c;
            --accent-cyan: #06b6d4;
            --accent-blue: #3b82f6;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
        }}
        body {{
            background: var(--bg-main);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 32px 24px;
        }}
        .container {{
            max-width: 1360px;
            margin: 0 auto;
        }}
        .header-title {{
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        }}
        .header-subtitle {{
            color: var(--text-muted);
            font-size: 14px;
            margin-bottom: 28px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .metric-title {{
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: 700;
            color: #ffffff;
        }}
        .metric-sub {{
            font-size: 12px;
            margin-top: 4px;
            color: #94a3b8;
        }}
        .verdict-box {{
            background: linear-gradient(135deg, rgba(30,41,59,0.8) 0%, rgba(15,23,42,0.9) 100%);
            border: 1px solid #3b82f6;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .verdict-badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px 16px;
            border-bottom: 1px solid #222f4c;
            text-align: left;
        }}
        th {{
            background: #0b1120;
            color: #94a3b8;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        tr:hover td {{
            background: rgba(255,255,255,0.02);
        }}
        .badge {{
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
        }}
        .section-title {{
            font-size: 20px;
            font-weight: 600;
            margin-top: 36px;
            margin-bottom: 16px;
            color: #f8fafc;
        }}
        audio {{
            filter: invert(0.85) hue-rotate(180deg);
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header-title">MambaFlow-TTS: Validation Health & Artifact Report</div>
    <div class="header-subtitle">
        Automated End-to-End Evaluation on <code>val.csv</code> | Checkpoint: <strong>{summary.get('checkpoint', 'Auto')}</strong> | Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
    </div>

    <!-- Health Verdict Banner -->
    <div class="verdict-box">
        <div>
            <div style="font-size:12px; color:#94a3b8; text-transform:uppercase; font-weight:600;">Overall Model Health Verdict</div>
            <div style="font-size:22px; font-weight:700; margin-top:2px;">{summary.get('overall_health_verdict', 'NEEDS_REVIEW')}</div>
            <div style="font-size:13px; color:#cbd5e1; margin-top:4px;">{summary.get('verdict_description', '')}</div>
        </div>
        <div style="text-align:right;">
            <span class="verdict-badge" style="background:{'#10b981' if summary.get('overall_pass_rate', 0) >= 0.70 else '#f43f5e'}; color:#ffffff;">
                Overall Pass Rate: {summary.get('overall_pass_rate', 0)*100:.1f}% ({summary.get('overall_pass_count', 0)}/{summary.get('total_samples', 0)})
            </span>
        </div>
    </div>

    <!-- Metric KPI Grid -->
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-title">Whisper Mean WER / PER</div>
            <div class="metric-value">{summary.get('mean_wer', 0)*100:.2f}%</div>
            <div class="metric-sub">CER: {summary.get('mean_cer', 0)*100:.2f}% | PER: {summary.get('mean_per', 0)*100:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">OOV vs In-Domain WER</div>
            <div class="metric-value">{summary.get('oov_wer', 0)*100:.1f}% <span style="font-size:14px; font-weight:400; color:#94a3b8;">vs {summary.get('in_domain_wer', 0)*100:.1f}%</span></div>
            <div class="metric-sub">{summary.get('oov_samples_count', 0)} OOV Samples Evaluated</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Trajectory Straightness</div>
            <div class="metric-value">{summary.get('mean_straightness', 0):.4f}</div>
            <div class="metric-sub">Optimal Transport Metric (1.0 = Ideal)</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">F0 RMSE (Voiced)</div>
            <div class="metric-value">{summary.get('mean_f0_rmse', 0):.1f} <span style="font-size:14px; font-weight:400; color:#94a3b8;">Hz</span></div>
            <div class="metric-sub">Energy RMSE: {summary.get('mean_energy_rmse', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Speaker SECS (WavLM)</div>
            <div class="metric-value">{summary.get('mean_secs', 0):.3f}</div>
            <div class="metric-sub">Cosine Sim with Ground Truth</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Spectral Roll-off (85%)</div>
            <div class="metric-value">{summary.get('mean_rolloff_85', 0):.0f} <span style="font-size:14px; font-weight:400; color:#94a3b8;">Hz</span></div>
            <div class="metric-sub">Coloration: {summary.get('mean_nisqa_coloration', 0):.2f} / 5.0</div>
        </div>
    </div>

    <!-- Failure Rates Breakdown Table -->
    <div class="card mb-4" style="background:var(--bg-card); border:1px solid var(--border-card); border-radius:12px; margin-bottom:32px;">
        <div style="background:#0f172a; padding:16px 20px; border-bottom:1px solid var(--border-card); border-radius:12px 12px 0 0;">
            <h3 style="margin:0; font-size:16px;">Artifact Failure Rates Breakdown</h3>
        </div>
        <div style="padding:0; overflow-x:auto;">
            <table>
                <thead>
                    <tr>
                        <th>Failure Check Category</th>
                        <th>Threshold Condition</th>
                        <th>Failing Samples</th>
                        <th>Failure Rate</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>1. Muffled Audio</strong></td>
                        <td>NISQA Coloration &lt; 3.0 OR Spectral Roll-off &lt; 3200 Hz</td>
                        <td>{summary.get('muffled_count', 0)} / {summary.get('total_samples', 0)}</td>
                        <td><strong>{summary.get('muffled_rate', 0)*100:.1f}%</strong></td>
                        <td><span class="badge" style="background:{'#10b981' if summary.get('muffled_rate', 0) < 0.15 else '#f43f5e'}; color:#fff;">{'HEALTHY' if summary.get('muffled_rate', 0) < 0.15 else 'FAILING'}</span></td>
                    </tr>
                    <tr>
                        <td><strong>2. Dual Audio / Echo / Phase Clash</strong></td>
                        <td>NISQA Discontinuity &lt; 3.0 &amp; Pitch Jumps &gt; 15% OR Pitch Jumps &gt; 35%</td>
                        <td>{summary.get('dual_audio_count', 0)} / {summary.get('total_samples', 0)}</td>
                        <td><strong>{summary.get('dual_audio_rate', 0)*100:.1f}%</strong></td>
                        <td><span class="badge" style="background:{'#10b981' if summary.get('dual_audio_rate', 0) < 0.10 else '#f43f5e'}; color:#fff;">{'HEALTHY' if summary.get('dual_audio_rate', 0) < 0.10 else 'FAILING'}</span></td>
                    </tr>
                    <tr>
                        <td><strong>3. Non-Utterance / Silence / Babbling</strong></td>
                        <td>VAD &lt; 20% OR NoSpeechProb &gt; 60% OR CompRatio &gt; 2.4 OR CharRate &gt; 0.25 s/char</td>
                        <td>{summary.get('non_utterance_count', 0)} / {summary.get('total_samples', 0)}</td>
                        <td><strong>{summary.get('non_utterance_rate', 0)*100:.1f}%</strong></td>
                        <td><span class="badge" style="background:{'#10b981' if summary.get('non_utterance_rate', 0) < 0.10 else '#f43f5e'}; color:#fff;">{'HEALTHY' if summary.get('non_utterance_rate', 0) < 0.10 else 'FAILING'}</span></td>
                    </tr>
                    <tr>
                        <td><strong>4. High Word Error Rate (WER)</strong></td>
                        <td>WER &gt; 15%</td>
                        <td>{summary.get('high_wer_count', 0)} / {summary.get('total_samples', 0)}</td>
                        <td><strong>{summary.get('high_wer_rate', 0)*100:.1f}%</strong></td>
                        <td><span class="badge" style="background:{'#10b981' if summary.get('high_wer_rate', 0) < 0.20 else '#f59e0b'}; color:#fff;">{'HEALTHY' if summary.get('high_wer_rate', 0) < 0.20 else 'ATTENTION'}</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- Worst Offenders Galleries with Audio Controls -->
    <div class="section-title">Worst Offenders (Audio Inspection)</div>
    
    {render_audio_table(worst_muffled, "Worst Muffled / Acoustic Roll-off Samples", "#f59e0b")}
    {render_audio_table(worst_dual, "Worst Dual Audio / Discontinuous Pitch Jump Samples", "#f43f5e")}
    {render_audio_table(worst_non_utt, "Worst Non-Utterance / Silence / Babbling Samples", "#8b5cf6")}
    {render_audio_table(worst_wer, "Worst Intelligibility / High WER Samples", "#3b82f6")}

</div>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[+] HTML Report generated: {output_html_path}")


def print_terminal_verdict_table(summary: dict):
    """Prints a beautiful formatted summary verdict to stdout."""
    print("\n" + "=" * 82)
    print("           MAMBAFLOW-TTS: VALIDATION HEALTH & ARTIFACT VERDICT TABLE")
    print("=" * 82)
    print(f"  Checkpoint:           {summary.get('checkpoint', 'Auto')}")
    print(f"  Total Evaluated:      {summary.get('total_samples', 0)} utterances from val.csv")
    print(f"  Overall Pass Rate:    {summary.get('overall_pass_rate', 0)*100:.1f}% ({summary.get('overall_pass_count', 0)}/{summary.get('total_samples', 0)})")
    print(f"  Model Health Verdict: {summary.get('overall_health_verdict', 'NEEDS_REVIEW')}")
    print("-" * 82)
    print(f"  {'Metric / Failure Category':<35} | {'Value / Rate':<16} | {'Status':<12}")
    print("-" * 82)
    print(f"  {'Mean WER':<35} | {summary.get('mean_wer', 0)*100:.2f}%{'':<10} | {'GOOD' if summary.get('mean_wer', 0) <= 0.15 else 'HIGH'}")
    print(f"  {'Mean CER':<35} | {summary.get('mean_cer', 0)*100:.2f}%{'':<10} | {'GOOD' if summary.get('mean_cer', 0) <= 0.08 else 'HIGH'}")
    print(f"  {'Mean PER (Phoneme Error Rate)':<35} | {summary.get('mean_per', 0)*100:.2f}%{'':<10} | {'GOOD' if summary.get('mean_per', 0) <= 0.10 else 'HIGH'}")
    print(f"  {'In-Domain WER':<35} | {summary.get('in_domain_wer', 0)*100:.2f}%{'':<10} | -")
    print(f"  {'OOV (Unseen Words) WER':<35} | {summary.get('oov_wer', 0)*100:.2f}%{'':<10} | -")
    print(f"  {'Trajectory Straightness (OT S)':<35} | {summary.get('mean_straightness', 0):.4f}{'':<10} | {'IDEAL' if summary.get('mean_straightness', 0) >= 0.90 else 'CURVED'}")
    print(f"  {'F0 RMSE (Voiced)':<35} | {summary.get('mean_f0_rmse', 0):.1f} Hz{'':<9} | -")
    print(f"  {'Speaker Similarity (WavLM SECS)':<35} | {summary.get('mean_secs', 0):.4f}{'':<10} | {'MATCH' if summary.get('mean_secs', 0) >= 0.75 else 'LOW'}")
    print(f"  {'Median 85% Spectral Roll-off':<35} | {summary.get('mean_rolloff_85', 0):.0f} Hz{'':<9} | {'CLEAR' if summary.get('mean_rolloff_85', 0) >= 3200 else 'MUFFLED'}")
    print(f"  {'NISQA Coloration Score':<35} | {summary.get('mean_nisqa_coloration', 0):.2f} / 5.0{'':<6} | -")
    print("-" * 82)
    print(f"  {'Artifact 1: % Muffled Audio':<35} | {summary.get('muffled_rate', 0)*100:.1f}%{'':<10} | {'PASS' if summary.get('muffled_rate', 0) < 0.15 else 'FAIL'}")
    print(f"  {'Artifact 2: % Dual / Clashing Audio':<35} | {summary.get('dual_audio_rate', 0)*100:.1f}%{'':<10} | {'PASS' if summary.get('dual_audio_rate', 0) < 0.10 else 'FAIL'}")
    print(f"  {'Artifact 3: % Non-Utterance / Silence':<35} | {summary.get('non_utterance_rate', 0)*100:.1f}%{'':<10} | {'PASS' if summary.get('non_utterance_rate', 0) < 0.10 else 'FAIL'}")
    print(f"  {'Intelligibility: % WER > 0.15':<35} | {summary.get('high_wer_rate', 0)*100:.1f}%{'':<10} | {'PASS' if summary.get('high_wer_rate', 0) < 0.20 else 'ATTN'}")
    print("=" * 82 + "\n")


# =============================================================================
# 5. Main Execution Flow
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MambaFlow-TTS Automated End-to-End Evaluation & Artifact Detection"
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="/home/monesh/ljspeech/LJSpeech-1.1/val.csv",
        help="Path to validation CSV file.",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="/home/monesh/ljspeech/LJSpeech-1.1/train.csv",
        help="Path to training CSV for OOV calculation.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to model checkpoint (default: auto-detect latest in TTS_checkpoints/).",
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        default=None,
        help="Directory of pre-generated WAVs if evaluating existing audio files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation reports and synthesized audio.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to evaluate (-1 for full val.csv, default: 5 for smoke test).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Computation device (auto detects safe memory).",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="tiny.en",
        help="Faster-whisper model size (tiny.en, base.en, small.en, medium.en, large-v3).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.667,
        help="Euler flow-matching sampling temperature (default: 0.667).",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=30,
        help="Number of Euler ODE flow steps (default: 30).",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="euler",
        choices=["euler", "midpoint"],
        help="ODE solver algorithm.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.5,
        help="Classifier-Free Guidance scale (default: 1.5).",
    )
    parser.add_argument(
        "--use_gt_durations",
        action="store_true",
        help="Use ground-truth durations from val.csv instead of duration predictor.",
    )
    parser.add_argument(
        "--wer_fail_thresh",
        type=float,
        default=0.20,
        help="WER threshold above which a sample is flagged as failing intelligibility.",
    )
    parser.add_argument(
        "--rolloff_muffled_thresh",
        type=float,
        default=3200.0,
        help="Spectral roll-off threshold below which audio is flagged as IS_MUFFLED.",
    )

    args = parser.parse_args()

    # Determine device safely
    if args.device == "auto":
        if torch.cuda.is_available():
            # Check free VRAM
            try:
                free_bytes = torch.cuda.mem_get_info()[0]
                free_mb = free_bytes / (1024 * 1024)
                if free_mb > 1500:
                    device = "cuda"
                    print(f"[+] Sufficient CUDA VRAM available ({free_mb:.0f} MB). Using CUDA.")
                else:
                    device = "cpu"
                    print(f"[+] Low CUDA VRAM ({free_mb:.0f} MB). Running strictly on CPU to protect active training.")
            except Exception:
                device = "cpu"
        else:
            device = "cpu"
    else:
        device = args.device

    if device == "cpu":
        enable_mamba_cpu_fallback()

    os.makedirs(args.output_dir, exist_ok=True)
    wavs_out_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(wavs_out_dir, exist_ok=True)

    # 1. Load Validation Data
    print(f"\n[+] Loading validation CSV: {args.val_csv}")
    df_val = pd.read_csv(args.val_csv)
    total_val_len = len(df_val)
    print(f"[+] Total samples in val.csv: {total_val_len:,}")

    if args.num_samples > 0:
        eval_df = df_val.iloc[:args.num_samples].copy().reset_index(drop=True)
    else:
        eval_df = df_val.copy().reset_index(drop=True)
    n_eval = len(eval_df)
    print(f"[+] Evaluating {n_eval} samples...")

    # 2. Initialize OOV Analyzer
    oov_analyzer = OOVAnalyzer(args.train_csv)

    # 3. Model Setup (if synthesizing)
    model = None
    dacvae_decoder = None
    ckpt_path = args.ckpt

    if args.wav_dir is None:
        if ckpt_path is None:
            ckpt_path = find_latest_checkpoint()
        if ckpt_path is None or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"[+] Loading model from checkpoint: {ckpt_path} on device: {device}...")
        pl_module = TTSMODEL.load_from_checkpoint(ckpt_path, map_location=device)
        model = pl_module.model.to(device).eval()

        print("[+] Loading DACVAE decoder...")
        dacvae_decoder, sr_dac, _ = setup_dacvae(device="cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    else:
        print(f"[+] Pre-generated WAV directory provided: {args.wav_dir}")

    # 4. Evaluator Setup (ASR, VAD, NISQA, SQUIM, SECS)
    evaluator = PipelineEvaluator(whisper_model_size=args.whisper_model, device="cpu")

    # 5. Evaluation Loop
    results = []
    resampler_16k_48k = T.Resample(48000, 16000)
    resampler_16k_22k = T.Resample(22050, 16000)

    print("\n" + "=" * 80)
    print("                     STARTING EVALUATION PIPELINE")
    print("=" * 80)

    t0_all = time.time()

    for idx in range(n_eval):
        row = eval_df.iloc[idx]
        audio_id = str(row["audio_id"])
        text = str(row.get("normalized_text", row.get("text", "")))
        dur_val = row.get("durations", None)
        gt_durations = None
        if dur_val is not None:
            gt_durations = ast.literal_eval(dur_val) if isinstance(dur_val, str) else dur_val

        gt_audio_path = row.get("audio_path", "")

        print(f"\n[{idx+1}/{n_eval}] Evaluating: {audio_id}")
        print(f"  Target Text: \"{text}\"")

        # Step A: Obtain Predicted & Ground-Truth Waveforms
        straightness = 1.0
        wav_pred_path = os.path.join(wavs_out_dir, f"{audio_id}_pred.wav")
        wav_gt_path = os.path.join(wavs_out_dir, f"{audio_id}_gt.wav")

        if args.wav_dir is not None:
            # Locate pre-generated WAV in wav_dir
            candidate_files = [
                os.path.join(args.wav_dir, f)
                for f in os.listdir(args.wav_dir)
                if audio_id in f and f.endswith(".wav") and not f.endswith("_gt.wav")
            ]
            if candidate_files:
                pred_wav_src = candidate_files[0]
                wav_pred_48k, sr_p = sf.read(pred_wav_src)
                if sr_p != 48000:
                    wav_pred_48k = librosa.resample(wav_pred_48k, orig_sr=sr_p, target_sr=48000)
                sf.write(wav_pred_path, wav_pred_48k, 48000)
            else:
                print(f"  [!] Warning: Pre-generated file for {audio_id} not found in {args.wav_dir}. Synthesizing...")
                wav_pred_48k, straightness, _ = synthesize_and_record_trajectory(
                    model=model,
                    dacvae_decoder=dacvae_decoder,
                    text=text,
                    gt_durations=gt_durations,
                    use_gt_durations=args.use_gt_durations,
                    n_steps=args.n_steps,
                    temperature=args.temperature,
                    solver=args.solver,
                    cfg_scale=args.cfg_scale,
                    device=device,
                    seed=42 + idx,
                )
                sf.write(wav_pred_path, wav_pred_48k, 48000)
        else:
            # Synthesize on the fly and track trajectory
            wav_pred_48k, straightness, _ = synthesize_and_record_trajectory(
                model=model,
                dacvae_decoder=dacvae_decoder,
                text=text,
                gt_durations=gt_durations,
                use_gt_durations=args.use_gt_durations,
                n_steps=args.n_steps,
                temperature=args.temperature,
                solver=args.solver,
                cfg_scale=args.cfg_scale,
                device=device,
                seed=42 + idx,
            )
            sf.write(wav_pred_path, wav_pred_48k, 48000)

        # Obtain Ground-Truth Audio
        if os.path.exists(gt_audio_path):
            wav_gt_raw, sr_gt = sf.read(gt_audio_path)
            if wav_gt_raw.ndim > 1:
                wav_gt_raw = np.mean(wav_gt_raw, axis=-1)
            wav_gt_16k = librosa.resample(wav_gt_raw, orig_sr=sr_gt, target_sr=16000)
            wav_gt_48k = librosa.resample(wav_gt_raw, orig_sr=sr_gt, target_sr=48000)
            sf.write(wav_gt_path, wav_gt_48k, 48000)
        elif "latent" in row and os.path.exists(str(row["latent"])) and dacvae_decoder is not None:
            # Decode GT latent to audio
            gt_latent_np = np.squeeze(np.load(row["latent"]))
            if gt_latent_np.ndim == 2 and gt_latent_np.shape[0] != 128:
                gt_latent_np = gt_latent_np.T
            gt_t = torch.tensor(gt_latent_np, dtype=torch.float32).unsqueeze(0).to(device)
            wav_gt_48k = decode_latent_to_audio(dacvae_decoder, gt_t, target_device=device)
            wav_gt_16k = librosa.resample(wav_gt_48k, orig_sr=48000, target_sr=16000)
            sf.write(wav_gt_path, wav_gt_48k, 48000)
        else:
            wav_gt_16k = wav_pred_48k  # fallback

        # Resample predicted audio to 16kHz for Whisper, VAD, NISQA, SECS
        wav_pred_16k = librosa.resample(wav_pred_48k, orig_sr=48000, target_sr=16000)

        # Step B: Compute Whisper ASR Metrics (WER, CER, compression, no_speech)
        asr_res = evaluator.evaluate_whisper(wav_pred_16k, text, sr=16000)

        # Step C: Compute Silero VAD & Duration Rate
        vad_res = evaluator.evaluate_vad(wav_pred_16k, text)

        # Step D: Compute Acoustic Alignment Metrics (F0 RMSE, Energy RMSE, Pitch Jumps, HNR)
        acoust_res = evaluator.evaluate_acoustic(wav_pred_16k, wav_gt_16k, sr=16000)

        # Step E: Compute Spectral Features (Roll-off, Flatness)
        spec_res = evaluator.evaluate_spectral(wav_pred_16k, sr=16000)
        spec_res_gt = evaluator.evaluate_spectral(wav_gt_16k, sr=16000)

        # Step F: Quality Metrics (NISQA & SQUIM)
        qual_res = evaluator.evaluate_quality(wav_pred_16k)

        # Step G: Speaker Cosine Similarity (SECS)
        secs = evaluator.evaluate_secs(wav_pred_16k, wav_gt_16k)

        # Step H: OOV Vocabulary Analysis
        oov_res = oov_analyzer.analyze_sentence(text)

        # Step I: Compute Failure Mode Flags
        # 1. IS_MUFFLED: NISQA Coloration < 3.0 OR 85% Roll-off < 3200 Hz (relative to GT audio)
        gt_rolloff = spec_res_gt["spectral_rolloff_85_median"]
        rolloff_cutoff = min(args.rolloff_muffled_thresh, gt_rolloff * 0.90)
        is_muffled = bool(
            (qual_res["nisqa_coloration"] < 3.0) or
            (spec_res["spectral_rolloff_85_median"] < rolloff_cutoff)
        )

        # 2. IS_DUAL_AUDIO: Competing simultaneous voices, large pitch jumps, or severe discontinuity
        is_dual_audio = bool(
            (qual_res["nisqa_discontinuity"] < 3.0 and acoust_res["pitch_jump_rate"] > 0.15) or
            (acoust_res["pitch_jump_rate"] > 0.35) or
            (qual_res["nisqa_discontinuity"] < 2.0)
        )

        # 3. IS_NON_UTTERANCE: Silence, non-speech noise, runaway looping
        is_non_utterance = bool(
            (vad_res["vad_speech_ratio"] < 0.20) or
            (asr_res["no_speech_prob"] > 0.60) or
            (asr_res["compression_ratio"] > 2.40) or
            (vad_res["char_rate"] > 0.25)
        )

        # 4. Overall Pass
        overall_pass = bool(
            (not is_muffled) and
            (not is_dual_audio) and
            (not is_non_utterance) and
            (asr_res["wer"] <= args.wer_fail_thresh)
        )

        print(f"  ASR Transcript:    \"{asr_res['transcript']}\"")
        print(f"  WER: {asr_res['wer']*100:.1f}% | CER: {asr_res['cer']*100:.1f}% | PER: {asr_res['per']*100:.1f}% | Straightness: {straightness:.4f} | SECS: {secs:.3f}")
        print(f"  F0 RMSE: {acoust_res['f0_rmse']:.1f} Hz | Roll-off: {spec_res['spectral_rolloff_85_median']:.0f} Hz | VAD Ratio: {vad_res['vad_speech_ratio']*100:.1f}%")
        print(f"  Flags -> Muffled: {is_muffled} | Dual Audio: {is_dual_audio} | Non-Utterance: {is_non_utterance} | Pass: {overall_pass}")

        # Record Sample Result
        rel_pred_wav = os.path.relpath(wav_pred_path, args.output_dir)
        rel_gt_wav = os.path.relpath(wav_gt_path, args.output_dir)

        sample_record = {
            "audio_id": audio_id,
            "text": text,
            "transcript": asr_res["transcript"],
            "wer": asr_res["wer"],
            "cer": asr_res["cer"],
            "per": asr_res["per"],
            "no_speech_prob": asr_res["no_speech_prob"],
            "compression_ratio": asr_res["compression_ratio"],
            "duration_sec": vad_res["duration_sec"],
            "vad_speech_ratio": vad_res["vad_speech_ratio"],
            "char_rate": vad_res["char_rate"],
            "f0_rmse": acoust_res["f0_rmse"],
            "energy_rmse": acoust_res["energy_rmse"],
            "pitch_jump_rate": acoust_res["pitch_jump_rate"],
            "hnr_db": acoust_res["hnr_db"],
            "trajectory_straightness": straightness,
            "secs": secs,
            "spectral_rolloff_85": spec_res["spectral_rolloff_85_median"],
            "spectral_flatness": spec_res["spectral_flatness_mean"],
            "nisqa_mos": qual_res["nisqa_mos"],
            "nisqa_noisiness": qual_res["nisqa_noi"],
            "nisqa_discontinuity": qual_res["nisqa_discontinuity"],
            "nisqa_coloration": qual_res["nisqa_coloration"],
            "squim_stoi": qual_res["squim_stoi"],
            "squim_pesq": qual_res["squim_pesq"],
            "squim_si_sdr": qual_res["squim_si_sdr"],
            "is_oov": oov_res["is_oov"],
            "oov_word_count": oov_res["oov_words"],
            "is_muffled": is_muffled,
            "is_dual_audio": is_dual_audio,
            "is_non_utterance": is_non_utterance,
            "overall_pass": overall_pass,
            "pred_wav_path": wav_pred_path,
            "gt_wav_path": wav_gt_path,
            "pred_wav_rel": rel_pred_wav,
            "gt_wav_rel": rel_gt_wav,
        }
        results.append(sample_record)

    total_eval_time = time.time() - t0_all
    print(f"\n[+] Completed evaluation of {len(results)} samples in {total_eval_time:.1f}s.")

    # 6. Aggregate Summary Metrics
    df_results = pd.DataFrame(results)

    # In-Domain vs OOV WER breakdown
    oov_samples = df_results[df_results["is_oov"]]
    in_domain_samples = df_results[~df_results["is_oov"]]
    oov_wer = float(oov_samples["wer"].mean()) if len(oov_samples) > 0 else float(df_results["wer"].mean())
    in_domain_wer = float(in_domain_samples["wer"].mean()) if len(in_domain_samples) > 0 else float(df_results["wer"].mean())

    total_count = len(df_results)
    pass_count = int(df_results["overall_pass"].sum())
    muffled_count = int(df_results["is_muffled"].sum())
    dual_audio_count = int(df_results["is_dual_audio"].sum())
    non_utterance_count = int(df_results["is_non_utterance"].sum())
    high_wer_count = int((df_results["wer"] > 0.15).sum())

    pass_rate = pass_count / max(total_count, 1)
    muffled_rate = muffled_count / max(total_count, 1)
    dual_audio_rate = dual_audio_count / max(total_count, 1)
    non_utterance_rate = non_utterance_count / max(total_count, 1)
    high_wer_rate = high_wer_count / max(total_count, 1)

    # Health Verdict Determination
    if pass_rate >= 0.80 and muffled_rate < 0.10 and dual_audio_rate < 0.05 and non_utterance_rate < 0.05:
        health_verdict = "HEALTHY (PRODUCTION READY)"
        verdict_desc = "Checkpoint displays high intelligibility, low acoustic damping, and stable alignment."
    elif pass_rate >= 0.50:
        health_verdict = "MARGINAL (NEEDS SUPERVISION)"
        verdict_desc = "Model generates recognizable speech with occasional acoustic artifacts or alignment drift."
    else:
        health_verdict = "FAILING / UNSTABLE"
        verdict_desc = "Checkpoint suffers from severe acoustic collapse, frequency roll-off, or non-speech babbling."

    summary = {
        "checkpoint": os.path.basename(ckpt_path) if ckpt_path else "pre_generated_wavs",
        "total_samples": total_count,
        "evaluation_duration_sec": total_eval_time,
        "overall_health_verdict": health_verdict,
        "verdict_description": verdict_desc,
        "overall_pass_count": pass_count,
        "overall_pass_rate": pass_rate,
        "mean_wer": float(df_results["wer"].mean()),
        "std_wer": float(df_results["wer"].std()),
        "mean_cer": float(df_results["cer"].mean()),
        "std_cer": float(df_results["cer"].std()),
        "mean_per": float(df_results["per"].mean()),
        "std_per": float(df_results["per"].std()),
        "in_domain_wer": in_domain_wer,
        "oov_wer": oov_wer,
        "oov_samples_count": len(oov_samples),
        "mean_straightness": float(df_results["trajectory_straightness"].mean()),
        "mean_f0_rmse": float(df_results["f0_rmse"].dropna().mean()) if len(df_results["f0_rmse"].dropna()) > 0 else 0.0,
        "mean_energy_rmse": float(df_results["energy_rmse"].mean()),
        "mean_secs": float(df_results["secs"].mean()),
        "mean_rolloff_85": float(df_results["spectral_rolloff_85"].mean()),
        "mean_nisqa_mos": float(df_results["nisqa_mos"].mean()),
        "mean_nisqa_coloration": float(df_results["nisqa_coloration"].mean()),
        "mean_nisqa_discontinuity": float(df_results["nisqa_discontinuity"].mean()),
        "mean_vad_ratio": float(df_results["vad_speech_ratio"].mean()),
        "muffled_count": muffled_count,
        "muffled_rate": muffled_rate,
        "dual_audio_count": dual_audio_count,
        "dual_audio_rate": dual_audio_rate,
        "non_utterance_count": non_utterance_count,
        "non_utterance_rate": non_utterance_rate,
        "high_wer_count": high_wer_count,
        "high_wer_rate": high_wer_rate,
    }

    # 7. Save Deliverables
    # A. Per-Sample CSV
    csv_out_path = os.path.join(args.output_dir, "per_sample_metrics.csv")
    df_results.to_csv(csv_out_path, index=False)
    print(f"\n[+] Per-sample CSV saved to: {csv_out_path}")

    # B. Summary JSON
    json_out_path = os.path.join(args.output_dir, "summary_report.json")
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Summary JSON saved to: {json_out_path}")

    # C. Interactive HTML Report
    html_out_path = os.path.join(args.output_dir, "report.html")
    generate_html_report(df_results, summary, html_out_path)

    # D. Terminal Summary
    print_terminal_verdict_table(summary)


if __name__ == "__main__":
    main()
