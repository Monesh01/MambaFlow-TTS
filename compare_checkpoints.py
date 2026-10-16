#!/usr/bin/env python3
import os
import sys
import time
import json
import ast
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import librosa

ROOT_DIR = "/home/monesh/TTSModel"
LATENT_DIR = os.path.join(ROOT_DIR, "MambaFlow-BiDirectional-Latent VAE")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if LATENT_DIR not in sys.path:
    sys.path.insert(0, LATENT_DIR)

from TTSDataModule import TTSMODEL
from generate_audio import setup_dacvae
from evaluate_val_health import PipelineEvaluator, OOVAnalyzer, synthesize_and_record_trajectory

def evaluate_checkpoint(ckpt_path, df_eval, evaluator, oov_analyzer, dacvae_decoder, device="cuda", n_samples=50):
    print(f"\n=======================================================")
    print(f"EVALUATING CHECKPOINT: {os.path.basename(ckpt_path)}")
    print(f"=======================================================")
    
    pl_module = TTSMODEL.load_from_checkpoint(ckpt_path, map_location=device)
    model = pl_module.model.to(device).eval()
    
    resampler_48k_16k = T.Resample(48000, 16000)
    resampler_22k_16k = T.Resample(22050, 16000)
    
    results = []
    t_start_all = time.time()
    
    for idx in range(n_samples):
        row = df_eval.iloc[idx]
        audio_id = str(row["audio_id"])
        text = str(row.get("normalized_text", row.get("text", "")))
        dur_val = row.get("durations", None)
        gt_durations = ast.literal_eval(dur_val) if isinstance(dur_val, str) else dur_val
        gt_audio_path = str(row.get("audio_path", ""))
        
        # Load GT 16k
        wav_gt_16k = None
        f0_gt = None
        rms_gt = None
        gt_emb = None
        
        if os.path.exists(gt_audio_path):
            wav_gt_raw, sr_gt = sf.read(gt_audio_path)
            if wav_gt_raw.ndim > 1:
                wav_gt_raw = np.mean(wav_gt_raw, axis=-1)
            tensor_gt = torch.from_numpy(wav_gt_raw).float()
            if sr_gt == 48000:
                wav_gt_16k = resampler_48k_16k(tensor_gt).numpy()
            elif sr_gt == 22050:
                wav_gt_16k = resampler_22k_16k(tensor_gt).numpy()
            else:
                wav_gt_16k = librosa.resample(wav_gt_raw, orig_sr=sr_gt, target_sr=16000)
            
            hop = 256
            f0_gt, _, _ = librosa.pyin(wav_gt_16k, fmin=50, fmax=500, sr=16000, hop_length=hop)
            rms_gt = librosa.feature.rms(y=wav_gt_16k, frame_length=1024, hop_length=hop)[0]
            
            if evaluator.wavlm_model is not None and evaluator.wavlm_extractor is not None:
                try:
                    inputs_gt = evaluator.wavlm_extractor(wav_gt_16k, sampling_rate=16000, return_tensors="pt")
                    with torch.no_grad():
                        gt_emb = evaluator.wavlm_model(**inputs_gt).embeddings
                except Exception:
                    gt_emb = None
        
        # Synthesize with temp=0.667, cfg=2.0 (same as benchmark table)
        t0 = time.time()
        wav_pred_48k, straightness, _ = synthesize_and_record_trajectory(
            model=model,
            dacvae_decoder=dacvae_decoder,
            text=text,
            gt_durations=gt_durations,
            use_gt_durations=False,
            n_steps=30,
            temperature=0.667,
            solver="euler",
            cfg_scale=2.0,
            device=device,
            seed=42 + idx,
        )
        t_syn = time.time() - t0
        audio_dur = len(wav_pred_48k) / 48000.0
        rtf = t_syn / max(audio_dur, 1e-4)
        
        # Resample to 16k
        tensor_pred = torch.from_numpy(wav_pred_48k).float()
        wav_pred_16k = resampler_48k_16k(tensor_pred).numpy()
        
        # Whisper WER, CER, PER
        asr_res = evaluator.evaluate_whisper(wav_pred_16k, text)
        
        # F0 RMSE
        f0_rmse = float("nan")
        if f0_gt is not None and rms_gt is not None:
            hop = 256
            f0_pred, _, _ = librosa.pyin(wav_pred_16k, fmin=50, fmax=500, sr=16000, hop_length=hop)
            rms_pred = librosa.feature.rms(y=wav_pred_16k, frame_length=1024, hop_length=hop)[0]
            try:
                X = rms_pred.reshape(-1, 1)
                Y = rms_gt.reshape(-1, 1)
                D, wp = librosa.sequence.dtw(X=X.T, Y=Y.T, metric="cosine")
                aligned_f0_pred = f0_pred[wp[:, 0]]
                aligned_f0_gt = f0_gt[wp[:, 1]]
            except Exception:
                min_len = min(len(f0_pred), len(f0_gt))
                aligned_f0_pred = f0_pred[:min_len]
                aligned_f0_gt = f0_gt[:min_len]
            voiced_mask = (~np.isnan(aligned_f0_pred)) & (~np.isnan(aligned_f0_gt)) & (aligned_f0_pred > 0) & (aligned_f0_gt > 0)
            if np.sum(voiced_mask) > 5:
                f0_rmse = float(np.sqrt(np.mean((aligned_f0_pred[voiced_mask] - aligned_f0_gt[voiced_mask]) ** 2)))
        
        # SECS
        secs = 0.5
        if evaluator.wavlm_model is not None and gt_emb is not None:
            try:
                inputs_pred = evaluator.wavlm_extractor(wav_pred_16k, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    emb_pred = evaluator.wavlm_model(**inputs_pred).embeddings
                    secs = float(F.cosine_similarity(emb_pred, gt_emb).item())
            except Exception:
                secs = 0.5
                
        # NISQA
        qual_res = evaluator.evaluate_quality(wav_pred_16k)
        
        results.append({
            "audio_id": audio_id,
            "wer": asr_res["wer"],
            "cer": asr_res["cer"],
            "per": asr_res["per"],
            "straightness": straightness,
            "f0_rmse": f0_rmse,
            "secs": secs,
            "nisqa_mos": qual_res["nisqa_mos"],
            "rtf": rtf,
            "speed_xrt": 1.0 / max(rtf, 1e-4),
        })
        
        if (idx + 1) % 10 == 0 or idx == n_samples - 1:
            print(f"[{idx+1:02d}/{n_samples}] WER: {asr_res['wer']*100:.1f}% | PER: {asr_res['per']*100:.1f}% | S: {straightness:.4f} | MOS: {qual_res['nisqa_mos']:.2f}")

    del model, pl_module
    torch.cuda.empty_cache()

    df_res = pd.DataFrame(results)
    summary = {
        "checkpoint": os.path.basename(ckpt_path),
        "mean_wer": float(df_res["wer"].mean()),
        "mean_cer": float(df_res["cer"].mean()),
        "mean_per": float(df_res["per"].mean()),
        "mean_straightness": float(df_res["straightness"].mean()),
        "mean_f0_rmse": float(df_res["f0_rmse"].dropna().mean()),
        "mean_secs": float(df_res["secs"].mean()),
        "mean_nisqa_mos": float(df_res["nisqa_mos"].mean()),
        "mean_rtf": float(df_res["rtf"].mean()),
        "mean_speed_xrt": float(df_res["speed_xrt"].mean()),
    }
    return summary

def main():
    val_csv = "/home/monesh/ljspeech/LJSpeech-1.1/val.csv"
    train_csv = "/home/monesh/ljspeech/LJSpeech-1.1/train.csv"
    df_val = pd.read_csv(val_csv)
    df_eval = df_val.iloc[:50].copy().reset_index(drop=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Using device: {device}")
    
    oov_analyzer = OOVAnalyzer(train_csv)
    evaluator = PipelineEvaluator(whisper_model_size="tiny.en", device="cpu")
    dacvae_decoder, _, _ = setup_dacvae(device=device)
    
    ckpts = [
        os.path.join(LATENT_DIR, "TTS_checkpoints/mambaflow_ca_bimamba_448d_cfg15_epoch_221_loss_0.8917.ckpt"),
        os.path.join(LATENT_DIR, "TTS_checkpoints/mambaflow_ca_bimamba_448d_cfg15_epoch_279_loss_0.8891.ckpt"),
        os.path.join(LATENT_DIR, "TTS_checkpoints/mambaflow_ca_bimamba_448d_cfg15_last.ckpt"),
    ]
    
    all_res = []
    for c in ckpts:
        if os.path.exists(c):
            s = evaluate_checkpoint(c, df_eval, evaluator, oov_analyzer, dacvae_decoder, device=device, n_samples=50)
            all_res.append(s)
            
    print("\n" + "=" * 90)
    print("                      HEAD-TO-HEAD CHECKPOINT COMPARISON")
    print("=" * 90)
    print(f"{'Checkpoint':<40} | {'WER (%)':<8} | {'PER (%)':<8} | {'MOS':<6} | {'SECS':<6} | {'Straight':<8} | {'F0 (Hz)':<8}")
    print("-" * 90)
    for s in all_res:
        print(f"{s['checkpoint']:<40} | {s['mean_wer']*100:<8.2f} | {s['mean_per']*100:<8.2f} | {s['mean_nisqa_mos']:<6.2f} | {s['mean_secs']:<6.3f} | {s['mean_straightness']:<8.4f} | {s['mean_f0_rmse']:<8.1f}")
    print("=" * 90)
    
    with open("compare_448d_results.json", "w") as f:
        json.dump(all_res, f, indent=2)

if __name__ == "__main__":
    main()
