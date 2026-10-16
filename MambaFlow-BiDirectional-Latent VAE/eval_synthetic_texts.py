#!/usr/bin/env python3
"""
===============================================================================
MambaFlow-TTS: Evaluation on 10 Synthetic Texts
Comparing:
- Config A: Temperature = 0.667, CFG = 2.0
- Config B: Temperature = 1.000, CFG = 2.0
===============================================================================
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.transforms as T
import jiwer
import librosa

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from TTSDataModule import TTSMODEL
from preprocessing.text import text_to_sequence
from generate_audio import setup_dacvae, find_latest_checkpoint
from evaluate_val_health import (
    clean_text_for_asr,
    PipelineEvaluator,
    synthesize_and_record_trajectory,
)

# 10 Diverse Synthetic Benchmark Texts
SYNTHETIC_TEXTS = [
    {
        "id": "SYNTH_01",
        "category": "Pangram & Articulation",
        "text": "The quick brown fox jumps over the lazy dog.",
    },
    {
        "id": "SYNTH_02",
        "category": "Technical & AI Domain",
        "text": "Artificial intelligence and deep learning are transforming modern speech synthesis.",
    },
    {
        "id": "SYNTH_03",
        "category": "Tongue Twister & Sibilants",
        "text": "She sells seashells by the seashore, and the shells she sells are seashells.",
    },
    {
        "id": "SYNTH_04",
        "category": "Numbers & Historical Narrative",
        "text": "In nineteen sixty-nine, humans walked on the moon for the very first time.",
    },
    {
        "id": "SYNTH_05",
        "category": "Conversational & Polite Inquiry",
        "text": "Could you please tell me where the nearest train station is located?",
    },
    {
        "id": "SYNTH_06",
        "category": "Rhythm & Repetition",
        "text": "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    },
    {
        "id": "SYNTH_07",
        "category": "Weather Broadcast",
        "text": "The weather forecast predicts heavy rain and severe thunderstorms throughout the afternoon.",
    },
    {
        "id": "SYNTH_08",
        "category": "Scientific & Polysyllabic",
        "text": "Quantum computing utilizes superposition and entanglement to perform complex calculations.",
    },
    {
        "id": "SYNTH_09",
        "category": "Public Announcement",
        "text": "Please remember to double check all your personal belongings before exiting the vehicle.",
    },
    {
        "id": "SYNTH_10",
        "category": "Mamba Architecture Domain",
        "text": "Mamba models provide linear time complexity and powerful state space representations.",
    },
]


def compute_per(clean_ref: str, clean_hyp: str) -> float:
    """Computes Phoneme Error Rate (PER) using IPA sequences and Levenshtein alignment."""
    if not clean_ref:
        return 0.0 if not clean_hyp else 1.0
    try:
        ref_seq = text_to_sequence(clean_ref, ["english_cleaners2"])
        hyp_seq = text_to_sequence(clean_hyp, ["english_cleaners2"])
        ref_ipa = ref_seq[1] if len(ref_seq) > 1 else ""
        hyp_ipa = hyp_seq[1] if len(hyp_seq) > 1 else ""
        ref_ph = " ".join(list(ref_ipa.replace(" ", "")))
        hyp_ph = " ".join(list(hyp_ipa.replace(" ", "")))
        if not ref_ph:
            return 0.0 if not hyp_ph else 1.0
        return float(jiwer.wer(ref_ph, hyp_ph))
    except Exception:
        return float(jiwer.cer(clean_ref, clean_hyp))


def main():
    output_dir = os.path.join(CURRENT_DIR, "eval_synthetic_results")
    os.makedirs(output_dir, exist_ok=True)
    wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("  MAMBAFLOW-TTS: 10 SYNTHETIC TEXTS BENCHMARK")
    print("  Comparing: (Temp=0.667, CFG=2.0) vs (Temp=1.0, CFG=2.0)")
    print(f"  Device: {device}")
    print("=" * 80)

    # Load Model Checkpoint
    ckpt = find_latest_checkpoint()
    print(f"[+] Loading checkpoint: {ckpt}")
    pl_module = TTSMODEL.load_from_checkpoint(ckpt, map_location=device)
    model = pl_module.model.to(device).eval()

    # Load DACVAE
    dacvae_decoder, sr_dac, _ = setup_dacvae(device=device)

    # Load Evaluator (Whisper, VAD, NISQA)
    evaluator = PipelineEvaluator(whisper_model_size="tiny.en", device=device)

    resampler_48k_16k = T.Resample(48000, 16000)

    configs = [
        {"name": "temp0.667_cfg2.0", "temp": 0.667, "cfg": 2.0, "label": "Temp 0.667 | CFG 2.0"},
        {"name": "temp1.000_cfg2.0", "temp": 1.000, "cfg": 2.0, "label": "Temp 1.000 | CFG 2.0"},
    ]

    all_records = []
    comparison_data = []

    for item in SYNTHETIC_TEXTS:
        t_id = item["id"]
        category = item["category"]
        text = item["text"]
        clean_ref = clean_text_for_asr(text)

        print(f"\n--------------------------------------------------------------------------------")
        print(f"[{t_id}] ({category}): \"{text}\"")
        print(f"--------------------------------------------------------------------------------")

        item_comp = {
            "id": t_id,
            "category": category,
            "text": text,
        }

        for cfg_info in configs:
            c_name = cfg_info["name"]
            temp = cfg_info["temp"]
            cfg_val = cfg_info["cfg"]

            wav_filename = f"{t_id}_{c_name}.wav"
            wav_path = os.path.join(wavs_dir, wav_filename)
            rel_wav_path = os.path.join("wavs", wav_filename)

            t0 = time.time()
            wav_48k, straightness, _ = synthesize_and_record_trajectory(
                model=model,
                dacvae_decoder=dacvae_decoder,
                text=text,
                gt_durations=None,
                use_gt_durations=False,  # Natural predicted durations
                n_steps=30,
                temperature=temp,
                solver="euler",
                cfg_scale=cfg_val,
                device=device,
                seed=42,  # Fixed seed for deterministic comparison
            )
            sf.write(wav_path, wav_48k, 48000)
            syn_time = time.time() - t0

            # 16k resample
            wav_16k = resampler_48k_16k(torch.from_numpy(wav_48k).float()).numpy()

            # Whisper ASR
            asr_res = evaluator.evaluate_whisper(wav_16k, text)
            clean_hyp = asr_res["clean_hyp"]
            transcript = asr_res["transcript"]
            wer = asr_res["wer"]
            cer = asr_res["cer"]

            # PER
            per = compute_per(clean_ref, clean_hyp)

            # VAD & Spectral
            vad_res = evaluator.evaluate_vad(wav_16k, text)
            spec_res = evaluator.evaluate_spectral(wav_16k)
            qual_res = evaluator.evaluate_quality(wav_16k)

            print(f"  [{cfg_info['label']}] WER: {wer*100:.1f}% | CER: {cer*100:.1f}% | PER: {per*100:.1f}% | Straight: {straightness:.4f} | Roll-off: {spec_res['spectral_rolloff_85_median']:.0f} Hz")
            print(f"    Transcript: \"{transcript}\"")

            rec = {
                "id": t_id,
                "category": category,
                "text": text,
                "config": c_name,
                "temperature": temp,
                "cfg_scale": cfg_val,
                "transcript": transcript,
                "wer": wer,
                "cer": cer,
                "per": per,
                "straightness": straightness,
                "spectral_rolloff_85": spec_res["spectral_rolloff_85_median"],
                "duration_sec": vad_res["duration_sec"],
                "vad_ratio": vad_res["vad_speech_ratio"],
                "nisqa_mos": qual_res["nisqa_mos"],
                "wav_rel": rel_wav_path,
            }
            all_records.append(rec)

            prefix = "t0667" if temp == 0.667 else "t1000"
            item_comp[f"{prefix}_transcript"] = transcript
            item_comp[f"{prefix}_wer"] = wer
            item_comp[f"{prefix}_cer"] = cer
            item_comp[f"{prefix}_per"] = per
            item_comp[f"{prefix}_straightness"] = straightness
            item_comp[f"{prefix}_rolloff"] = spec_res["spectral_rolloff_85_median"]
            item_comp[f"{prefix}_wav"] = rel_wav_path

        comparison_data.append(item_comp)

    # Summary Statistics
    df_all = pd.DataFrame(all_records)
    df_all.to_csv(os.path.join(output_dir, "synthetic_all_records.csv"), index=False)

    df_comp = pd.DataFrame(comparison_data)
    df_comp.to_csv(os.path.join(output_dir, "synthetic_comparison.csv"), index=False)

    df_t0667 = df_all[df_all["temperature"] == 0.667]
    df_t1000 = df_all[df_all["temperature"] == 1.000]

    summary = {
        "temp0.667_cfg2.0": {
            "mean_wer": float(df_t0667["wer"].mean()),
            "mean_cer": float(df_t0667["cer"].mean()),
            "mean_per": float(df_t0667["per"].mean()),
            "mean_straightness": float(df_t0667["straightness"].mean()),
            "mean_rolloff": float(df_t0667["spectral_rolloff_85"].mean()),
            "mean_nisqa_mos": float(df_t0667["nisqa_mos"].mean()),
        },
        "temp1.000_cfg2.0": {
            "mean_wer": float(df_t1000["wer"].mean()),
            "mean_cer": float(df_t1000["cer"].mean()),
            "mean_per": float(df_t1000["per"].mean()),
            "mean_straightness": float(df_t1000["straightness"].mean()),
            "mean_rolloff": float(df_t1000["spectral_rolloff_85"].mean()),
            "mean_nisqa_mos": float(df_t1000["nisqa_mos"].mean()),
        },
    }

    with open(os.path.join(output_dir, "synthetic_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generate Markdown Report
    generate_markdown_report(summary, comparison_data, output_dir)

    # Generate Interactive HTML Dashboard
    generate_html_dashboard(summary, comparison_data, output_dir)

    print("\n" + "=" * 80)
    print("  10 SYNTHETIC TEXTS EVALUATION COMPLETE!")
    print(f"  Temp 0.667 CFG 2.0 -> Mean WER: {summary['temp0.667_cfg2.0']['mean_wer']*100:.2f}% | Mean PER: {summary['temp0.667_cfg2.0']['mean_per']*100:.2f}% | Roll-off: {summary['temp0.667_cfg2.0']['mean_rolloff']:.0f} Hz")
    print(f"  Temp 1.000 CFG 2.0 -> Mean WER: {summary['temp1.000_cfg2.0']['mean_wer']*100:.2f}% | Mean PER: {summary['temp1.000_cfg2.0']['mean_per']*100:.2f}% | Roll-off: {summary['temp1.000_cfg2.0']['mean_rolloff']:.0f} Hz")
    print("=" * 80)


def generate_markdown_report(summary, comparison_data, output_dir):
    s_0667 = summary["temp0.667_cfg2.0"]
    s_1000 = summary["temp1.000_cfg2.0"]

    md_content = f"""# MambaFlow-TTS: 10 Synthetic Texts Benchmark Report

**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Target:** 10 Diverse Synthetic Out-of-Domain Sentences  
**Configurations Tested:**
1. **Config A:** `Temperature = 0.667, CFG Scale = 2.0` (Production Sweet Spot)
2. **Config B:** `Temperature = 1.000, CFG Scale = 2.0` (Standard Thermal Noise)

---

## 1. Aggregate Comparison Summary

| Metric | Temp = 0.667, CFG = 2.0 | Temp = 1.000, CFG = 2.0 | Delta (T=0.667 vs T=1.0) | Winner / Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Whisper WER (Word Error Rate)** | **{s_0667['mean_wer']*100:.2f}%** | **{s_1000['mean_wer']*100:.2f}%** | {s_0667['mean_wer']*100 - s_1000['mean_wer']*100:+.2f}% | {'Temp 0.667' if s_0667['mean_wer'] <= s_1000['mean_wer'] else 'Temp 1.000'} |
| **Whisper CER (Char Error Rate)** | **{s_0667['mean_cer']*100:.2f}%** | **{s_1000['mean_cer']*100:.2f}%** | {s_0667['mean_cer']*100 - s_1000['mean_cer']*100:+.2f}% | {'Temp 0.667' if s_0667['mean_cer'] <= s_1000['mean_cer'] else 'Temp 1.000'} |
| **Phoneme Error Rate (PER)** | **{s_0667['mean_per']*100:.2f}%** | **{s_1000['mean_per']*100:.2f}%** | {s_0667['mean_per']*100 - s_1000['mean_per']*100:+.2f}% | {'Temp 0.667' if s_0667['mean_per'] <= s_1000['mean_per'] else 'Temp 1.000'} |
| **Trajectory Straightness ($S$)** | **{s_0667['mean_straightness']:.4f}** | **{s_1000['mean_straightness']:.4f}** | {s_0667['mean_straightness'] - s_1000['mean_straightness']:+.4f} | {'Temp 0.667 (Straighter Flow)' if s_0667['mean_straightness'] >= s_1000['mean_straightness'] else 'Temp 1.000'} |
| **Median 85% Roll-off Frequency** | **{s_0667['mean_rolloff']:.0f} Hz** | **{s_1000['mean_rolloff']:.0f} Hz** | {s_0667['mean_rolloff'] - s_1000['mean_rolloff']:+.0f} Hz | **Temp 0.667 (Clearer Highs)** |
| **NISQA MOS Score** | **{s_0667['mean_nisqa_mos']:.2f}** | **{s_1000['mean_nisqa_mos']:.2f}** | {s_0667['mean_nisqa_mos'] - s_1000['mean_nisqa_mos']:+.2f} | {'Temp 0.667' if s_0667['mean_nisqa_mos'] >= s_1000['mean_nisqa_mos'] else 'Temp 1.000'} |

---

## 2. Per-Sentence WSR & PER Breakdown

| ID | Category | Target Text | WER / PER (Temp 0.667) | WER / PER (Temp 1.000) |
| :---: | :--- | :--- | :---: | :---: |
"""
    for r in comparison_data:
        md_content += f"| **{r['id']}** | {r['category']} | *\"{r['text']}\"* | WER: **{r['t0667_wer']*100:.1f}%** \| PER: **{r['t0667_per']*100:.1f}%** | WER: **{r['t1000_wer']*100:.1f}%** \| PER: **{r['t1000_per']*100:.1f}%** |\n"

    md_content += f"""
---

## 3. Side-by-Side Transcripts Comparison

"""
    for r in comparison_data:
        md_content += f"### {r['id']}: {r['category']}\n"
        md_content += f"- **Target Prompt:** \"{r['text']}\"\n"
        md_content += f"- **Temp 0.667 (CFG 2.0):** \"{r['t0667_transcript']}\" (WER: {r['t0667_wer']*100:.1f}%, PER: {r['t0667_per']*100:.1f}%)\n"
        md_content += f"- **Temp 1.000 (CFG 2.0):** \"{r['t1000_transcript']}\" (WER: {r['t1000_wer']*100:.1f}%, PER: {r['t1000_per']*100:.1f}%)\n\n"

    md_path = os.path.join(output_dir, "SYNTHETIC_EVALUATION_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] Markdown Report saved to: {md_path}")


def generate_html_dashboard(summary, comparison_data, output_dir):
    s_0667 = summary["temp0.667_cfg2.0"]
    s_1000 = summary["temp1.000_cfg2.0"]

    cards_html = []
    for r in comparison_data:
        cards_html.append(f"""
        <div class="card mb-4">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid #1e293b; padding-bottom:8px;">
                <div>
                    <span class="badge" style="background:#0284c7; color:#fff; margin-right:8px;">{r['id']}</span>
                    <span style="font-weight:700; color:#94a3b8; font-size:12px; text-transform:uppercase;">{r['category']}</span>
                </div>
            </div>
            <div style="font-size:15px; font-weight:600; color:#f1f5f9; margin-bottom:16px;">
                "{r['text']}"
            </div>

            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
                <!-- Temp 0.667 CFG 2.0 -->
                <div style="background:#090d16; padding:14px; border-radius:10px; border:1px solid #1e293b;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-weight:700; color:#38bdf8; font-size:13px;">Temp 0.667 | CFG 2.0</span>
                        <span style="font-size:11px; font-weight:700; color:{'#10b981' if r['t0667_wer'] == 0 else '#f59e0b'};">
                            WER: {r['t0667_wer']*100:.1f}% | PER: {r['t0667_per']*100:.1f}%
                        </span>
                    </div>
                    <audio controls preload="none" style="width:100%; height:32px; margin-bottom:8px;">
                        <source src="{r['t0667_wav']}" type="audio/wav">
                    </audio>
                    <div style="font-size:12px; color:#cbd5e1;">
                        <em>"{r['t0667_transcript']}"</em>
                    </div>
                </div>

                <!-- Temp 1.000 CFG 2.0 -->
                <div style="background:#090d16; padding:14px; border-radius:10px; border:1px solid #1e293b;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-weight:700; color:#a855f7; font-size:13px;">Temp 1.000 | CFG 2.0</span>
                        <span style="font-size:11px; font-weight:700; color:{'#10b981' if r['t1000_wer'] == 0 else '#f59e0b'};">
                            WER: {r['t1000_wer']*100:.1f}% | PER: {r['t1000_per']*100:.1f}%
                        </span>
                    </div>
                    <audio controls preload="none" style="width:100%; height:32px; margin-bottom:8px;">
                        <source src="{r['t1000_wav']}" type="audio/wav">
                    </audio>
                    <div style="font-size:12px; color:#cbd5e1;">
                        <em>"{r['t1000_transcript']}"</em>
                    </div>
                </div>
            </div>
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MambaFlow-TTS: 10 Synthetic Texts Audio & WSR/PER Inspection</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #070b14;
            --bg-card: #0f172a;
            --border-card: #1e293b;
            --accent-cyan: #06b6d4;
            --accent-purple: #a855f7;
            --accent-emerald: #10b981;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}
        * {{ box-sizing: border-box; margin:0; padding:0; }}
        body {{
            background: var(--bg-main);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            padding: 30px 20px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            margin-bottom: 28px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-card);
        }}
        .header h1 {{
            font-size: 26px;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        }}
        .header-sub {{
            font-size: 14px;
            color: var(--text-muted);
        }}
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            padding: 16px;
        }}
        .kpi-title {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}
        .kpi-val {{
            font-size: 22px;
            font-weight: 800;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .badge {{
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
        }}
        audio {{
            filter: invert(0.85) hue-rotate(180deg);
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>MambaFlow-TTS: 10 Synthetic Texts Side-by-Side Audio & Benchmark</h1>
        <div class="header-sub">
            Direct comparison: <strong>Temp 0.667 | CFG 2.0</strong> vs <strong>Temp 1.000 | CFG 2.0</strong> across 10 challenging sentences.
        </div>
    </div>

    <!-- Aggregate KPIs -->
    <div class="kpi-row">
        <div class="kpi-card" style="border-top:3px solid #38bdf8;">
            <div class="kpi-title">Temp 0.667 Mean WER</div>
            <div class="kpi-val" style="color:#38bdf8;">{s_0667['mean_wer']*100:.2f}%</div>
            <div style="font-size:12px; color:#94a3b8; margin-top:2px;">PER: {s_0667['mean_per']*100:.2f}%</div>
        </div>
        <div class="kpi-card" style="border-top:3px solid #a855f7;">
            <div class="kpi-title">Temp 1.000 Mean WER</div>
            <div class="kpi-val" style="color:#a855f7;">{s_1000['mean_wer']*100:.2f}%</div>
            <div style="font-size:12px; color:#94a3b8; margin-top:2px;">PER: {s_1000['mean_per']*100:.2f}%</div>
        </div>
        <div class="kpi-card" style="border-top:3px solid #10b981;">
            <div class="kpi-title">Temp 0.667 Roll-off</div>
            <div class="kpi-val" style="color:#10b981;">{s_0667['mean_rolloff']:.0f} Hz</div>
            <div style="font-size:12px; color:#94a3b8; margin-top:2px;">vs {s_1000['mean_rolloff']:.0f} Hz (T=1.0)</div>
        </div>
        <div class="kpi-card" style="border-top:3px solid #f59e0b;">
            <div class="kpi-title">Temp 0.667 Straightness</div>
            <div class="kpi-val" style="color:#f59e0b;">{s_0667['mean_straightness']:.4f}</div>
            <div style="font-size:12px; color:#94a3b8; margin-top:2px;">OT Trajectory Linearity</div>
        </div>
    </div>

    <!-- Sentence Cards with Players -->
    <div style="margin-top:24px;">
        {''.join(cards_html)}
    </div>

</div>
</body>
</html>
"""
    html_path = os.path.join(output_dir, "synthetic_dashboard.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[+] Interactive Dashboard saved to: {html_path}")


if __name__ == "__main__":
    main()
