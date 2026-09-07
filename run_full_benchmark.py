import os
import sys
import subprocess
import time
import json
import numpy as np
import pandas as pd

MODELS = [
    {
        "id": "sequential_tetra",
        "name": "Tetra Sequential (Ours)",
        "dir": "MambaFlow-Sequential-Tetra",
        "ckpt": "Checkpoints/MambaFlow-Sequential-Tetra-epoch=132-val_loss=0.3295.ckpt",
        "output": "benchmark_results/sequential_tetra.json",
    },
    {
        "id": "staircase_xt",
        "name": "Staircase-XTEncoder",
        "dir": "MambaFlow-Staircase-XTEncoder",
        "ckpt": "Checkpoints/MambaFlow-Staircase-XTEncoder-epoch=145-val_loss=0.5805.ckpt",
        "output": "benchmark_results/staircase_xt.json",
    },
    {
        "id": "staircase_mamba2",
        "name": "Staircase-Mamba2",
        "dir": "MambaFlow-Staircase-Mamba2",
        "ckpt": "Checkpoints/MambaFlow-Staircase-Mamba2-epoch=127-val_loss=0.5824.ckpt",
        "output": "benchmark_results/staircase_mamba2.json",
    },
    {
        "id": "twostage_mamba2",
        "name": "TwoStage-Mamba2",
        "dir": "MambaFlow-TwoStage-Mamba2",
        "ckpt": "Checkpoints/MambaFlow-TwoStage-Mamba2-epoch=169-val_loss=0.5837.ckpt",
        "output": "benchmark_results/twostage_mamba2.json",
    },
    {
        "id": "unet_onebottleneck",
        "name": "UNet-OneBottleneck",
        "dir": "MambaFlow-UNet-OneBottleneck",
        "ckpt": "Checkpoints/MambaFlow-UNet-OneBottleneck-last_decoder_only.ckpt",
        "output": "benchmark_results/unet_onebottleneck.json",
    },
]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Master Multi-Model Benchmark Orchestrator")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to evaluate (default: all 1310)")
    parser.add_argument("--n_timesteps", type=int, default=30, help="Flow matching steps")
    parser.add_argument("--temperature", type=float, default=0.667, help="ODE sampling temperature")
    parser.add_argument("--solver", type=str, default="euler", help="ODE solver (euler/midpoint)")
    args = parser.parse_args()

    os.makedirs("benchmark_results", exist_ok=True)

    print("=" * 80)
    print("      MAMBAFLOW-TTS COMPREHENSIVE MULTI-MODEL BENCHMARK SUITE")
    print(f"      Samples: {args.max_samples if args.max_samples is not None else 'ALL (1,310)'} | Steps: {args.n_timesteps} | Solver: {args.solver}")
    print("=" * 80)

    start_total_time = time.perf_counter()

    for idx, m in enumerate(MODELS, 1):
        print(f"\n[{idx}/5] LAUNCHING BENCHMARK: {m['name']} ({m['id']})")
        print(f"    Directory:  {m['dir']}")
        print(f"    Checkpoint: {m['ckpt']}")
        print(f"    Output:     {m['output']}")

        cmd = [
            sys.executable,
            "benchmark_engine.py",
            "--model_id", m["id"],
            "--model_dir", m["dir"],
            "--ckpt", m["ckpt"],
            "--output", m["output"],
            "--n_timesteps", str(args.n_timesteps),
            "--temperature", str(args.temperature),
            "--solver", args.solver,
        ]
        if args.max_samples is not None:
            cmd.extend(["--max_samples", str(args.max_samples)])

        t0 = time.perf_counter()
        res = subprocess.run(cmd)
        t1 = time.perf_counter()

        if res.returncode != 0:
            print(f"ERROR: Benchmark for {m['id']} failed with exit code {res.returncode}!")
            sys.exit(res.returncode)

        print(f"    Completed in {t1 - t0:.1f} seconds.\n")

    # -------------------------------------------------------------------------
    # Compile Master Comparison Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("             COMPILING MASTER COMPARATIVE REPORT")
    print("=" * 80)

    master_summary = {}

    for m in MODELS:
        with open(m["output"], "r") as f:
            data = json.load(f)
            master_summary[m["id"]] = {
                "name": m["name"],
                "params": data["params"],
                "metrics": data["metrics"],
                "n_samples": data["n_samples"],
                "time_sec": data["total_benchmark_time_sec"],
            }

    master_path = "benchmark_results/master_summary.json"
    with open(master_path, "w") as f:
        json.dump(master_summary, f, indent=2)
    print(f"Saved master summary to: {master_path}")

    # Build Comparative Table
    table_rows = []
    for m in MODELS:
        s = master_summary[m["id"]]
        p = s["params"]
        met = s["metrics"]
        row = {
            "Model Architecture": s["name"],
            "Total Params (M)": f"{p['total_params'] / 1e6:.2f}M",
            "Dec Params (M)": f"{p['decoder_params'] / 1e6:.2f}M",
            "Ckpt (MB)": f"{p['ckpt_size_mb']:.1f}",
            "Mel MAE (raw)": f"{met['mae_raw']['mean']:.3f} ± {met['mae_raw']['std']:.3f}",
            "Mel RMSE (raw)": f"{met['rmse_raw']['mean']:.3f} ± {met['rmse_raw']['std']:.3f}",
            "Cosine Sim": f"{met['cos_sim']['mean']:.4f}",
            "Pearson (r)": f"{met['pearson_r']['mean']:.4f}",
            "Spectral Conv": f"{met['spectral_conv']['mean']:.4f}",
            "Low-Freq MAE": f"{met['mae_low']['mean']:.3f}",
            "Mid-Freq MAE": f"{met['mae_mid']['mean']:.3f}",
            "High-Freq MAE": f"{met['mae_high']['mean']:.3f}",
            "DTW Dist": f"{met['dtw_dist_norm']['mean']:.3f}",
            "Dur Err (fr)": f"{met['dur_err_frames']['mean']:.1f}",
            "RTF": f"{met['rtf']['mean']:.4f}",
            "Speed (xRT)": f"{1.0 / met['rtf']['mean']:.1f}x",
        }
        table_rows.append(row)

    df_comp = pd.DataFrame(table_rows)
    csv_path = "benchmark_results/master_comparison_table.csv"
    df_comp.to_csv(csv_path, index=False)

    print("\n" + df_comp.to_string(index=False))
    print(f"\nSaved master comparison table CSV to: {csv_path}")
    print(f"Total benchmark run time: {(time.perf_counter() - start_total_time)/60:.2f} minutes.")

if __name__ == "__main__":
    main()
