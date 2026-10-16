#!/usr/bin/env bash
# =============================================================================
# MambaFlow-TTS Evaluation Runner
# =============================================================================
# Usage:
#   ./run_eval.sh --smoke               # Run 5-sample smoke test
#   ./run_eval.sh --samples 20          # Evaluate 20 samples
#   ./run_eval.sh --full                # Evaluate all samples in val.csv
#   ./run_eval.sh --wav_dir <path>      # Evaluate pre-generated WAVs
#   ./run_eval.sh --ckpt <path>         # Evaluate a specific checkpoint
# =============================================================================

set -e

ENV_PYTHON="/home/monesh/miniconda3/envs/myenv/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_val_health.py"

NUM_SAMPLES=5
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --smoke)
      NUM_SAMPLES=5
      shift
      ;;
    --full)
      NUM_SAMPLES=-1
      shift
      ;;
    --samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "================================================================================"
echo " Starting MambaFlow-TTS Validation Health & Artifact Pipeline"
echo " Python:      $ENV_PYTHON"
echo " Script:      $EVAL_SCRIPT"
echo " Samples:     $NUM_SAMPLES"
echo "================================================================================"

"$ENV_PYTHON" "$EVAL_SCRIPT" --num_samples "$NUM_SAMPLES" "${EXTRA_ARGS[@]}"
