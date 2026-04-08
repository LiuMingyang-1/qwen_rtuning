#!/usr/bin/env bash
# run_base_eval_and_icr.sh
#
# One-shot script to:
#   1. Pull latest code from git
#   2. Run eval.py on base model (no adapter) → outputs/eval/unsure-id-base/
#   3. Delete stale (wrong-label) icr_scores_base.jsonl
#   4. Re-collect base ICR scores from correct base predictions
#
# Usage (from /root/autodl-tmp/qwen_rtuning):
#   bash scripts/run_base_eval_and_icr.sh [--batch_size N] [--dry_run]
#
# Estimated time on 32GB card:
#   eval.py  (12 989 samples, batch=8): ~1–2 h
#   collect  (12 787 samples, eager):   ~3–5 h
# Run in tmux/screen to avoid disconnection.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config (edit if your paths differ)
# ---------------------------------------------------------------------------
PYTHON=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/qwen_rtuning
MODEL=/root/autodl-tmp/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
DATA_ROOT=/root/autodl-tmp/qwen_rtuning/R-Tuning-data

EVAL_OUT=${ROOT}/outputs/eval/unsure-id-base
ICR_OUT=${ROOT}/icr_analysis/outputs/icr_scores_base.jsonl

BATCH_SIZE=8    # safe for 32GB; reduce to 4 if you hit OOM during eval
DRY_RUN=0

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
for arg in "$@"; do
  case $arg in
    --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
    --batch_size)   shift; BATCH_SIZE="$1" ;;
    --dry_run)      DRY_RUN=1 ;;
  esac
done

run() {
  echo ""
  echo ">>> $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    "$@"
  fi
}

cd "$ROOT"

# ---------------------------------------------------------------------------
# Step 0: pull latest code
# ---------------------------------------------------------------------------
echo "========================================="
echo "Step 0: git pull"
echo "========================================="
run git pull

# ---------------------------------------------------------------------------
# Step 1: base model eval
# ---------------------------------------------------------------------------
echo ""
echo "========================================="
echo "Step 1: eval.py (base model, batch=${BATCH_SIZE})"
echo "========================================="

if [[ -f "${EVAL_OUT}/predictions.jsonl" ]]; then
  echo "[SKIP] ${EVAL_OUT}/predictions.jsonl already exists — delete it to re-run eval."
else
  run $PYTHON eval.py \
      --model_name_or_path "$MODEL" \
      --data_root "$DATA_ROOT" \
      --output_dir "$EVAL_OUT" \
      --tasks pararel hotpotqa \
      --prompt_domain ID \
      --split test \
      --bf16 \
      --batch_size "$BATCH_SIZE"

  echo ""
  echo "--- eval metrics ---"
  cat "${EVAL_OUT}/metrics.json" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Step 2: delete stale base ICR scores (wrong labels from rtuning predictions)
# ---------------------------------------------------------------------------
echo ""
echo "========================================="
echo "Step 2: remove stale icr_scores_base.jsonl"
echo "========================================="

if [[ -f "$ICR_OUT" ]]; then
  echo "Removing stale file: $ICR_OUT"
  run rm "$ICR_OUT"
else
  echo "[OK] No stale file found."
fi

# ---------------------------------------------------------------------------
# Step 3: collect base ICR scores
# ---------------------------------------------------------------------------
echo ""
echo "========================================="
echo "Step 3: collect_icr_scores.py (base model)"
echo "========================================="

run $PYTHON icr_analysis/collect_icr_scores.py \
    --model_name_or_path "$MODEL" \
    --predictions_path "${EVAL_OUT}/predictions.jsonl" \
    --output_path "$ICR_OUT" \
    --model_tag base \
    --dtype bfloat16

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "========================================="
echo "All done."
echo "  Base predictions : ${EVAL_OUT}/predictions.jsonl"
echo "  Base ICR scores  : ${ICR_OUT}"
echo ""
echo "Next: download both ICR score files to local machine, then run:"
echo "  python icr_analysis/four_way_eval.py \\"
echo "      --icr_path_base    icr_analysis/outputs/icr_scores_base.jsonl \\"
echo "      --icr_path_rtuning icr_analysis/outputs/icr_scores_rtuning.jsonl \\"
echo "      --output_dir       icr_analysis/outputs/figures"
echo "========================================="
