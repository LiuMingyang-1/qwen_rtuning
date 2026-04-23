#!/usr/bin/env bash
# run_new_baselines.sh
#
# One-shot runner for the two new rejection baselines:
#   1. uncertainty-threshold reject
#   2. 4-sample consistency reject
#   3. optional compare/aggregation step
#
# Usage:
#   bash scripts/run_new_baselines.sh --mode smoke
#   bash scripts/run_new_baselines.sh --mode full
#   bash scripts/run_new_baselines.sh --mode compare
#   bash scripts/run_new_baselines.sh --mode full --with_compare
#
# Notes:
# - Intended to run inside screen/tmux.
# - Smoke and full runs are intentionally separated.
# - Compare step requires scikit-learn to be installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Config (edit if your paths differ)
# ---------------------------------------------------------------------------
PYTHON=${PYTHON:-python3}
ROOT=${ROOT:-${REPO_ROOT}}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
DATA_ROOT=${DATA_ROOT:-${ROOT}/R-Tuning-data}

TASKS=(${TASKS:-pararel hotpotqa})
PROMPT_DOMAIN=${PROMPT_DOMAIN:-ID}

SMOKE_LIMIT=${SMOKE_LIMIT:-20}
NUM_SAMPLES=${NUM_SAMPLES:-4}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}

MODE=full
WITH_COMPARE=0
DRY_RUN=0
FORCE_RERUN=0
ALLOW_DEBUG_COMPARE=0

UNCERTAINTY_OUT_SMOKE=${ROOT}/outputs/eval/baselines/uncertainty-smoke
CONSISTENCY_OUT_SMOKE=${ROOT}/outputs/eval/baselines/consistency-smoke
UNCERTAINTY_OUT_FULL=${ROOT}/outputs/eval/baselines/uncertainty
CONSISTENCY_OUT_FULL=${ROOT}/outputs/eval/baselines/consistency
COMPARE_OUT=${ROOT}/outputs/eval/baselines/comparison

BASE_PRED_PATH=${ROOT}/outputs/eval/unsure-id-base/predictions.jsonl
RTUNING_PRED_PATH=${ROOT}/outputs/eval/unsure-id-rtuning/predictions.jsonl
ICR_BASE_PATH=${ROOT}/icr_analysis/outputs/outputs/icr_scores_base.jsonl
ICR_RTUNING_PATH=${ROOT}/icr_analysis/outputs/outputs/icr_scores_rtuning.jsonl

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#*=}"
      shift
      ;;
    --with_compare)
      WITH_COMPARE=1
      shift
      ;;
    --dry_run)
      DRY_RUN=1
      shift
      ;;
    --force_rerun)
      FORCE_RERUN=1
      shift
      ;;
    --allow_debug_compare)
      ALLOW_DEBUG_COMPARE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "smoke" && "$MODE" != "full" && "$MODE" != "compare" ]]; then
  echo "Invalid --mode: ${MODE}. Expected one of: smoke, full, compare" >&2
  exit 1
fi

run() {
  echo ""
  echo ">>> $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    "$@"
  fi
}

ensure_parent_dir() {
  local path="$1"
  mkdir -p "$(dirname "$path")"
}

maybe_remove_dir() {
  local dir="$1"
  if [[ $FORCE_RERUN -eq 1 && -d "$dir" ]]; then
    echo "Removing existing directory: $dir"
    run rm -rf "$dir"
  fi
}

run_uncertainty() {
  local output_dir="$1"
  local limit_arg=()
  local label="$2"

  if [[ "$label" == "smoke" ]]; then
    limit_arg=(--limit_per_task "$SMOKE_LIMIT")
  fi

  echo ""
  echo "========================================="
  echo "Step: uncertainty baseline (${label})"
  echo "========================================="

  maybe_remove_dir "$output_dir"
  if [[ -f "${output_dir}/predictions.jsonl" ]]; then
    echo "[SKIP] ${output_dir}/predictions.jsonl already exists."
    return
  fi

  run "$PYTHON" "${ROOT}/baseline_reject_eval.py" \
    --model_name_or_path "$MODEL" \
    --data_root "$DATA_ROOT" \
    --tasks "${TASKS[@]}" \
    --prompt_domain "$PROMPT_DOMAIN" \
    --baseline uncertainty \
    "${limit_arg[@]}" \
    --output_dir "$output_dir" \
    --load_in_4bit \
    --bf16
}

run_consistency() {
  local output_dir="$1"
  local label="$2"
  local limit_arg=()

  if [[ "$label" == "smoke" ]]; then
    limit_arg=(--limit_per_task "$SMOKE_LIMIT")
  fi

  echo ""
  echo "========================================="
  echo "Step: consistency baseline (${label})"
  echo "========================================="

  maybe_remove_dir "$output_dir"
  if [[ -f "${output_dir}/predictions.jsonl" ]]; then
    echo "[SKIP] ${output_dir}/predictions.jsonl already exists."
    return
  fi

  run "$PYTHON" "${ROOT}/baseline_reject_eval.py" \
    --model_name_or_path "$MODEL" \
    --data_root "$DATA_ROOT" \
    --tasks "${TASKS[@]}" \
    --prompt_domain "$PROMPT_DOMAIN" \
    --baseline consistency \
    --num_samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    "${limit_arg[@]}" \
    --output_dir "$output_dir" \
    --load_in_4bit \
    --bf16
}

run_compare() {
  local uncertainty_path="$1"
  local consistency_path="$2"
  local compare_args=()

  echo ""
  echo "========================================="
  echo "Step: compare baselines"
  echo "========================================="

  maybe_remove_dir "$COMPARE_OUT"
  if [[ -f "${COMPARE_OUT}/comparison_results.json" ]]; then
    echo "[SKIP] ${COMPARE_OUT}/comparison_results.json already exists."
    return
  fi

  if [[ $ALLOW_DEBUG_COMPARE -eq 1 ]]; then
    compare_args+=(--allow_probe_train_on_all_common_for_debug)
  fi

  if [[ ${#compare_args[@]} -gt 0 ]]; then
    run "$PYTHON" "${ROOT}/icr_analysis/compare_baselines.py" \
      --base_predictions_path "$BASE_PRED_PATH" \
      --rtuning_predictions_path "$RTUNING_PRED_PATH" \
      --icr_base_path "$ICR_BASE_PATH" \
      --icr_rtuning_path "$ICR_RTUNING_PATH" \
      --uncertainty_predictions_path "$uncertainty_path" \
      --consistency_predictions_path "$consistency_path" \
      --tasks "${TASKS[@]}" \
      "${compare_args[@]}" \
      --output_dir "$COMPARE_OUT"
  else
    run "$PYTHON" "${ROOT}/icr_analysis/compare_baselines.py" \
      --base_predictions_path "$BASE_PRED_PATH" \
      --rtuning_predictions_path "$RTUNING_PRED_PATH" \
      --icr_base_path "$ICR_BASE_PATH" \
      --icr_rtuning_path "$ICR_RTUNING_PATH" \
      --uncertainty_predictions_path "$uncertainty_path" \
      --consistency_predictions_path "$consistency_path" \
      --tasks "${TASKS[@]}" \
      --output_dir "$COMPARE_OUT"
  fi
}

cd "$ROOT"

echo "========================================="
echo "run_new_baselines.sh"
echo "ROOT          : $ROOT"
echo "MODEL         : $MODEL"
echo "DATA_ROOT     : $DATA_ROOT"
echo "TASKS         : ${TASKS[*]}"
echo "PROMPT_DOMAIN : $PROMPT_DOMAIN"
echo "MODE          : $MODE"
echo "WITH_COMPARE  : $WITH_COMPARE"
echo "FORCE_RERUN   : $FORCE_RERUN"
echo "DRY_RUN       : $DRY_RUN"
echo "========================================="

case "$MODE" in
  smoke)
    run_uncertainty "$UNCERTAINTY_OUT_SMOKE" "smoke"
    run_consistency "$CONSISTENCY_OUT_SMOKE" "smoke"
    ;;
  full)
    run_uncertainty "$UNCERTAINTY_OUT_FULL" "full"
    run_consistency "$CONSISTENCY_OUT_FULL" "full"
    if [[ $WITH_COMPARE -eq 1 ]]; then
      run_compare "${UNCERTAINTY_OUT_FULL}/predictions.jsonl" "${CONSISTENCY_OUT_FULL}/predictions.jsonl"
    fi
    ;;
  compare)
    run_compare "${UNCERTAINTY_OUT_FULL}/predictions.jsonl" "${CONSISTENCY_OUT_FULL}/predictions.jsonl"
    ;;
esac

echo ""
echo "========================================="
echo "Done."
echo "Smoke uncertainty : ${UNCERTAINTY_OUT_SMOKE}"
echo "Smoke consistency : ${CONSISTENCY_OUT_SMOKE}"
echo "Full uncertainty  : ${UNCERTAINTY_OUT_FULL}"
echo "Full consistency  : ${CONSISTENCY_OUT_FULL}"
echo "Compare output    : ${COMPARE_OUT}"
echo "========================================="
