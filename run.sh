#!/bin/bash
set -e

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
DATA_ROOT=${DATA_ROOT:-R-Tuning-data}
METHOD=${METHOD:-unsure}
OUTPUT_ROOT=${OUTPUT_ROOT:-../outputs}

DATA_PATH="${OUTPUT_ROOT}/data/${METHOD}.jsonl"
CKPT_PATH="${OUTPUT_ROOT}/checkpoints/qwen2.5-rtuning-${METHOD}"

echo "=== Step 1: Build dataset (method=${METHOD}) ==="
python3 build_dataset.py \
  --model_name_or_path "${MODEL}" \
  --data_root "${DATA_ROOT}" \
  --tasks pararel mmlu fever hotpotqa wice \
  --method "${METHOD}" \
  --output_path "${DATA_PATH}" \
  --load_in_4bit

echo "=== Step 2: Train ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 train.py \
  --model_name_or_path "${MODEL}" \
  --dataset_path "${DATA_PATH}" \
  --output_dir "${CKPT_PATH}" \
  --load_in_4bit --bf16 \
  --max_length 4096 \
  --skip_truncated \
  --num_train_epochs 3

echo "=== Done. Checkpoint saved to ${CKPT_PATH} ==="
