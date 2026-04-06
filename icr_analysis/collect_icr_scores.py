#!/usr/bin/env python3
"""Collect ICR scores for R-Tuning predictions.

Reads predictions.jsonl produced by eval.py, runs a single forward pass
per sample (prompt + prediction text), computes per-layer ICR scores, and
saves the results to an output JSONL file.

Requires GPU with attn_implementation='eager' (flash_attention_2 / sdpa do
not support output_attentions=True).

Example
-------
python icr_analysis/collect_icr_scores.py \
    --model_name_or_path /root/autodl-tmp/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/... \
    --adapter_path /root/autodl-tmp/outputs/checkpoints/qwen2.5-rtuning-unsure \
    --predictions_path outputs/eval/unsure-id-rtuning/predictions.jsonl \
    --output_path icr_analysis/outputs/icr_scores.jsonl \
    --task pararel          # optional: filter to one task
    --dtype bfloat16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch

# Resolve paths ---------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent          # qwen_rtuning/
_REPO_ROOT = _PROJECT_ROOT.parent     # parent of qwen_rtuning (e.g. /root/autodl-tmp)

# qwen_rtuning package lives one level above the project root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# icr_analysis/ is on the path so we can import icr_score directly
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from icr_score import ICRScore  # noqa: E402  (vendored copy in icr_analysis/)
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from qwen_rtuning.chat_utils import render_chat_text  # noqa: E402


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect ICR scores for R-Tuning predictions.")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--predictions_path", required=True, help="Path to predictions.jsonl from eval.py")
    p.add_argument("--output_path", required=True, help="Output JSONL path for ICR scores")
    p.add_argument("--model_tag", default=None,
                   help="Short tag stored in each output row, e.g. 'base' or 'rtuning'. "
                        "Defaults to 'base' if no adapter_path, else 'rtuning'.")
    p.add_argument("--task", default=None, help="Filter to a specific task (e.g. pararel, hotpotqa)")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--max_response_tokens", type=int, default=64,
                   help="Truncate prediction to this many tokens (keeps memory bounded)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", default="eager",
                   help="Must be 'eager' for output_attentions=True to work")
    # ICR hyperparameters (defaults match the paper)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--top_p", type=float, default=0.1)
    p.add_argument("--pooling", default="mean", choices=["mean", "max", "min"])
    p.add_argument("--use_induction_head", action="store_true")
    p.add_argument("--skew_threshold", type=float, default=0)
    p.add_argument("--entropy_threshold", type=float, default=1e5)
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def derive_label(is_correct_strict: bool, is_refusal: bool) -> str:
    """Four-class label derived from R-Tuning eval fields."""
    if is_correct_strict and not is_refusal:
        return "correct_confident"
    if is_correct_strict and is_refusal:
        return "false_refusal"
    if not is_correct_strict and is_refusal:
        return "correct_refusal"
    return "hallucination"


def collect_stepwise_cache(
    model: Any,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    device: str,
) -> Tuple[List[Any], List[Any]]:
    """Single forward pass over prompt+response, then reshape into the
    step-by-step format that ICRScore expects.  Equivalent to token-by-token
    KV-cache decoding (causal mask guarantees identical values) but much faster.
    """
    prompt_len = prompt_ids.numel()
    full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(
            input_ids=full_ids,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states_steps: List[Any] = []
    attentions_steps: List[Any] = []

    # Step 0: prompt prefix
    hs_prompt = tuple(h[:, :prompt_len, :] for h in out.hidden_states)
    attn_prompt = tuple(a[:, :, :prompt_len, :prompt_len] for a in out.attentions)
    hidden_states_steps.append(hs_prompt)
    attentions_steps.append(attn_prompt)

    # Steps 1..N: one per response token
    for i in range(response_ids.numel()):
        pos = prompt_len + i
        hs_step = tuple(h[:, pos : pos + 1, :] for h in out.hidden_states)
        attn_step = tuple(a[:, :, pos : pos + 1, : pos + 1] for a in out.attentions)
        hidden_states_steps.append(hs_step)
        attentions_steps.append(attn_step)

    return hidden_states_steps, attentions_steps


def load_done_ids(output_path: Path) -> set:
    """Return set of sample IDs already written (for resume support)."""
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_tag = args.model_tag or ("base" if not args.adapter_path else "rtuning")

    # Resume: skip already-processed samples (match by id + model_tag)
    done_ids = load_done_ids(output_path)
    if done_ids:
        print(f"Resuming — {len(done_ids)} samples already done, skipping them.")

    # Load tokenizer
    tokenizer_source = args.adapter_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    dtype = dtype_from_name(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=None,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(args.device)

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    # Load predictions
    predictions_path = Path(args.predictions_path)
    records = [json.loads(l) for l in predictions_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Filter by task if requested
    if args.task:
        records = [r for r in records if r.get("task") == args.task]
        print(f"Filtered to task='{args.task}': {len(records)} samples")

    # Apply index slicing
    records = records[args.start_index:]
    if args.max_samples is not None:
        records = records[: args.max_samples]

    written = skipped = 0

    with output_path.open("a", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            sample_id = rec["id"]

            if sample_id in done_ids:
                continue

            prediction = rec.get("prediction", "").strip()
            if not prediction:
                skipped += 1
                continue

            # Render prompt with chat template (same as eval.py)
            messages = [{"role": "user", "content": rec["prompt"]}]
            rendered_prompt = render_chat_text(tokenizer, messages, add_generation_prompt=True)

            prompt_ids = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            response_ids = tokenizer(prediction, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

            if args.max_response_tokens:
                response_ids = response_ids[: args.max_response_tokens]

            if response_ids.numel() == 0:
                skipped += 1
                continue

            try:
                hidden_states, attentions = collect_stepwise_cache(
                    model=model,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    device=args.device,
                )

                input_len = int(prompt_ids.numel())
                core_positions = {
                    "user_prompt_start": 0,
                    "user_prompt_end": input_len,
                    "response_start": input_len,
                }

                icr_calculator = ICRScore(
                    hidden_states=hidden_states,
                    attentions=attentions,
                    skew_threshold=args.skew_threshold,
                    entropy_threshold=args.entropy_threshold,
                    core_positions=core_positions,
                    icr_device=args.device,
                )
                icr_scores, top_p_mean = icr_calculator.compute_icr(
                    top_k=args.top_k,
                    top_p=args.top_p,
                    pooling=args.pooling,
                    attention_uniform=False,
                    hidden_uniform=False,
                    use_induction_head=args.use_induction_head,
                )
            except Exception as exc:
                print(f"[WARN] Sample {sample_id} failed: {exc} — skipping")
                skipped += 1
                continue

            label = derive_label(rec["is_correct_strict"], rec["is_refusal"])

            row = {
                "id": sample_id,
                "model_tag": model_tag,
                "task": rec["task"],
                "label": label,
                "is_correct_strict": rec["is_correct_strict"],
                "is_refusal": rec["is_refusal"],
                "gold_answer": rec["gold_answer"],
                "prediction": prediction,
                "icr_scores": icr_scores,   # [num_layers][num_response_tokens]
                "top_p_mean": float(top_p_mean),
                "num_layers": len(icr_scores),
                "num_response_tokens": int(response_ids.numel()),
                "core_positions": core_positions,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

            if (i + 1) % 100 == 0:
                total = len(records)
                print(f"[{i + 1}/{total}] written={written} skipped={skipped}")

    print(f"Done. written={written} skipped={skipped} → {output_path}")


if __name__ == "__main__":
    main()
