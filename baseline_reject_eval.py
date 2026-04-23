#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from chat_utils import render_chat_text
from tasks import SUPPORTED_TASKS, TaskExample, is_open_answer_correct, load_task_examples, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run uncertainty-threshold or consistency-threshold rejection baselines."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_path", default=None, help="Optional adapter path if needed.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--tasks", nargs="+", default=["pararel", "hotpotqa"], choices=SUPPORTED_TASKS)
    parser.add_argument("--prompt_domain", choices=["ID", "OOD"], default="ID")
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--limit_per_task", type=int, default=None)
    parser.add_argument("--baseline", required=True, choices=["uncertainty", "consistency"])
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional reject threshold. "
            "uncertainty: reject if uncertainty_score > threshold; "
            "consistency: reject if consistency_score < threshold."
        ),
    )
    parser.add_argument("--num_samples", type=int, default=4, help="Used by consistency baseline.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Used by consistency baseline.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Used by consistency baseline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--attn_implementation", default=None)
    return parser.parse_args()


def resolve_torch_dtype(args: argparse.Namespace) -> torch.dtype | None:
    if args.bf16 and args.fp16:
        raise ValueError("Choose either --bf16 or --fp16, not both.")
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return None


def build_quantization_config(args: argparse.Namespace, torch_dtype: torch.dtype | None) -> BitsAndBytesConfig | None:
    if not args.load_in_4bit:
        return None
    if not torch.cuda.is_available():
        raise ValueError("--load_in_4bit requires CUDA.")
    compute_dtype = torch_dtype or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    tokenizer_source = args.adapter_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = resolve_torch_dtype(args)
    quantization_config = build_quantization_config(args, torch_dtype)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return model, tokenizer


def get_model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def postprocess_generation(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def build_prompt(example: TaskExample, tokenizer: Any) -> str:
    return render_chat_text(
        tokenizer,
        [{"role": "user", "content": example.prompt}],
        add_generation_prompt=True,
    )


def truncate_generated_ids_for_prediction(tokenizer: Any, generated_ids: torch.Tensor) -> torch.Tensor:
    if generated_ids.numel() == 0:
        return generated_ids
    decoded_full = tokenizer.decode(generated_ids, skip_special_tokens=True)
    target_prediction = postprocess_generation(decoded_full)
    if not target_prediction:
        return generated_ids[:0]

    best_cutoff = generated_ids.numel()
    for cutoff in range(1, generated_ids.numel() + 1):
        decoded_prefix = tokenizer.decode(generated_ids[:cutoff], skip_special_tokens=True)
        if postprocess_generation(decoded_prefix) != target_prediction:
            continue
        best_cutoff = cutoff
        if "\n" in decoded_prefix or decoded_prefix.strip() == target_prediction:
            break
    return generated_ids[:best_cutoff]


def generate_answer(
    model: Any,
    tokenizer: Any,
    rendered_prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    device = get_model_device(model)
    inputs = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    prompt_ids = inputs["input_ids"][0].detach().cpu()
    generated_ids_full = outputs[0][prompt_len:].detach().cpu()
    generated_ids = truncate_generated_ids_for_prediction(tokenizer, generated_ids_full)
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return postprocess_generation(generated), generated_ids, prompt_ids


def uncertainty_score_for_generated_tokens(
    model: Any,
    prompt_ids: torch.Tensor,
    generated_ids: torch.Tensor,
) -> float:
    if generated_ids.numel() == 0:
        return float("inf")

    device = get_model_device(model)
    prompt_ids_device = prompt_ids.to(device)
    generated_ids_device = generated_ids.to(device)
    input_ids = torch.cat([prompt_ids_device, generated_ids_device], dim=0).unsqueeze(0)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits

    prompt_len = prompt_ids_device.numel()
    start = prompt_len - 1
    end = start + generated_ids_device.numel()
    token_logits = logits[0, start:end, :]
    token_log_probs = torch.log_softmax(token_logits, dim=-1)
    token_seq_log_probs = token_log_probs.gather(
        dim=-1,
        index=generated_ids_device.unsqueeze(-1),
    ).squeeze(-1)
    return float(-token_seq_log_probs.mean().item())


def is_correct_prediction(example: TaskExample, prediction: str) -> bool:
    if example.answer_kind == "open":
        return is_open_answer_correct(prediction, example.gold_answer)
    pred = prediction.strip()
    if example.candidates:
        if pred not in example.candidates and pred:
            first = pred[0].upper()
            if first in example.candidates:
                pred = first
        return pred == example.gold_answer
    return pred == example.gold_answer


def derive_label(is_correct: bool, is_refusal: bool) -> str:
    if is_correct and not is_refusal:
        return "correct_confident"
    if is_correct and is_refusal:
        return "false_refusal"
    if not is_correct and is_refusal:
        return "correct_refusal"
    return "hallucination"


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)

    def _stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        labels = [derive_label(item["is_correct_strict"], item["is_refusal"]) for item in items]
        counts = {
            "correct_confident": labels.count("correct_confident"),
            "correct_refusal": labels.count("correct_refusal"),
            "false_refusal": labels.count("false_refusal"),
            "hallucination": labels.count("hallucination"),
        }
        total = len(items)
        return {
            "total": total,
            "accuracy": counts["correct_confident"] / total if total else 0.0,
            "reliability": (counts["correct_confident"] + counts["correct_refusal"]) / total if total else 0.0,
            "refusal_rate": (counts["correct_refusal"] + counts["false_refusal"]) / total if total else 0.0,
            **counts,
        }

    per_task = {task: _stats(items) for task, items in sorted(grouped.items())}
    overall = _stats(rows)
    return {
        "baseline": args.baseline,
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "data_root": args.data_root,
        "tasks": args.tasks,
        "prompt_domain": args.prompt_domain,
        "split": args.split,
        "limit_per_task": args.limit_per_task,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "threshold": args.threshold,
        "total": len(rows),
        "overall": overall,
        "per_task": per_task,
    }


def evaluate_uncertainty(
    model: Any,
    tokenizer: Any,
    example: TaskExample,
    threshold: float | None,
) -> dict[str, Any]:
    rendered_prompt = build_prompt(example, tokenizer)
    prediction, generated_ids, prompt_ids = generate_answer(
        model=model,
        tokenizer=tokenizer,
        rendered_prompt=rendered_prompt,
        max_new_tokens=example.max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )
    score = uncertainty_score_for_generated_tokens(model, prompt_ids, generated_ids)
    is_refusal = bool(threshold is not None and score > threshold)
    return {
        "id": example.sample_id,
        "task": example.task,
        "gold_answer": example.gold_answer,
        "prediction": prediction,
        "is_correct_strict": is_correct_prediction(example, prediction),
        "is_refusal": is_refusal,
        "uncertainty_score": score,
    }


def evaluate_consistency(
    model: Any,
    tokenizer: Any,
    example: TaskExample,
    threshold: float | None,
    num_samples: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    rendered_prompt = build_prompt(example, tokenizer)
    sampled_answers: list[str] = []
    for _ in range(num_samples):
        sampled_answers.append(
            generate_answer(
                model=model,
                tokenizer=tokenizer,
                rendered_prompt=rendered_prompt,
                max_new_tokens=example.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )[0]
        )

    normalized_samples = [normalize_text(answer) for answer in sampled_answers]
    counts: dict[str, int] = {}
    for normalized in normalized_samples:
        counts[normalized] = counts.get(normalized, 0) + 1
    max_count = max(counts.values()) if counts else 0
    winning_index = 0
    if counts:
        for idx, normalized in enumerate(normalized_samples):
            if counts[normalized] == max_count:
                winning_index = idx
                break
    majority_answer = sampled_answers[winning_index] if sampled_answers else ""
    consistency_score = max_count / num_samples if num_samples else 0.0
    is_refusal = bool(threshold is not None and consistency_score < threshold)
    return {
        "id": example.sample_id,
        "task": example.task,
        "gold_answer": example.gold_answer,
        "prediction": majority_answer,
        "is_correct_strict": is_correct_prediction(example, majority_answer),
        "is_refusal": is_refusal,
        "sampled_answers": sampled_answers,
        "normalized_samples": normalized_samples,
        "majority_answer": majority_answer,
        "consistency_score": consistency_score,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.baseline == "consistency" and args.num_samples <= 0:
        raise ValueError("--num_samples must be > 0 for consistency baseline.")
    if args.baseline == "consistency" and not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top_p must be in (0, 1].")

    model, tokenizer = load_model_and_tokenizer(args)
    examples = load_task_examples(
        data_root=args.data_root,
        tasks=args.tasks,
        prompt_domain=args.prompt_domain,
        limit_per_task=args.limit_per_task,
        split=args.split,
    )

    rows: list[dict[str, Any]] = []
    for example in tqdm(examples, desc=f"Evaluating {args.baseline}"):
        if args.baseline == "uncertainty":
            row = evaluate_uncertainty(model, tokenizer, example, args.threshold)
        else:
            row = evaluate_consistency(
                model=model,
                tokenizer=tokenizer,
                example=example,
                threshold=args.threshold,
                num_samples=args.num_samples,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = summarize(rows, args)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
