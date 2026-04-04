from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import defaultdict
from typing import Any

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from qwen_rtuning.chat_utils import render_chat_text
from qwen_rtuning.tasks import SUPPORTED_TASKS, TaskExample, is_open_answer_correct, load_task_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation for Qwen R-Tuning models.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_path", default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--data_root", required=True, help="Path to R-Tuning-data directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=SUPPORTED_TASKS)
    parser.add_argument("--prompt_domain", choices=["ID", "OOD"], default="ID")
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--limit_per_task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        default=None,
        help="Optional attention backend, for example flash_attention_2 or sdpa.",
    )
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
    compute_dtype = torch_dtype or (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()
    return model, tokenizer


def get_model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def postprocess_generation(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def build_prompt_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def generate_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> str:
    rendered = render_chat_text(
        tokenizer,
        build_prompt_messages(prompt),
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(get_model_device(model))
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return postprocess_generation(tokenizer.decode(generated_ids, skip_special_tokens=True))


def candidate_logprob(model: Any, tokenizer: Any, prompt: str, candidate: str) -> float:
    rendered = render_chat_text(
        tokenizer,
        build_prompt_messages(prompt),
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to(get_model_device(model))
    candidate_ids = tokenizer(candidate, add_special_tokens=False, return_tensors="pt").to(get_model_device(model))

    input_ids = torch.cat([prompt_ids["input_ids"], candidate_ids["input_ids"]], dim=1)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    prompt_len = prompt_ids["input_ids"].shape[1]
    candidate_logits = logits[:, prompt_len - 1:-1, :]
    candidate_token_ids = candidate_ids["input_ids"]
    token_log_probs = torch.log_softmax(candidate_logits, dim=-1)
    gathered = token_log_probs.gather(dim=-1, index=candidate_token_ids.unsqueeze(-1)).squeeze(-1)
    return gathered.sum().item()


def score_candidates(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: list[str],
) -> tuple[str, dict[str, float]]:
    scores = {candidate: candidate_logprob(model, tokenizer, prompt, candidate) for candidate in candidates}
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    raw_scores = torch.tensor([item[1] for item in ordered], dtype=torch.float32)
    probabilities = torch.softmax(raw_scores, dim=0)
    probability_map = {
        ordered[index][0]: float(probabilities[index].item())
        for index in range(len(ordered))
    }
    return ordered[0][0], probability_map


def evaluate_example(model: Any, tokenizer: Any, example: TaskExample) -> dict[str, Any]:
    if example.answer_kind == "classification":
        prediction, candidate_probabilities = score_candidates(
            model=model,
            tokenizer=tokenizer,
            prompt=example.prompt,
            candidates=example.candidates,
        )
        is_correct = prediction == example.gold_answer
        return {
            "id": example.sample_id,
            "task": example.task,
            "answer_kind": example.answer_kind,
            "prompt": example.prompt,
            "gold_answer": example.gold_answer,
            "prediction": prediction,
            "is_correct": is_correct,
            "candidate_probabilities": candidate_probabilities,
            "metadata": example.metadata,
        }

    prediction = generate_answer(
        model=model,
        tokenizer=tokenizer,
        prompt=example.prompt,
        max_new_tokens=example.max_new_tokens,
    )
    is_correct = is_open_answer_correct(prediction, example.gold_answer)
    return {
        "id": example.sample_id,
        "task": example.task,
        "answer_kind": example.answer_kind,
        "prompt": example.prompt,
        "gold_answer": example.gold_answer,
        "prediction": prediction,
        "is_correct": is_correct,
        "metadata": example.metadata,
    }


def summarize_predictions(predictions: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_task[prediction["task"]].append(prediction)

    task_metrics: dict[str, Any] = {}
    total_correct = 0
    for task, items in sorted(by_task.items()):
        correct = sum(1 for item in items if item["is_correct"])
        total = len(items)
        total_correct += correct
        task_metrics[task] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
        }

    total = len(predictions)
    return {
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "data_root": args.data_root,
        "split": args.split,
        "prompt_domain": args.prompt_domain,
        "tasks": args.tasks,
        "limit_per_task": args.limit_per_task,
        "total": total,
        "correct": total_correct,
        "accuracy": total_correct / total if total else 0.0,
        "per_task": task_metrics,
    }


def write_outputs(predictions: list[dict[str, Any]], metrics: dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions_path = output_path / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    metrics_path = output_path / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    examples = load_task_examples(
        data_root=args.data_root,
        tasks=args.tasks,
        prompt_domain=args.prompt_domain,
        limit_per_task=args.limit_per_task,
        split=args.split,
    )

    predictions: list[dict[str, Any]] = []
    for example in tqdm(examples, desc=f"Evaluating {args.split} split"):
        predictions.append(evaluate_example(model=model, tokenizer=tokenizer, example=example))

    metrics = summarize_predictions(predictions, args)
    write_outputs(predictions, metrics, args.output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
