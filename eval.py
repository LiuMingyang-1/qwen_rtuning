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
from qwen_rtuning.tasks import FOLLOW_UP_QUESTION, SUPPORTED_TASKS, TaskExample, is_open_answer_correct, load_task_examples


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
    parser.add_argument("--batch_size", type=int, default=1)
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


def batch_generate(
    model: Any,
    tokenizer: Any,
    rendered_prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    inputs = tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(get_model_device(model))
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return [
        postprocess_generation(tokenizer.decode(output[prompt_len:], skip_special_tokens=True))
        for output in outputs
    ]


def is_refusal(followup_response: str) -> bool:
    return "i am unsure" in followup_response.lower()


def batch_candidate_logprob(
    model: Any,
    tokenizer: Any,
    rendered_prompts: list[str],
    candidates: list[str],
) -> list[float]:
    """Compute log-prob of each candidate given the corresponding prompt, in a single batched forward pass."""
    device = get_model_device(model)
    prompt_id_list = [
        tokenizer(p, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        for p in rendered_prompts
    ]
    candidate_id_list = [
        tokenizer(c, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        for c in candidates
    ]
    combined = [torch.cat([p, c]) for p, c in zip(prompt_id_list, candidate_id_list)]
    max_len = max(t.shape[0] for t in combined)

    padded_input_ids = []
    attention_masks = []
    for seq in combined:
        pad_len = max_len - seq.shape[0]
        padded_input_ids.append(torch.cat([torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long), seq]))
        attention_masks.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(seq.shape[0], dtype=torch.long)]))

    input_ids = torch.stack(padded_input_ids).to(device)
    attention_mask = torch.stack(attention_masks).to(device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    results = []
    for i, (prompt_ids, cand_ids) in enumerate(zip(prompt_id_list, candidate_id_list)):
        pad_len = max_len - (prompt_ids.shape[0] + cand_ids.shape[0])
        # position of first candidate token in padded sequence
        cand_start = pad_len + prompt_ids.shape[0] - 1
        cand_end = cand_start + cand_ids.shape[0]
        cand_logits = logits[i, cand_start:cand_end, :]
        log_probs = torch.log_softmax(cand_logits, dim=-1)
        gathered = log_probs.gather(dim=-1, index=cand_ids.to(device).unsqueeze(-1)).squeeze(-1)
        results.append(gathered.sum().item())
    return results


def evaluate_classification_batch(model: Any, tokenizer: Any, examples: list[TaskExample]) -> list[dict[str, Any]]:
    rendered_prompts = [
        render_chat_text(tokenizer, build_prompt_messages(ex.prompt), add_generation_prompt=True)
        for ex in examples
    ]
    # Score each candidate across all examples: one batched forward pass per candidate.
    candidates = examples[0].candidates
    scores_per_candidate: dict[str, list[float]] = {}
    for candidate in candidates:
        scores_per_candidate[candidate] = batch_candidate_logprob(
            model, tokenizer, rendered_prompts, [candidate] * len(examples)
        )

    results = []
    for i, ex in enumerate(examples):
        scores = {c: scores_per_candidate[c][i] for c in candidates}
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        raw_scores = torch.tensor([item[1] for item in ordered], dtype=torch.float32)
        probabilities = torch.softmax(raw_scores, dim=0)
        probability_map = {ordered[j][0]: float(probabilities[j].item()) for j in range(len(ordered))}
        prediction = ordered[0][0]
        is_correct = prediction == ex.gold_answer
        results.append({
            "id": ex.sample_id,
            "task": ex.task,
            "answer_kind": ex.answer_kind,
            "prompt": ex.prompt,
            "gold_answer": ex.gold_answer,
            "prediction": prediction,
            "is_correct_strict": is_correct,
            "is_correct_rtuning": is_correct,
            "is_refusal": False,
            "followup_response": None,
            "candidate_probabilities": probability_map,
            "metadata": ex.metadata,
        })
    return results


def evaluate_open_batch(model: Any, tokenizer: Any, examples: list[TaskExample]) -> list[dict[str, Any]]:
    max_new_tokens = max(ex.max_new_tokens for ex in examples)
    answer_prompts = [
        render_chat_text(tokenizer, build_prompt_messages(ex.prompt), add_generation_prompt=True)
        for ex in examples
    ]
    predictions = batch_generate(model, tokenizer, answer_prompts, max_new_tokens)

    followup_prompts = [
        render_chat_text(
            tokenizer,
            [
                {"role": "user", "content": ex.prompt},
                {"role": "assistant", "content": pred},
                {"role": "user", "content": FOLLOW_UP_QUESTION},
            ],
            add_generation_prompt=True,
        )
        for ex, pred in zip(examples, predictions)
    ]
    followup_responses = batch_generate(model, tokenizer, followup_prompts, max_new_tokens=16)

    results = []
    for ex, prediction, followup_response in zip(examples, predictions, followup_responses):
        is_correct_strict = is_open_answer_correct(prediction, ex.gold_answer)
        refused = is_refusal(followup_response)
        results.append({
            "id": ex.sample_id,
            "task": ex.task,
            "answer_kind": ex.answer_kind,
            "prompt": ex.prompt,
            "gold_answer": ex.gold_answer,
            "prediction": prediction,
            "is_correct_strict": is_correct_strict,
            "is_correct_rtuning": is_correct_strict or refused,
            "is_refusal": refused,
            "followup_response": followup_response,
            "metadata": ex.metadata,
        })
    return results


def summarize_predictions(predictions: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_task[prediction["task"]].append(prediction)

    task_metrics: dict[str, Any] = {}
    total_correct_strict = 0
    total_correct_rtuning = 0
    total_refused = 0
    for task, items in sorted(by_task.items()):
        correct_strict = sum(1 for item in items if item["is_correct_strict"])
        correct_rtuning = sum(1 for item in items if item["is_correct_rtuning"])
        refused = sum(1 for item in items if item["is_refusal"])
        total = len(items)
        total_correct_strict += correct_strict
        total_correct_rtuning += correct_rtuning
        total_refused += refused
        task_metrics[task] = {
            "total": total,
            "correct_strict": correct_strict,
            "correct_rtuning": correct_rtuning,
            "refused": refused,
            "accuracy_strict": correct_strict / total if total else 0.0,
            "accuracy_rtuning": correct_rtuning / total if total else 0.0,
            "refusal_rate": refused / total if total else 0.0,
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
        "correct_strict": total_correct_strict,
        "correct_rtuning": total_correct_rtuning,
        "refused": total_refused,
        "accuracy_strict": total_correct_strict / total if total else 0.0,
        "accuracy_rtuning": total_correct_rtuning / total if total else 0.0,
        "refusal_rate": total_refused / total if total else 0.0,
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
    open_buffer: list[TaskExample] = []
    classification_buffer: list[TaskExample] = []

    def flush_open_buffer() -> None:
        if open_buffer:
            predictions.extend(evaluate_open_batch(model, tokenizer, open_buffer))
            open_buffer.clear()

    def flush_classification_buffer() -> None:
        if classification_buffer:
            predictions.extend(evaluate_classification_batch(model, tokenizer, classification_buffer))
            classification_buffer.clear()

    for example in tqdm(examples, desc=f"Evaluating {args.split} split"):
        if example.answer_kind == "classification":
            flush_open_buffer()
            classification_buffer.append(example)
            if len(classification_buffer) >= args.batch_size:
                flush_classification_buffer()
        else:
            flush_classification_buffer()
            open_buffer.append(example)
            if len(open_buffer) >= args.batch_size:
                flush_open_buffer()

    flush_open_buffer()
    flush_classification_buffer()

    metrics = summarize_predictions(predictions, args)
    write_outputs(predictions, metrics, args.output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
