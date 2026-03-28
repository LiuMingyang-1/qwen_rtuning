from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qwen_rtuning.chat_utils import render_chat_text
from qwen_rtuning.tasks import (
    FALSE_RESPONSES,
    FOLLOW_UP_QUESTION,
    SUPPORTED_TASKS,
    TaskExample,
    is_open_answer_correct,
    load_task_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build R-Tuning SFT data for Qwen-style instruct models.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_root", required=True, help="Path to the extracted R-Tuning-data directory.")
    parser.add_argument("--output_path", required=True, help="Destination JSONL path.")
    parser.add_argument(
        "--method",
        choices=["unsure", "unknown", "uncertain"],
        default="unsure",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=SUPPORTED_TASKS,
        help="Tasks to include. Defaults to all supported tasks.",
    )
    parser.add_argument("--prompt_domain", choices=["ID", "OOD"], default="ID")
    parser.add_argument("--limit_per_task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_uncertainty_samples", type=int, default=5)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        default=None,
        help="Optional attention backend, for example flash_attention_2 or sdpa.",
    )
    return parser.parse_args()


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def build_quantization_config(args: argparse.Namespace) -> BitsAndBytesConfig | None:
    if not args.load_in_4bit:
        return None
    compute_dtype = get_compute_dtype()
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
        "torch_dtype": get_compute_dtype(),
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    quantization_config = build_quantization_config(args)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )
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
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    rendered = render_chat_text(
        tokenizer,
        build_prompt_messages(prompt),
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(get_model_device(model))
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

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
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
    prompt_len = prompt_ids["input_ids"].shape[1]
    with torch.inference_mode():
        # Only compute lm_head on the positions we need (candidate tokens)
        # to avoid a huge [seq_len, vocab_size] allocation that OOMs on long prompts.
        hidden_states = model.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        candidate_logits = model.lm_head(hidden_states[:, prompt_len - 1 : -1, :])

    candidate_token_ids = candidate_ids["input_ids"]
    token_log_probs = torch.log_softmax(candidate_logits, dim=-1)
    gathered = token_log_probs.gather(dim=-1, index=candidate_token_ids.unsqueeze(-1)).squeeze(-1)
    return gathered.sum().item()


def score_candidates(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: list[str],
) -> tuple[str, dict[str, float], float]:
    scores = {candidate: candidate_logprob(model, tokenizer, prompt, candidate) for candidate in candidates}
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    raw_scores = torch.tensor([item[1] for item in ordered], dtype=torch.float32)
    probabilities = torch.softmax(raw_scores, dim=0)
    entropy = float(-(probabilities * torch.log(probabilities + 1e-12)).sum().item())
    probability_map = {
        ordered[index][0]: float(probabilities[index].item())
        for index in range(len(ordered))
    }
    return ordered[0][0], probability_map, entropy


def empirical_entropy(samples: list[str]) -> float:
    counts = Counter(samples)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability + 1e-12)
    return entropy


def build_unknown_record(example: TaskExample, is_correct: bool, prediction: str, rng: random.Random) -> dict[str, Any]:
    assistant_text = example.gold_answer if is_correct else rng.choice(FALSE_RESPONSES)
    return {
        "id": example.sample_id,
        "task": example.task,
        "method": "unknown",
        "messages": [
            {"role": "user", "content": example.prompt},
            {"role": "assistant", "content": assistant_text},
        ],
        "metadata": {
            **example.metadata,
            "gold_answer": example.gold_answer,
            "model_prediction": prediction,
            "is_correct": is_correct,
        },
    }


def build_reflection_record(
    example: TaskExample,
    method: str,
    label: str,
    extra_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": example.sample_id,
        "task": example.task,
        "method": method,
        "messages": [
            {"role": "user", "content": example.prompt},
            {"role": "assistant", "content": example.gold_answer},
            {"role": "user", "content": FOLLOW_UP_QUESTION},
            {"role": "assistant", "content": label},
        ],
        "metadata": {
            **example.metadata,
            "gold_answer": example.gold_answer,
            **extra_metadata,
        },
    }


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    random.seed(args.seed)
    rng = random.Random(args.seed)
    model, tokenizer = load_model_and_tokenizer(args)
    examples = load_task_examples(
        data_root=args.data_root,
        tasks=args.tasks,
        prompt_domain=args.prompt_domain,
        limit_per_task=args.limit_per_task,
    )

    records: list[dict[str, Any]] = []
    uncertain_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    progress = tqdm(examples, desc=f"Building {args.method} dataset")
    for example in progress:
        if example.answer_kind == "classification":
            prediction, probabilities, entropy = score_candidates(
                model=model,
                tokenizer=tokenizer,
                prompt=example.prompt,
                candidates=example.candidates,
            )
            is_correct = prediction == example.gold_answer
            if args.method == "unknown":
                records.append(build_unknown_record(example, is_correct=is_correct, prediction=prediction, rng=rng))
            elif args.method == "unsure":
                label = "I am sure." if is_correct else "I am unsure."
                records.append(
                    build_reflection_record(
                        example,
                        method="unsure",
                        label=label,
                        extra_metadata={
                            "model_prediction": prediction,
                            "is_correct": is_correct,
                            "candidate_probabilities": probabilities,
                            "uncertainty_score": entropy,
                        },
                    )
                )
            else:
                uncertain_by_task[example.task].append(
                    {
                        "example": example,
                        "score": entropy,
                        "metadata": {
                            "model_prediction": prediction,
                            "candidate_probabilities": probabilities,
                        },
                    }
                )
            continue

        prediction = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=example.prompt,
            max_new_tokens=example.max_new_tokens,
            do_sample=False,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        is_correct = is_open_answer_correct(prediction, example.gold_answer)
        if args.method == "unknown":
            records.append(build_unknown_record(example, is_correct=is_correct, prediction=prediction, rng=rng))
        elif args.method == "unsure":
            label = "I am sure." if is_correct else "I am unsure."
            records.append(
                build_reflection_record(
                    example,
                    method="unsure",
                    label=label,
                    extra_metadata={
                        "model_prediction": prediction,
                        "is_correct": is_correct,
                    },
                )
            )
        else:
            sampled_answers = [
                generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=example.prompt,
                    max_new_tokens=example.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                for _ in range(args.num_uncertainty_samples)
            ]
            normalized_samples = [sample.lower().strip() for sample in sampled_answers]
            uncertain_by_task[example.task].append(
                {
                    "example": example,
                    "score": empirical_entropy(normalized_samples),
                    "metadata": {
                        "model_prediction": prediction,
                        "sampled_answers": sampled_answers,
                        "is_correct": is_correct,
                    },
                }
            )

    if args.method == "uncertain":
        # uncertain 方法的标签依据：按熵（不确定性得分）从低到高排序后，
        # 前一半（熵较低、模型输出更集中）打 "I am sure."，
        # 后一半（熵较高、模型输出更分散）打 "I am unsure."。
        #
        # 注意：此方法与 unsure/unknown 不同，标签**不依赖预测是否正确**，
        # 而是训练模型表达自身的置信度分布。因此对于某类系统性错误（模型总答同一个错
        # 误答案，熵低），模型也会被训练成"I am sure."——这是该方法的设计取舍，
        # 如需基于正确性打标签，请改用 --method unsure 或 --method unknown。
        for task, task_items in uncertain_by_task.items():
            sorted_items = sorted(task_items, key=lambda item: item["score"])
            split_index = len(sorted_items) // 2
            for index, item in enumerate(sorted_items):
                label = "I am sure." if index < split_index else "I am unsure."
                records.append(
                    build_reflection_record(
                        item["example"],
                        method="uncertain",
                        label=label,
                        extra_metadata={
                            **item["metadata"],
                            "uncertainty_score": item["score"],
                        },
                    )
                )

    summary = {
        "method": args.method,
        "model_name_or_path": args.model_name_or_path,
        "total_records": len(records),
        "tasks": {
            task: sum(1 for record in records if record["task"] == task)
            for task in sorted({record["task"] for record in records})
        },
    }
    return records, summary


def write_outputs(records: list[dict[str, Any]], summary: dict[str, Any], output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = output_file.with_suffix(output_file.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    records, summary = build_dataset(args)
    write_outputs(records, summary, args.output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

