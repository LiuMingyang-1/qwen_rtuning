from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from qwen_rtuning.chat_utils import tokenize_conversation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT for Qwen R-Tuning datasets.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_path", required=True, help="Path to JSONL built by build_dataset.py.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        default=None,
        help="Optional attention backend, for example flash_attention_2 or sdpa.",
    )
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA.",
    )
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument(
        "--skip_truncated",
        action="store_true",
        help="跳过 tokenization 后长度超过 max_length 的样本，而不是截断它们。"
             "适用于 HotpotQA 等长文档任务，避免用残缺上下文训练。",
    )
    return parser.parse_args()


def parse_target_modules(raw_value: str) -> list[str]:
    return [module.strip() for module in raw_value.split(",") if module.strip()]


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


def load_records(dataset_path: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        raise ValueError("JSON dataset must be a list of records.")

    raise ValueError("Unsupported dataset format. Use .jsonl or .json.")


class ConversationDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        skip_truncated: bool = False,
    ):
        self.samples: list[dict[str, list[int]]] = []
        self.truncated_count = 0
        self.skipped_count = 0

        for record in records:
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"Record is missing messages: {record}")

            tokenized = tokenize_conversation(tokenizer, messages, max_length=max_length)
            if not any(label != -100 for label in tokenized["labels"]):
                raise ValueError(f"Record has no assistant supervision after tokenization: {record.get('id')}")

            if tokenized["was_truncated"]:
                self.truncated_count += 1
                if skip_truncated:
                    self.skipped_count += 1
                    continue
            self.samples.append(
                {
                    "input_ids": tokenized["input_ids"],
                    "labels": tokenized["labels"],
                    "attention_mask": tokenized["attention_mask"],
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.samples[index]


class SupervisedDataCollator:
    def __init__(self, tokenizer: Any):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("train.py currently expects at least one CUDA device.")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=parse_target_modules(args.target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    tokenizer.model_max_length = args.max_length
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
        logging_strategy="steps",
        save_strategy="steps",
        seed=args.seed,
        tf32=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    records = load_records(args.dataset_path)
    dataset = ConversationDataset(
        records,
        tokenizer=tokenizer,
        max_length=args.max_length,
        skip_truncated=args.skip_truncated,
    )
    print(f"Loaded {len(dataset)} samples from {args.dataset_path}")
    if dataset.truncated_count:
        if args.skip_truncated:
            print(f"Skipped {dataset.skipped_count} samples exceeding max_length={args.max_length}")
        else:
            print(f"Samples truncated to max_length={args.max_length}: {dataset.truncated_count}")

    training_args = build_training_arguments(args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SupervisedDataCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
