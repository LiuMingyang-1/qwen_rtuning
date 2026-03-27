from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


FOLLOW_UP_QUESTION = "Are you sure you accurately answered the question based on your internal knowledge?"

FALSE_RESPONSES = [
    "The answer is unknown.",
    "The answer is uncertain.",
    "The answer is unclear.",
    "It is not known.",
    "I do not know the answer.",
    "I'm not sure.",
    "There is no definitive answer.",
    "There is much debate.",
    "There is no concrete answer to this question.",
    "It is impossible to answer.",
    "There is no known case.",
    "There is no public information available.",
    "There is no scientific evidence.",
    "There is no right answer.",
    "It is impossible to know.",
    "It is difficult to predict.",
]

SUPPORTED_TASKS = ["pararel", "mmlu", "fever", "hotpotqa", "wice"]

DEFAULT_TASK_FILES = {
    "pararel": Path("pararel/training_data.json"),
    "mmlu_train": Path("MMLU/MMLU_ID_train.json"),
    "mmlu_prompt_id": Path("MMLU/MMLU_ID_prompt.json"),
    "mmlu_prompt_ood": Path("MMLU/MMLU_OOD_prompt.json"),
    "fever": Path("FEVER/fever_10k.json"),
    "hotpotqa": Path("HotpotQA/hotpot_10k.json"),
    "wice": Path("WiCE/wice_train.json"),
}


@dataclass
class TaskExample:
    task: str
    sample_id: str
    prompt: str
    gold_answer: str
    answer_kind: str
    candidates: list[str] = field(default_factory=list)
    max_new_tokens: int = 32
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_task_name(task: str) -> str:
    normalized = task.strip().lower()
    if normalized not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {', '.join(SUPPORTED_TASKS)}")
    return normalized


def load_task_examples(
    data_root: str | Path,
    tasks: Iterable[str],
    prompt_domain: str = "ID",
    limit_per_task: int | None = None,
) -> list[TaskExample]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    examples: list[TaskExample] = []
    for task in tasks:
        normalized = normalize_task_name(task)
        task_examples = list(_iter_task_examples(root, normalized, prompt_domain=prompt_domain))
        if limit_per_task is not None:
            task_examples = task_examples[:limit_per_task]
        examples.extend(task_examples)
    return examples


def normalize_text(text: str) -> str:
    lowered = text.lower()
    no_punc = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punc)
    return " ".join(no_articles.split())


def is_open_answer_correct(prediction: str, gold_answer: str) -> bool:
    pred = normalize_text(prediction)
    gold = normalize_text(gold_answer)
    if not pred or not gold:
        return False
    return pred == gold or pred in gold or gold in pred


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset file was not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_task_examples(root: Path, task: str, prompt_domain: str) -> Iterable[TaskExample]:
    if task == "pararel":
        yield from _iter_pararel_examples(root / DEFAULT_TASK_FILES["pararel"])
        return
    if task == "mmlu":
        train_path = root / DEFAULT_TASK_FILES["mmlu_train"]
        prompt_key = "mmlu_prompt_id" if prompt_domain.upper() == "ID" else "mmlu_prompt_ood"
        prompt_path = root / DEFAULT_TASK_FILES[prompt_key]
        yield from _iter_mmlu_examples(train_path, prompt_path)
        return
    if task == "fever":
        yield from _iter_fever_examples(root / DEFAULT_TASK_FILES["fever"])
        return
    if task == "hotpotqa":
        yield from _iter_hotpot_examples(root / DEFAULT_TASK_FILES["hotpotqa"])
        return
    if task == "wice":
        yield from _iter_wice_examples(root / DEFAULT_TASK_FILES["wice"])
        return
    raise ValueError(f"Unexpected task: {task}")


def _iter_pararel_examples(path: Path) -> Iterable[TaskExample]:
    data = _load_json(path)
    for index, sample in enumerate(data):
        question, answer, *_ = sample
        prompt = (
            f"Question: {question}\n"
            "Answer with a short entity or phrase only."
        )
        yield TaskExample(
            task="pararel",
            sample_id=f"pararel-{index}",
            prompt=prompt,
            gold_answer=str(answer).strip(),
            answer_kind="open",
            max_new_tokens=20,
            metadata={"question": question},
        )


def _format_subject(subject: str) -> str:
    return " ".join(subject.split("_"))


def _format_mmlu_example(sample: list[str]) -> str:
    prompt = sample[0]
    num_choices = len(sample) - 2
    for choice_index in range(num_choices):
        option = chr(ord("A") + choice_index)
        prompt += f"\n{option}. {sample[choice_index + 1]}"
    prompt += "\nAnswer:"
    return prompt


def _format_mmlu_shots(prompt_data: list[list[str]]) -> str:
    formatted: list[str] = []
    for sample in prompt_data:
        num_choices = len(sample) - 2
        block = [sample[0]]
        for choice_index in range(num_choices):
            option = chr(ord("A") + choice_index)
            block.append(f"{option}. {sample[choice_index + 1]}")
        block.append(f"Answer: {sample[num_choices + 1]}")
        formatted.append("\n".join(block))
    return "\n\n".join(formatted)


def _iter_mmlu_examples(train_path: Path, prompt_path: Path) -> Iterable[TaskExample]:
    train_data = _load_json(train_path)
    prompt_data = _load_json(prompt_path)
    for subject, samples in train_data.items():
        subject_prompt = (
            "The following are multiple choice questions (with answers) "
            f"about {_format_subject(subject)}. Respond with the option letter only.\n\n"
        )
        formatted_shots = _format_mmlu_shots(prompt_data[subject])
        for index, sample in enumerate(samples):
            prompt = subject_prompt + formatted_shots + "\n\n" + _format_mmlu_example(sample)
            yield TaskExample(
                task="mmlu",
                sample_id=f"mmlu-{subject}-{index}",
                prompt=prompt,
                gold_answer=str(sample[5]).strip(),
                answer_kind="classification",
                candidates=["A", "B", "C", "D"],
                max_new_tokens=4,
                metadata={"subject": subject},
            )


def _iter_fever_examples(path: Path) -> Iterable[TaskExample]:
    mapping = {"SUPPORTS": "A", "REFUTES": "B", "NOT ENOUGH INFO": "C"}
    candidate_answer = ["SUPPORTS.", "REFUTES.", "NOT ENOUGH INFO."]
    data = _load_json(path)
    for index, sample in enumerate(data):
        evidence = " ".join(sample["evidence"])
        prompt = (
            f"Evidence: {evidence}\n"
            f"Claim: {sample['claim']}\n"
            "Question: Does the evidence support the claim?\n"
            f"A. {candidate_answer[0]}\n"
            f"B. {candidate_answer[1]}\n"
            f"C. {candidate_answer[2]}\n"
            "Answer with the option letter only."
        )
        yield TaskExample(
            task="fever",
            sample_id=f"fever-{index}",
            prompt=prompt,
            gold_answer=mapping[sample["label"]],
            answer_kind="classification",
            candidates=["A", "B", "C"],
            max_new_tokens=4,
            metadata={"label": sample["label"]},
        )


def _iter_hotpot_examples(path: Path) -> Iterable[TaskExample]:
    data = _load_json(path)
    for index, sample in enumerate(data):
        context_blocks = []
        for title, sentences in sample["context"]:
            context_blocks.append(f"{title}: {' '.join(sentences)}")
        context = "\n".join(context_blocks)
        prompt = (
            f"{context}\n\n"
            f"Question: {sample['question']}\n"
            "Answer with a short entity or phrase only."
        )
        yield TaskExample(
            task="hotpotqa",
            sample_id=f"hotpotqa-{index}",
            prompt=prompt,
            gold_answer=str(sample["answer"]).strip(),
            answer_kind="open",
            max_new_tokens=24,
            metadata={"question": sample["question"]},
        )


def _iter_wice_examples(path: Path) -> Iterable[TaskExample]:
    mapping = {"supported": "A", "partially_supported": "B", "not_supported": "C"}
    candidate_answer = ["supported.", "partially_supported.", "not_supported."]
    data = _load_json(path)
    for index, sample in enumerate(data):
        evidence = " ".join(sample["evidence"])
        prompt = (
            f"Evidence: {evidence}\n"
            f"Claim: {sample['claim']}\n"
            "Question: Does the evidence support the claim?\n"
            f"A. {candidate_answer[0]}\n"
            f"B. {candidate_answer[1]}\n"
            f"C. {candidate_answer[2]}\n"
            "Answer with the option letter only."
        )
        yield TaskExample(
            task="wice",
            sample_id=f"wice-{index}",
            prompt=prompt,
            gold_answer=mapping[sample["label"]],
            answer_kind="classification",
            candidates=["A", "B", "C"],
            max_new_tokens=4,
            metadata={"label": sample["label"]},
        )

