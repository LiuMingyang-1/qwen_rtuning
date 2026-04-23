#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError(
        "compare_baselines.py requires scikit-learn. Install it with: pip install scikit-learn"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare base / r-tuning / ICR methods and new rejection baselines on a shared val/test split."
        )
    )
    parser.add_argument(
        "--base_predictions_path",
        default="outputs/eval/unsure-id-base/predictions.jsonl",
    )
    parser.add_argument(
        "--rtuning_predictions_path",
        default="outputs/eval/unsure-id-rtuning/predictions.jsonl",
    )
    parser.add_argument(
        "--icr_base_path",
        default="icr_analysis/outputs/outputs/icr_scores_base.jsonl",
    )
    parser.add_argument(
        "--icr_rtuning_path",
        default="icr_analysis/outputs/outputs/icr_scores_rtuning.jsonl",
    )
    parser.add_argument(
        "--uncertainty_predictions_path",
        default="outputs/eval/baselines/uncertainty/predictions.jsonl",
    )
    parser.add_argument(
        "--consistency_predictions_path",
        default="outputs/eval/baselines/consistency/predictions.jsonl",
    )
    parser.add_argument("--tasks", nargs="+", default=["pararel", "hotpotqa"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument(
        "--icr_probe_threshold",
        type=float,
        default=0.5,
        help="Fixed threshold for ICR probe firing in icr_only and icr+r-tuning(OR).",
    )
    parser.add_argument(
        "--allow_probe_train_on_all_common_for_debug",
        action="store_true",
        help=(
            "Debug only: if val split is unusable for probe training (single class / too few samples), "
            "allow fallback to training on all common IDs."
        ),
    )
    parser.add_argument("--output_dir", default="outputs/eval/baselines/comparison")
    return parser.parse_args()


def load_jsonl_as_map(path: Path, tasks: set[str] | None = None) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task = rec.get("task")
            if tasks is not None and task not in tasks:
                continue
            sid = rec["id"]
            records[sid] = rec
    return records


def icr_vector(icr_scores: list[list[float]]) -> np.ndarray:
    return np.array([np.mean(layer_scores) for layer_scores in icr_scores], dtype=np.float32)


def derive_label(is_correct: bool, is_refusal: bool) -> str:
    if is_correct and not is_refusal:
        return "correct_confident"
    if is_correct and is_refusal:
        return "false_refusal"
    if not is_correct and is_refusal:
        return "correct_refusal"
    return "hallucination"


def compute_stats(labels: list[str]) -> dict[str, Any]:
    total = len(labels)
    if total == 0:
        return {
            "total": 0,
            "accuracy": 0.0,
            "reliability": 0.0,
            "refusal_rate": 0.0,
            "correct_confident": 0,
            "correct_refusal": 0,
            "false_refusal": 0,
            "hallucination": 0,
        }
    counts = {
        "correct_confident": labels.count("correct_confident"),
        "correct_refusal": labels.count("correct_refusal"),
        "false_refusal": labels.count("false_refusal"),
        "hallucination": labels.count("hallucination"),
    }
    return {
        "total": total,
        "accuracy": counts["correct_confident"] / total,
        "reliability": (counts["correct_confident"] + counts["correct_refusal"]) / total,
        "refusal_rate": (counts["correct_refusal"] + counts["false_refusal"]) / total,
        **counts,
    }


def build_probe_dataset(
    records: dict[str, dict[str, Any]],
    ids: list[str],
    pos_labels: set[str],
    neg_labels: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for sid in ids:
        rec = records[sid]
        label = rec.get("label")
        if label in pos_labels:
            labels.append(1)
        elif label in neg_labels:
            labels.append(0)
        else:
            continue
        features.append(icr_vector(rec["icr_scores"]))
    if not features:
        raise ValueError("No samples found for probe training.")
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    if y.sum() == 0 or int((y == 0).sum()) == 0:
        raise ValueError("Probe training labels contain only one class.")
    return X, y


def build_probe_dataset_with_fallback(
    records: dict[str, dict[str, Any]],
    primary_ids: list[str],
    fallback_ids: list[str],
    pos_labels: set[str],
    neg_labels: set[str],
    probe_name: str,
    allow_debug_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        X, y = build_probe_dataset(records, primary_ids, pos_labels=pos_labels, neg_labels=neg_labels)
        return X, y, "val"
    except ValueError as exc:
        if not allow_debug_fallback:
            raise ValueError(
                f"{probe_name} probe cannot train on val split ({exc}). "
                "Use --allow_probe_train_on_all_common_for_debug only for smoke/debug runs."
            ) from exc
        print(f"[WARN] {probe_name} probe cannot train on val split ({exc}); using debug fallback: all common IDs.")
        X, y = build_probe_dataset(records, fallback_ids, pos_labels=pos_labels, neg_labels=neg_labels)
        return X, y, "all_common"


def train_probe(X: np.ndarray, y: np.ndarray, seed: int) -> Any:
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, C=1.0),
    )
    probe.fit(X, y)
    return probe


def compute_probe_probabilities(
    records: dict[str, dict[str, Any]],
    ids: list[str],
    probe: Any,
) -> dict[str, float]:
    probs: dict[str, float] = {}
    for sid in ids:
        vec = icr_vector(records[sid]["icr_scores"]).reshape(1, -1)
        probs[sid] = float(probe.predict_proba(vec)[0, 1])
    return probs


def threshold_candidates(scores: list[float]) -> list[float]:
    if not scores:
        raise ValueError("Cannot select threshold with empty score list.")
    unique = sorted(set(float(s) for s in scores))
    min_score = unique[0]
    max_score = unique[-1]
    eps = 1e-12
    return [min_score - eps] + unique + [max_score + eps]


def evaluate_threshold_method(
    ids: list[str],
    records: dict[str, dict[str, Any]],
    score_key: str,
    threshold: float,
    reject_rule: Callable[[float, float], bool],
) -> dict[str, Any]:
    labels = []
    for sid in ids:
        rec = records[sid]
        score = float(rec[score_key])
        is_correct = bool(rec["is_correct_strict"])
        is_refusal = reject_rule(score, threshold)
        labels.append(derive_label(is_correct, is_refusal))
    return compute_stats(labels)


def select_best_threshold(
    ids: list[str],
    records: dict[str, dict[str, Any]],
    score_key: str,
    reject_rule: Callable[[float, float], bool],
) -> dict[str, Any]:
    scores = [float(records[sid][score_key]) for sid in ids]
    best: dict[str, Any] | None = None
    for threshold in threshold_candidates(scores):
        stats = evaluate_threshold_method(ids, records, score_key, threshold, reject_rule)
        rank = (stats["reliability"], stats["accuracy"], -stats["refusal_rate"])
        if best is None or rank > best["rank"]:
            best = {"threshold": threshold, "stats": stats, "rank": rank}
    if best is None:
        raise ValueError("Threshold selection failed unexpectedly.")
    return {"threshold": best["threshold"], "val_stats": best["stats"]}


def evaluate_method_by_task(
    ids: list[str],
    id_to_task: dict[str, str],
    method_label_by_id: dict[str, str],
    tasks: list[str],
) -> dict[str, Any]:
    overall_labels = [method_label_by_id[sid] for sid in ids]
    per_task = {}
    for task in tasks:
        task_ids = [sid for sid in ids if id_to_task[sid] == task]
        per_task[task] = compute_stats([method_label_by_id[sid] for sid in task_ids])
    return {"overall": compute_stats(overall_labels), "per_task": per_task}


def build_split(
    common_ids: list[str],
    base_records: dict[str, dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    stratify_labels = [
        f"{base_records[sid]['task']}|{int(bool(base_records[sid]['is_correct_strict']))}" for sid in common_ids
    ]
    try:
        val_ids, test_ids = train_test_split(
            common_ids,
            test_size=1.0 - val_ratio,
            stratify=stratify_labels,
            random_state=seed,
        )
    except ValueError as exc:
        print(f"[WARN] Stratified split failed ({exc}); falling back to random split.")
        val_ids, test_ids = train_test_split(
            common_ids,
            test_size=1.0 - val_ratio,
            random_state=seed,
            shuffle=True,
        )
    return sorted(val_ids), sorted(test_ids)


def write_csv_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val_ratio must be in (0, 1).")

    tasks = [task.strip().lower() for task in args.tasks]
    task_set = set(tasks)

    base_records = load_jsonl_as_map(Path(args.base_predictions_path), tasks=task_set)
    rtuning_records = load_jsonl_as_map(Path(args.rtuning_predictions_path), tasks=task_set)
    icr_base_records = load_jsonl_as_map(Path(args.icr_base_path), tasks=task_set)
    icr_rtuning_records = load_jsonl_as_map(Path(args.icr_rtuning_path), tasks=task_set)
    uncertainty_records = load_jsonl_as_map(Path(args.uncertainty_predictions_path), tasks=task_set)
    consistency_records = load_jsonl_as_map(Path(args.consistency_predictions_path), tasks=task_set)

    id_sets = {
        "base_predictions": set(base_records),
        "rtuning_predictions": set(rtuning_records),
        "icr_base": set(icr_base_records),
        "icr_rtuning": set(icr_rtuning_records),
        "uncertainty": set(uncertainty_records),
        "consistency": set(consistency_records),
    }
    base_rt_common = id_sets["base_predictions"] & id_sets["rtuning_predictions"]
    common_ids = sorted(set.intersection(*id_sets.values()))
    if not common_ids:
        raise ValueError("No common IDs across required inputs.")

    val_ids, test_ids = build_split(common_ids, base_records, val_ratio=args.val_ratio, seed=args.seed)
    id_to_task = {sid: base_records[sid]["task"] for sid in common_ids}

    # Train ICR probes on val split only.
    X_base, y_base, base_probe_train_source = build_probe_dataset_with_fallback(
        icr_base_records,
        primary_ids=val_ids,
        fallback_ids=common_ids,
        pos_labels={"hallucination"},
        neg_labels={"correct_confident"},
        probe_name="base",
        allow_debug_fallback=args.allow_probe_train_on_all_common_for_debug,
    )
    X_rt, y_rt, rt_probe_train_source = build_probe_dataset_with_fallback(
        icr_rtuning_records,
        primary_ids=val_ids,
        fallback_ids=common_ids,
        pos_labels={"hallucination", "correct_refusal"},
        neg_labels={"correct_confident", "false_refusal"},
        probe_name="rtuning",
        allow_debug_fallback=args.allow_probe_train_on_all_common_for_debug,
    )
    probe_base = train_probe(X_base, y_base, seed=args.seed)
    probe_rt = train_probe(X_rt, y_rt, seed=args.seed)
    base_probe_probs = compute_probe_probabilities(icr_base_records, test_ids, probe_base)
    rt_probe_probs = compute_probe_probabilities(icr_rtuning_records, test_ids, probe_rt)

    uncertainty_threshold = select_best_threshold(
        val_ids,
        uncertainty_records,
        score_key="uncertainty_score",
        reject_rule=lambda score, threshold: score > threshold,
    )
    consistency_threshold = select_best_threshold(
        val_ids,
        consistency_records,
        score_key="consistency_score",
        reject_rule=lambda score, threshold: score < threshold,
    )

    method_labels: dict[str, dict[str, str]] = {}

    method_labels["base"] = {
        sid: derive_label(bool(base_records[sid]["is_correct_strict"]), False) for sid in test_ids
    }
    method_labels["r-tuning_only"] = {
        sid: derive_label(bool(rtuning_records[sid]["is_correct_strict"]), bool(rtuning_records[sid]["is_refusal"]))
        for sid in test_ids
    }
    method_labels["icr_only"] = {
        sid: derive_label(
            bool(base_records[sid]["is_correct_strict"]),
            base_probe_probs[sid] >= args.icr_probe_threshold,
        )
        for sid in test_ids
    }
    method_labels["icr+r-tuning(OR)"] = {
        sid: derive_label(
            bool(rtuning_records[sid]["is_correct_strict"]),
            bool(rtuning_records[sid]["is_refusal"]) or (rt_probe_probs[sid] >= args.icr_probe_threshold),
        )
        for sid in test_ids
    }

    uncertainty_t = float(uncertainty_threshold["threshold"])
    consistency_t = float(consistency_threshold["threshold"])
    method_labels["uncertainty-threshold reject"] = {
        sid: derive_label(
            bool(uncertainty_records[sid]["is_correct_strict"]),
            float(uncertainty_records[sid]["uncertainty_score"]) > uncertainty_t,
        )
        for sid in test_ids
    }
    method_labels["4-sample consistency reject"] = {
        sid: derive_label(
            bool(consistency_records[sid]["is_correct_strict"]),
            float(consistency_records[sid]["consistency_score"]) < consistency_t,
        )
        for sid in test_ids
    }

    metrics = {
        method: evaluate_method_by_task(test_ids, id_to_task, labels, tasks)
        for method, labels in method_labels.items()
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_payload = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "tasks": tasks,
        "total_common_ids": len(common_ids),
        "input_counts": {name: len(ids) for name, ids in id_sets.items()},
        "dropped_from_base_rt_common": {name: len(base_rt_common - ids) for name, ids in id_sets.items()},
        "val_size": len(val_ids),
        "test_size": len(test_ids),
        "val_ids": val_ids,
        "test_ids": test_ids,
    }
    (output_dir / "split.json").write_text(json.dumps(split_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    thresholds_payload = {
        "uncertainty": uncertainty_threshold,
        "consistency": consistency_threshold,
        "icr_probe_threshold": args.icr_probe_threshold,
        "probe_train_source": {
            "base": base_probe_train_source,
            "rtuning": rt_probe_train_source,
        },
        "allow_probe_train_on_all_common_for_debug": args.allow_probe_train_on_all_common_for_debug,
    }
    (output_dir / "best_thresholds.json").write_text(
        json.dumps(thresholds_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    overall_payload = {method: result["overall"] for method, result in metrics.items()}
    per_task_payload = {method: result["per_task"] for method, result in metrics.items()}
    (output_dir / "overall_metrics.json").write_text(
        json.dumps(overall_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "per_task_metrics.json").write_text(
        json.dumps(per_task_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "comparison_results.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    table_rows: list[dict[str, Any]] = []
    for method, result in metrics.items():
        overall = result["overall"]
        table_rows.append(
            {
                "method": method,
                "task": "all",
                "accuracy": overall["accuracy"],
                "reliability": overall["reliability"],
                "refusal_rate": overall["refusal_rate"],
                "total": overall["total"],
            }
        )
        for task in tasks:
            task_stats = result["per_task"][task]
            table_rows.append(
                {
                    "method": method,
                    "task": task,
                    "accuracy": task_stats["accuracy"],
                    "reliability": task_stats["reliability"],
                    "refusal_rate": task_stats["refusal_rate"],
                    "total": task_stats["total"],
                }
            )
    write_csv_table(output_dir / "comparison_table.csv", table_rows)

    print(json.dumps(
        {
            "split": {
                "total_common_ids": len(common_ids),
                "val_size": len(val_ids),
                "test_size": len(test_ids),
            },
            "best_thresholds": thresholds_payload,
            "overall": overall_payload,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
