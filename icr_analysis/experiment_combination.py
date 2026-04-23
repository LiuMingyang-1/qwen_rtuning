#!/usr/bin/env python3
"""Experiment: threshold sweep (route 1) and AND combination (route 2).

Does NOT overwrite any existing result files. Prints results to stdout only.

Usage:
    python icr_analysis/experiment_combination.py \
        --icr_path_base    icr_analysis/outputs/outputs/icr_scores_base.jsonl \
        --icr_path_rtuning icr_analysis/outputs/outputs/icr_scores_rtuning.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Shared helpers (mirrors four_way_eval.py, no import dependency)
# ---------------------------------------------------------------------------

def icr_vector(icr_scores) -> np.ndarray:
    return np.array([np.mean(layer) for layer in icr_scores], dtype=np.float32)


def load_records(path: Path) -> Dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["id"]] = rec
    return records


def derive_label(is_correct: bool, is_refusal: bool) -> str:
    if is_correct and not is_refusal:
        return "correct_confident"
    if is_correct and is_refusal:
        return "false_refusal"
    if not is_correct and is_refusal:
        return "correct_refusal"
    return "hallucination"


def compute_stats(labels: List[str]) -> Dict[str, float]:
    total = len(labels)
    if total == 0:
        return {}
    counts = {k: labels.count(k) for k in
              ("correct_confident", "correct_refusal", "false_refusal", "hallucination")}
    return {
        "reliability":        (counts["correct_confident"] + counts["correct_refusal"]) / total,
        "accuracy":           counts["correct_confident"] / total,
        "refusal_rate":       (counts["correct_refusal"] + counts["false_refusal"]) / total,
        "hallucination_rate": counts["hallucination"] / total,
        **counts,
    }


def build_dataset(records: Dict[str, dict], ids: List[str],
                  pos_labels: List[str], neg_labels: List[str]):
    X, y = [], []
    for sid in ids:
        rec = records.get(sid)
        if rec is None:
            continue
        label = rec.get("label")
        if label in pos_labels:
            y.append(1)
        elif label in neg_labels:
            y.append(0)
        else:
            continue
        X.append(icr_vector(rec["icr_scores"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_probe(X, y, seed=42):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, random_state=seed, C=1.0))
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Evaluation under a given combination strategy
# ---------------------------------------------------------------------------

def evaluate(base_records, rt_records, probe_base, probe_rt,
             test_ids, threshold, combination, task_filter=None):
    """
    combination: 'or' | 'and'
      OR  – refuse if rtuning refuses OR probe fires
      AND – refuse if rtuning refuses AND probe fires
    """
    labels_base = []
    labels_refuse_only = []
    labels_icr_only = []
    labels_icr_combined = []

    for sid in test_ids:
        base_rec = base_records.get(sid)
        rt_rec   = rt_records.get(sid)
        if base_rec is None or rt_rec is None:
            continue
        if task_filter and base_rec.get("task") != task_filter:
            continue

        base_correct = base_rec["is_correct_strict"]
        rt_correct   = rt_rec["is_correct_strict"]
        rt_refusal   = rt_rec["is_refusal"]

        base_vec = icr_vector(base_rec["icr_scores"]).reshape(1, -1)
        rt_vec   = icr_vector(rt_rec["icr_scores"]).reshape(1, -1)

        probe_base_flag = probe_base.predict_proba(base_vec)[0, 1] >= threshold
        probe_rt_flag   = probe_rt.predict_proba(rt_vec)[0, 1] >= threshold

        if combination == "or":
            combined = rt_refusal or bool(probe_rt_flag)
        else:  # and
            combined = rt_refusal and bool(probe_rt_flag)

        labels_base.append(derive_label(base_correct, False))
        labels_refuse_only.append(derive_label(rt_correct, rt_refusal))
        labels_icr_only.append(derive_label(base_correct, bool(probe_base_flag)))
        labels_icr_combined.append(derive_label(rt_correct, combined))

    return {
        "base":         compute_stats(labels_base),
        "refuse_only":  compute_stats(labels_refuse_only),
        "icr_only":     compute_stats(labels_icr_only),
        "icr_combined": compute_stats(labels_icr_combined),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--icr_path_base",    required=True)
    p.add_argument("--icr_path_rtuning", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    base_all = load_records(Path(args.icr_path_base))
    rt_all   = load_records(Path(args.icr_path_rtuning))

    common_ids  = sorted(set(base_all) & set(rt_all))
    base_labels = [base_all[sid]["label"] for sid in common_ids]
    train_ids, test_ids = train_test_split(
        common_ids, test_size=args.test_size,
        stratify=base_labels, random_state=args.seed,
    )
    print(f"Common: {len(common_ids)}  Train: {len(train_ids)}  Test: {len(test_ids)}\n")

    # Train probes (same as four_way_eval.py)
    X_base, y_base = build_dataset(base_all, train_ids,
                                   pos_labels=["hallucination"],
                                   neg_labels=["correct_confident"])
    X_rt,   y_rt   = build_dataset(rt_all,   train_ids,
                                   pos_labels=["hallucination", "correct_refusal"],
                                   neg_labels=["correct_confident", "false_refusal"])
    probe_base = train_probe(X_base, y_base, args.seed)
    probe_rt   = train_probe(X_rt,   y_rt,   args.seed)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for combination in ("or", "and"):
        print("=" * 72)
        print(f"Combination: {combination.upper()}")
        print("=" * 72)
        print(f"{'Thresh':>7}  {'Condition':<16}  "
              f"{'Reliability':>11}  {'Accuracy':>8}  {'RefusalR':>8}  {'Hallu':>8}")
        print("-" * 72)

        for threshold in thresholds:
            stats = evaluate(base_all, rt_all, probe_base, probe_rt,
                             test_ids, threshold, combination)

            # Print baseline rows only once (at threshold=0.5 for reference)
            if threshold == 0.5:
                for cond in ("base", "refuse_only", "icr_only"):
                    s = stats[cond]
                    print(f"{'(ref)':>7}  {cond:<16}  "
                          f"{s.get('reliability',0):>11.3f}  "
                          f"{s.get('accuracy',0):>8.3f}  "
                          f"{s.get('refusal_rate',0):>8.3f}  "
                          f"{s.get('hallucination_rate',0):>8.3f}")
                print("-" * 72)

            s = stats["icr_combined"]
            marker = " ◀" if s.get("reliability", 0) > stats["refuse_only"].get("reliability", 0) else ""
            print(f"{threshold:>7.1f}  {'icr_combined':<16}  "
                  f"{s.get('reliability',0):>11.3f}  "
                  f"{s.get('accuracy',0):>8.3f}  "
                  f"{s.get('refusal_rate',0):>8.3f}  "
                  f"{s.get('hallucination_rate',0):>8.3f}{marker}")

        print()

    # Per-task breakdown at the most interesting thresholds
    tasks = [("all", None), ("hotpotqa", "hotpotqa"), ("pararel", "pararel")]
    interesting = {"or": [0.6, 0.7], "and": [0.4, 0.5]}

    print("=" * 72)
    print("Per-task breakdown at selected thresholds")
    print("=" * 72)

    for combination, selected_thresholds in interesting.items():
        for threshold in selected_thresholds:
            print(f"\n[{combination.upper()}  threshold={threshold}]")
            print(f"{'Task':<10}  {'Condition':<16}  "
                  f"{'Reliability':>11}  {'Accuracy':>8}  {'RefusalR':>8}  {'Hallu':>8}")
            print("-" * 68)
            for task_label, task_filter in tasks:
                stats = evaluate(base_all, rt_all, probe_base, probe_rt,
                                 test_ids, threshold, combination,
                                 task_filter=task_filter)
                for cond in ("refuse_only", "icr_only", "icr_combined"):
                    s = stats[cond]
                    marker = " ◀" if (cond == "icr_combined" and
                                      s.get("reliability", 0) > stats["refuse_only"].get("reliability", 0)) else ""
                    print(f"{task_label:<10}  {cond:<16}  "
                          f"{s.get('reliability',0):>11.3f}  "
                          f"{s.get('accuracy',0):>8.3f}  "
                          f"{s.get('refusal_rate',0):>8.3f}  "
                          f"{s.get('hallucination_rate',0):>8.3f}{marker}")
                print()


if __name__ == "__main__":
    main()
