#!/usr/bin/env python3
"""Train and evaluate ICR probes on collected ICR scores.

Reads icr_scores.jsonl produced by collect_icr_scores.py, extracts
per-layer feature vectors, and trains logistic regression probes with
5-fold cross-validation.

Two binary classification tasks:
  - hallucination detection: hallucination (1) vs correct_confident (0)
  - refusal calibration:     false_refusal  (1) vs correct_refusal  (0)

Example
-------
python icr_analysis/train_probe.py \
    --icr_path icr_analysis/outputs/icr_scores.jsonl \
    --output_dir icr_analysis/outputs/probe_results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: F401

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

def icr_to_feature_vector(icr_scores: List[List[float]]) -> np.ndarray:
    """Mean-pool over response tokens → one scalar per layer → [num_layers]."""
    return np.array([np.mean(layer) for layer in icr_scores], dtype=np.float32)


def load_dataset(
    icr_path: Path,
    pos_label: str,
    neg_label: str,
    task_filter: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load samples matching pos/neg labels, return (X, y)."""
    X, y = [], []
    with icr_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if task_filter and rec.get("task") != task_filter:
                continue
            label = rec.get("label")
            if label == pos_label:
                y.append(1)
            elif label == neg_label:
                y.append(0)
            else:
                continue
            X.append(icr_to_feature_vector(rec["icr_scores"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, C=1.0),
    )

    aurocs, f1s, accs = [], [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_val)[:, 1]
        pred = clf.predict(X_val)

        aurocs.append(roc_auc_score(y_val, proba))
        f1s.append(f1_score(y_val, pred, zero_division=0))
        accs.append(accuracy_score(y_val, pred))

    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "n_splits": n_splits,
    }


def run_experiment(
    icr_path: Path,
    pos_label: str,
    neg_label: str,
    task_filter: Optional[str],
    n_splits: int,
    seed: int,
) -> Dict:
    X, y = load_dataset(icr_path, pos_label, neg_label, task_filter)
    if len(X) == 0:
        return {"error": "no samples found"}
    if y.sum() == 0 or (y == 0).sum() == 0:
        return {"error": "only one class present"}
    if len(X) < n_splits * 2:
        return {"error": f"too few samples ({len(X)}) for {n_splits}-fold CV"}

    metrics = cross_validate(X, y, n_splits=n_splits, seed=seed)
    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def cross_model_experiment(
    path_train: Path,
    path_test: Path,
    pos_label: str,
    neg_label: str,
    task_filter: Optional[str],
    seed: int,
) -> Dict:
    """Train probe on base model scores, test on R-Tuning scores (or vice versa)."""
    X_train, y_train = load_dataset(path_train, pos_label, neg_label, task_filter)
    X_test, y_test = load_dataset(path_test, pos_label, neg_label, task_filter)

    if len(X_train) == 0 or len(X_test) == 0:
        return {"error": "no samples found"}
    if y_train.sum() == 0 or (y_train == 0).sum() == 0:
        return {"error": "only one class in train"}
    if y_test.sum() == 0 or (y_test == 0).sum() == 0:
        return {"error": "only one class in test"}

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, C=1.0),
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = clf.predict(X_test)

    return {
        "auroc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "n_train_pos": int(y_train.sum()),
        "n_train_neg": int((y_train == 0).sum()),
        "n_test_pos": int(y_test.sum()),
        "n_test_neg": int((y_test == 0).sum()),
    }


def print_metrics(name: str, metrics: Dict) -> None:
    if "error" in metrics:
        print(f"  {name}: {metrics['error']}")
    elif "auroc_mean" in metrics:
        print(
            f"  {name}: "
            f"AUROC={metrics['auroc_mean']:.3f}±{metrics['auroc_std']:.3f}  "
            f"F1={metrics['f1_mean']:.3f}±{metrics['f1_std']:.3f}  "
            f"Acc={metrics['accuracy_mean']:.3f}±{metrics['accuracy_std']:.3f}  "
            f"(pos={metrics['n_pos']}, neg={metrics['n_neg']})"
        )
    else:
        print(
            f"  {name}: "
            f"AUROC={metrics['auroc']:.3f}  "
            f"F1={metrics['f1']:.3f}  "
            f"Acc={metrics['accuracy']:.3f}  "
            f"(train pos={metrics['n_train_pos']}, test pos={metrics['n_test_pos']})"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ICR probes on collected scores.")
    p.add_argument("--icr_path", default=None, help="Single-file mode: path to icr_scores.jsonl")
    p.add_argument("--icr_path_base", default=None, help="Comparison mode: base model scores")
    p.add_argument("--icr_path_rtuning", default=None, help="Comparison mode: R-Tuning model scores")
    p.add_argument("--output_dir", default=None, help="Directory to save results JSON")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _get_tasks(icr_path: Path) -> List[Optional[str]]:
    tasks = set()
    with icr_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.add(json.loads(line).get("task"))
    return [None] + sorted(t for t in tasks if t)


def main() -> None:
    args = parse_args()

    results = {}
    comparison_mode = args.icr_path_base and args.icr_path_rtuning

    # ---- Single-file CV experiments ----------------------------------------
    for path_arg, mode_name in [
        (args.icr_path, "single"),
        (args.icr_path_base, "base"),
        (args.icr_path_rtuning, "rtuning"),
    ]:
        if not path_arg:
            continue
        icr_path = Path(path_arg)
        tasks = _get_tasks(icr_path)

        print(f"\n=== [{mode_name}] Within-model CV ({icr_path.name}) ===")
        for task in tasks:
            task_label = task or "all"
            for pos, neg, exp_name in [
                ("hallucination", "correct_confident", "hallucination_detection"),
                ("false_refusal", "correct_refusal", "refusal_calibration"),
            ]:
                key = f"{mode_name}/{exp_name}/{task_label}"
                m = run_experiment(icr_path, pos, neg, task, args.n_splits, args.seed)
                results[key] = m
                print_metrics(f"{exp_name}/{task_label}", m)

    # ---- Cross-model experiments (train on base, test on rtuning) ----------
    if comparison_mode:
        base_path = Path(args.icr_path_base)
        rt_path = Path(args.icr_path_rtuning)
        tasks = _get_tasks(base_path)

        print(f"\n=== Cross-model: train on base, test on R-Tuning ===")
        for task in tasks:
            task_label = task or "all"
            for pos, neg, exp_name in [
                ("hallucination", "correct_confident", "hallucination_detection"),
                ("false_refusal", "correct_refusal", "refusal_calibration"),
            ]:
                key = f"cross_base_to_rtuning/{exp_name}/{task_label}"
                m = cross_model_experiment(base_path, rt_path, pos, neg, task, args.seed)
                results[key] = m
                print_metrics(f"{exp_name}/{task_label}", m)

        print(f"\n=== Cross-model: train on R-Tuning, test on base ===")
        for task in tasks:
            task_label = task or "all"
            for pos, neg, exp_name in [
                ("hallucination", "correct_confident", "hallucination_detection"),
                ("false_refusal", "correct_refusal", "refusal_calibration"),
            ]:
                key = f"cross_rtuning_to_base/{exp_name}/{task_label}"
                m = cross_model_experiment(rt_path, base_path, pos, neg, task, args.seed)
                results[key] = m
                print_metrics(f"{exp_name}/{task_label}", m)

    # ---- Save ---------------------------------------------------------------
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "probe_results.json"
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
