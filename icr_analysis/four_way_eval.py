#!/usr/bin/env python3
"""Four-way comparison of hallucination-mitigation strategies.

Conditions:
  base        – raw base model output, no refusal
  refuse_only – refusal fine-tuned (R-Tuning) model output
  icr_only    – base model answer + ICR probe decides refusal
  icr_refuse  – R-Tuning model answer + ICR probe decides refusal

Inputs:
  --icr_path_base      icr_scores_base.jsonl    (base model, base predictions)
  --icr_path_rtuning   icr_scores_rtuning.jsonl (rtuning model, rtuning predictions)

Both files must share the same sample IDs (same test set).
The probe is trained on base model ICR scores and applied to both models
to decide refusal (replacing each model's own refusal signal).

Example
-------
python icr_analysis/four_way_eval.py \
    --icr_path_base    icr_analysis/outputs/icr_scores_base.jsonl \
    --icr_path_rtuning icr_analysis/outputs/icr_scores_rtuning.jsonl \
    --output_dir       icr_analysis/outputs/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONDITIONS = ["base", "refuse_only", "icr_only", "icr_refuse"]
CONDITION_COLORS = {
    "base":        "#90CAF9",
    "refuse_only": "#1565C0",
    "icr_only":    "#A5D6A7",
    "icr_refuse":  "#2E7D32",
}
CONDITION_LABELS = {
    "base":        "Base\n(no refusal)",
    "refuse_only": "Refuse Only\n(R-Tuning)",
    "icr_only":    "ICR Only\n(base + probe)",
    "icr_refuse":  "ICR + Refuse\n(rtuning + probe)",
}

LABEL_COLORS = {
    "correct_confident": "#2196F3",
    "correct_refusal":   "#4CAF50",
    "false_refusal":     "#FF9800",
    "hallucination":     "#F44336",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def icr_vector(icr_scores: List[List[float]]) -> np.ndarray:
    """Mean over response tokens per layer → [num_layers]."""
    return np.array([np.mean(layer) for layer in icr_scores], dtype=np.float32)


def load_icr_records(path: Path, task_filter: Optional[str] = None) -> Dict[str, dict]:
    """Load all records keyed by sample ID."""
    records: Dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if task_filter and rec.get("task") != task_filter:
                continue
            sid = rec["id"]
            records[sid] = rec
    return records


def get_tasks(path: Path) -> List[Optional[str]]:
    tasks: set = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.add(json.loads(line).get("task"))
    return [None] + sorted(t for t in tasks if t)


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------

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
        "reliability": (counts["correct_confident"] + counts["correct_refusal"]) / total,
        "accuracy":    counts["correct_confident"] / total,
        "refusal_rate":(counts["correct_refusal"] + counts["false_refusal"]) / total,
        "hallucination_rate": counts["hallucination"] / total,
        "total":       total,
        **counts,
    }


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def build_probe_dataset(
    records: Dict[str, dict],
    ids: List[str],
    pos_labels: List[str],
    neg_labels: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract (X, y, ids) for probe training.

    pos_labels: labels assigned y=1 (probe should fire / refuse)
    neg_labels: labels assigned y=0 (probe should not fire)
    """
    X, y, used_ids = [], [], []
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
        used_ids.append(sid)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), used_ids


def train_probe(X: np.ndarray, y: np.ndarray, use_mlp: bool = False, seed: int = 42):
    """Train and return a fitted probe pipeline."""
    if use_mlp:
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.1,
            ),
        )
    else:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed, C=1.0),
        )
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Four-way evaluation
# ---------------------------------------------------------------------------

def evaluate_four_way(
    base_records: Dict[str, dict],
    rt_records: Dict[str, dict],
    probe_base,
    probe_rt,
    test_ids: List[str],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    For each test sample, compute outcome under each of the 4 conditions.

    base        – base is_correct, no refusal
    refuse_only – rtuning is_correct + rtuning is_refusal
    icr_only    – base is_correct + probe_base(base ICR)
    icr_refuse  – rtuning is_correct + (rtuning refusal OR probe_rt(rtuning ICR))

    Each probe is trained on its own model's ICR scores so the feature
    distributions match.  icr_refuse uses OR so the probe can only add
    refusals on top of rtuning's existing refusal behaviour.
    """
    results: Dict[str, List[str]] = {c: [] for c in CONDITIONS}

    for sid in test_ids:
        base_rec = base_records.get(sid)
        rt_rec   = rt_records.get(sid)
        if base_rec is None or rt_rec is None:
            continue

        base_correct = base_rec["is_correct_strict"]
        rt_correct   = rt_rec["is_correct_strict"]
        rt_refusal   = rt_rec["is_refusal"]

        base_vec = icr_vector(base_rec["icr_scores"]).reshape(1, -1)
        rt_vec   = icr_vector(rt_rec["icr_scores"]).reshape(1, -1)

        probe_base_flag = probe_base.predict_proba(base_vec)[0, 1] >= threshold
        probe_rt_flag   = probe_rt.predict_proba(rt_vec)[0, 1] >= threshold

        # icr_refuse: refuse if rtuning already refuses OR probe fires
        combined_refusal = rt_refusal or bool(probe_rt_flag)

        results["base"].append(derive_label(base_correct, False))
        results["refuse_only"].append(derive_label(rt_correct, rt_refusal))
        results["icr_only"].append(derive_label(base_correct, bool(probe_base_flag)))
        results["icr_refuse"].append(derive_label(rt_correct, combined_refusal))

    return {cond: compute_stats(labels) for cond, labels in results.items()}


# ---------------------------------------------------------------------------
# Probe cross-validation (for reporting probe quality)
# ---------------------------------------------------------------------------

def probe_auroc_cv(X: np.ndarray, y: np.ndarray, use_mlp: bool, seed: int = 42, n_splits: int = 5):
    if len(X) < n_splits * 2 or y.sum() == 0 or (y == 0).sum() == 0:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs = []
    for train_idx, val_idx in skf.split(X, y):
        clf = train_probe(X[train_idx], y[train_idx], use_mlp=use_mlp, seed=seed)
        proba = clf.predict_proba(X[val_idx])[:, 1]
        try:
            aurocs.append(roc_auc_score(y[val_idx], proba))
        except Exception:
            pass
    return float(np.mean(aurocs)) if aurocs else None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_four_way(
    stats_by_task: Dict[Optional[str], Dict[str, Dict[str, float]]],
    output_path: Path,
) -> None:
    """Bar chart: reliability per condition per task/dataset."""
    tasks = list(stats_by_task.keys())
    task_labels = [t or "All" for t in tasks]
    metric = "reliability"

    x = np.arange(len(tasks))
    n = len(CONDITIONS)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 3), 5))

    for i, cond in enumerate(CONDITIONS):
        vals = [stats_by_task[t].get(cond, {}).get(metric, 0) for t in tasks]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      label=CONDITION_LABELS[cond],
                      color=CONDITION_COLORS[cond], alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() if t else "All" for t in task_labels])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Reliability  (correct_confident + correct_refusal) / total")
    ax.set_title("Four-way Comparison: Hallucination Mitigation Strategies", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_stacked(
    stats_by_task: Dict[Optional[str], Dict[str, Dict[str, float]]],
    output_path: Path,
) -> None:
    """Stacked bar: shows composition of correct_confident / correct_refusal /
    false_refusal / hallucination for each condition × task."""
    tasks = list(stats_by_task.keys())
    n_tasks = len(tasks)

    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 4.5), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    stack_keys = ["correct_confident", "correct_refusal", "false_refusal", "hallucination"]

    for ax, task in zip(axes, tasks):
        task_label = task or "All"
        bottoms = np.zeros(len(CONDITIONS))
        for sk in stack_keys:
            vals = np.array([
                stats_by_task[task].get(cond, {}).get(sk, 0)
                / max(stats_by_task[task].get(cond, {}).get("total", 1), 1)
                for cond in CONDITIONS
            ])
            ax.bar(
                range(len(CONDITIONS)), vals, bottom=bottoms,
                color=LABEL_COLORS[sk], alpha=0.85, label=sk if ax == axes[0] else "",
            )
            bottoms += vals

        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(
            [CONDITION_LABELS[c].split("\n")[0] for c in CONDITIONS],
            rotation=15, ha="right", fontsize=8,
        )
        ax.set_title(task_label.capitalize() if task else "All", fontsize=10)
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("Fraction of samples")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("Outcome Breakdown: Four-way Comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_multi_metric(
    stats_by_task: Dict[Optional[str], Dict[str, Dict[str, float]]],
    output_path: Path,
) -> None:
    """Grid: accuracy / refusal_rate / reliability per condition per task."""
    metrics = [
        ("reliability",      "Reliability\n(correct + correct_refusal)"),
        ("accuracy",         "Accuracy\n(correct answers only)"),
        ("refusal_rate",     "Refusal Rate"),
        ("hallucination_rate", "Hallucination Rate"),
    ]
    tasks = list(stats_by_task.keys())
    x = np.arange(len(tasks))
    n = len(CONDITIONS)
    width = 0.8 / n

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        for i, cond in enumerate(CONDITIONS):
            vals = [stats_by_task[t].get(cond, {}).get(metric, 0) for t in tasks]
            offset = (i - (n - 1) / 2) * width
            ax.bar(x + offset, vals, width,
                   label=CONDITION_LABELS[cond].replace("\n", " "),
                   color=CONDITION_COLORS[cond], alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() if t else "All" for t in tasks])
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Four-way Comparison: Multiple Metrics", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Four-way ICR vs refusal comparison.")
    p.add_argument("--icr_path_base",    required=True,
                   help="icr_scores_base.jsonl (base model, base predictions)")
    p.add_argument("--icr_path_rtuning", required=True,
                   help="icr_scores_rtuning.jsonl (rtuning model, rtuning predictions)")
    p.add_argument("--output_dir", default="icr_analysis/outputs/figures")
    p.add_argument("--test_size", type=float, default=0.3,
                   help="Fraction of samples held out for evaluation")
    p.add_argument("--probe_threshold", type=float, default=0.5,
                   help="Probability threshold for probe to trigger refusal")
    p.add_argument("--use_mlp", action="store_true",
                   help="Use MLP probe instead of logistic regression")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.icr_path_base)
    rt_path   = Path(args.icr_path_rtuning)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all records
    base_all = load_icr_records(base_path)
    rt_all   = load_icr_records(rt_path)

    # Use only samples present in both files
    common_ids = sorted(set(base_all) & set(rt_all))
    print(f"Base records: {len(base_all)}  |  R-Tuning records: {len(rt_all)}")
    print(f"Common IDs: {len(common_ids)}")

    if len(common_ids) < 10:
        raise ValueError("Too few common sample IDs — check that both files share the same test set.")
    coverage = len(common_ids) / max(len(base_all), len(rt_all))
    if coverage < 0.9:
        print(f"[WARN] Only {coverage:.1%} of samples are common to both files "
              f"({len(common_ids)}/{max(len(base_all), len(rt_all))}). "
              f"Results reflect this subset only.")

    # Derive labels for base (for stratified split)
    base_labels = [base_all[sid]["label"] for sid in common_ids]

    # Stratified train/test split
    train_ids, test_ids = train_test_split(
        common_ids,
        test_size=args.test_size,
        stratify=base_labels,
        random_state=args.seed,
    )
    print(f"Train: {len(train_ids)}  |  Test: {len(test_ids)}")

    probe_type = "MLP" if args.use_mlp else "LR"

    # ---- Probe for base model (icr_only) ------------------------------------
    # Base model only has hallucination / correct_confident labels.
    X_base, y_base, _ = build_probe_dataset(
        base_all, train_ids,
        pos_labels=["hallucination"],
        neg_labels=["correct_confident"],
    )
    print(f"Base probe train set: {len(X_base)} samples "
          f"(pos={y_base.sum()}, neg={(y_base==0).sum()})")
    if len(X_base) < 10:
        raise ValueError("Too few base probe training samples.")

    auroc_base = probe_auroc_cv(X_base, y_base, use_mlp=args.use_mlp, seed=args.seed)
    print(f"Base probe ({probe_type}) 5-fold CV AUROC: "
          f"{auroc_base:.3f}" if auroc_base else "N/A")
    probe_base = train_probe(X_base, y_base, use_mlp=args.use_mlp, seed=args.seed)

    # ---- Probe for R-Tuning model (icr_refuse) ------------------------------
    # Trained on ALL rtuning samples: positive = answer wrong (hallucination +
    # correct_refusal), negative = answer right (correct_confident + false_refusal).
    # Combined with rtuning's own refusal via OR at eval time.
    X_rt, y_rt, _ = build_probe_dataset(
        rt_all, train_ids,
        pos_labels=["hallucination", "correct_refusal"],
        neg_labels=["correct_confident", "false_refusal"],
    )
    print(f"RTuning probe train set: {len(X_rt)} samples "
          f"(pos={y_rt.sum()}, neg={(y_rt==0).sum()})")
    if len(X_rt) < 10:
        raise ValueError("Too few rtuning probe training samples.")

    auroc_rt = probe_auroc_cv(X_rt, y_rt, use_mlp=args.use_mlp, seed=args.seed)
    print(f"RTuning probe ({probe_type}) 5-fold CV AUROC: "
          f"{auroc_rt:.3f}" if auroc_rt else "N/A")
    probe_rt = train_probe(X_rt, y_rt, use_mlp=args.use_mlp, seed=args.seed)

    # ---- Evaluate per task -----------------------------------------------
    tasks = get_tasks(base_path)
    stats_by_task: Dict[Optional[str], Dict[str, Dict[str, float]]] = {}

    print("\n" + "=" * 70)
    print(f"{'Task':<12}  {'Condition':<16}  "
          f"{'Reliability':>11}  {'Accuracy':>8}  {'RefusalR':>8}  {'Hallu':>8}")
    print("-" * 70)

    for task in tasks:
        # Filter test_ids to this task
        if task is None:
            task_test_ids = test_ids
        else:
            task_test_ids = [sid for sid in test_ids
                             if base_all[sid].get("task") == task]

        if not task_test_ids:
            continue

        stats = evaluate_four_way(base_all, rt_all, probe_base, probe_rt,
                                  task_test_ids, threshold=args.probe_threshold)
        stats_by_task[task] = stats

        task_label = task or "all"
        for cond in CONDITIONS:
            s = stats.get(cond, {})
            print(f"{task_label:<12}  {cond:<16}  "
                  f"{s.get('reliability', 0):>11.3f}  "
                  f"{s.get('accuracy', 0):>8.3f}  "
                  f"{s.get('refusal_rate', 0):>8.3f}  "
                  f"{s.get('hallucination_rate', 0):>8.3f}")
        print()

    # ---- Save JSON results -----------------------------------------------
    results_path = out_dir.parent / "four_way_results.json"
    serializable = {
        str(k) if k else "all": v
        for k, v in stats_by_task.items()
    }
    results_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Results saved: {results_path}")

    # ---- Plots -----------------------------------------------------------
    plot_four_way(stats_by_task, out_dir / "four_way_reliability.png")
    plot_stacked(stats_by_task, out_dir / "four_way_stacked.png")
    plot_multi_metric(stats_by_task, out_dir / "four_way_metrics.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
