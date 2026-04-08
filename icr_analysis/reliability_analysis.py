"""Reliability analysis: base vs refusal fine-tuned model.

Defines:
  reliable   = correct_confident + correct_refusal
  unreliable = hallucination    + false_refusal

Outputs:
  1. reliability_stats.png  — bar chart of accuracy / refusal / reliability rates
  2. reliability_probe.png  — probe AUROC (reliable vs unreliable) per dataset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RELIABLE_LABELS   = {"correct_confident", "correct_refusal"}
UNRELIABLE_LABELS = {"hallucination", "false_refusal"}
CORRECT_LABELS    = {"correct_confident"}
REFUSAL_LABELS    = {"correct_refusal", "false_refusal"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(path: Path, task_filter: Optional[str] = None):
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if task_filter and rec.get("task") != task_filter:
                continue
            records.append(rec)
    return records


def get_tasks(path: Path):
    tasks = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.add(json.loads(line).get("task"))
    return [None] + sorted(t for t in tasks if t)


def icr_vector(icr_scores) -> np.ndarray:
    return np.array([np.mean(layer) for layer in icr_scores], dtype=np.float32)


def build_Xy(records):
    X, y = [], []
    for rec in records:
        label = rec.get("label")
        if label in RELIABLE_LABELS:
            y.append(0)
        elif label in UNRELIABLE_LABELS:
            y.append(1)
        else:
            continue
        X.append(icr_vector(rec["icr_scores"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(records):
    total = len(records)
    if total == 0:
        return {}
    counts = {k: 0 for k in ("correct_confident", "correct_refusal",
                               "false_refusal", "hallucination")}
    for rec in records:
        lbl = rec.get("label")
        if lbl in counts:
            counts[lbl] += 1

    accuracy    = counts["correct_confident"] / total
    refusal     = (counts["correct_refusal"] + counts["false_refusal"]) / total
    reliability = (counts["correct_confident"] + counts["correct_refusal"]) / total
    return {
        "accuracy":    accuracy,
        "refusal":     refusal,
        "reliability": reliability,
        "total":       total,
        **counts,
    }


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def probe_cv(X, y, n_splits=5, seed=42):
    if len(X) == 0 or y.sum() == 0 or (y == 0).sum() == 0:
        return None
    if min(int(y.sum()), int((y == 0).sum())) < n_splits:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, C=1.0),
    )
    aurocs = []
    for train_idx, val_idx in skf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[val_idx])[:, 1]
        aurocs.append(roc_auc_score(y[val_idx], proba))
    return float(np.mean(aurocs)), float(np.std(aurocs))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_stats(stats_base, stats_rt, datasets, output_path: Path):
    metrics = [("accuracy", "Accuracy\n(correct answers only)"),
               ("refusal",  "Refusal Rate\n(all refusals)"),
               ("reliability", "Reliability\n(correct + correct_refusal)")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    colors = {"base": "#5B9BD5", "rtuning": "#ED7D31"}
    ds_labels = [d or "all" for d in datasets]

    for ax, (metric, title) in zip(axes, metrics):
        x = np.arange(len(datasets))
        w = 0.35
        for i, (model, stats_dict, color) in enumerate([
            ("Base", stats_base, colors["base"]),
            ("Refusal Fine-tuned", stats_rt, colors["rtuning"]),
        ]):
            vals = [stats_dict.get(d or "all", {}).get(metric, 0) for d in datasets]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=model,
                          color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() if d else "All" for d in ds_labels])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Rate")
        ax.legend(fontsize=8)

    fig.suptitle("Base vs Refusal Fine-tuned: Response Statistics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_probe(probe_base, probe_rt, datasets, output_path: Path):
    ds_labels = [d or "all" for d in datasets]
    x = np.arange(len(datasets))
    w = 0.35
    colors = {"base": "#5B9BD5", "rtuning": "#ED7D31"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (model, probe_dict, color) in enumerate([
        ("Base", probe_base, colors["base"]),
        ("Refusal Fine-tuned", probe_rt, colors["rtuning"]),
    ]):
        vals  = [probe_dict.get(d or "all", (0, 0))[0] for d in datasets]
        errs  = [probe_dict.get(d or "all", (0, 0))[1] for d in datasets]
        bars  = ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=4,
                       label=model, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Reliability Probe AUROC\n(unreliable vs reliable, 5-fold CV)",
                 fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() if d else "All" for d in ds_labels])
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("AUROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--icr_path_base",    required=True)
    p.add_argument("--icr_path_rtuning", required=True)
    p.add_argument("--output_dir",       default="icr_analysis/outputs/figures")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed",     type=int, default=42)
    args = p.parse_args()

    base_path = Path(args.icr_path_base)
    rt_path   = Path(args.icr_path_rtuning)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = get_tasks(base_path)

    stats_base, stats_rt   = {}, {}
    probe_base, probe_rt   = {}, {}

    for task in tasks:
        key = task or "all"
        recs_base = load_records(base_path, task)
        recs_rt   = load_records(rt_path,   task)

        stats_base[key] = compute_stats(recs_base)
        stats_rt[key]   = compute_stats(recs_rt)

        X_b, y_b = build_Xy(recs_base)
        X_r, y_r = build_Xy(recs_rt)

        res_b = probe_cv(X_b, y_b, args.n_splits, args.seed)
        res_r = probe_cv(X_r, y_r, args.n_splits, args.seed)

        probe_base[key] = res_b if res_b else (0.0, 0.0)
        probe_rt[key]   = res_r if res_r else (0.0, 0.0)

        print(f"[{key}] base reliability={stats_base[key]['reliability']:.3f} "
              f"probe_auroc={probe_base[key][0]:.3f} | "
              f"rtuning reliability={stats_rt[key]['reliability']:.3f} "
              f"probe_auroc={probe_rt[key][0]:.3f}")

    plot_stats(stats_base, stats_rt, tasks,
               out_dir / "reliability_stats.png")
    plot_probe(probe_base, probe_rt, tasks,
               out_dir / "reliability_probe.png")


if __name__ == "__main__":
    main()
