#!/usr/bin/env python3
"""Threshold sweep experiment: OR vs AND combination of R-Tuning + ICR probe.

Sweeps probe threshold from 0.1 to 0.95 for both OR and AND strategies,
plots reliability curves, and saves numerical results.

Outputs (in --output_dir):
  threshold_sweep_results.json   — full numerical results
  threshold_sweep.png            — reliability curves (main figure)
  threshold_sweep_hallu.png      — hallucination rate curves

Usage:
    python icr_analysis/threshold_sweep.py \
        --icr_path_base    icr_analysis/outputs/outputs/icr_scores_base.jsonl \
        --icr_path_rtuning icr_analysis/outputs/outputs/icr_scores_rtuning.jsonl \
        --output_dir       icr_analysis/outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
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
    if is_correct and not is_refusal:     return "correct_confident"
    if is_correct and is_refusal:         return "false_refusal"
    if not is_correct and is_refusal:     return "correct_refusal"
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


def build_dataset(records, ids, pos_labels, neg_labels):
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


def sweep(base_all, rt_all, probe_base, probe_rt,
          test_ids, thresholds, combination, task_filter=None):
    """Return list of stats dicts, one per threshold."""
    # Pre-compute probabilities once
    base_probs, rt_probs = {}, {}
    for sid in test_ids:
        base_rec = base_all.get(sid)
        rt_rec   = rt_all.get(sid)
        if base_rec is None or rt_rec is None:
            continue
        if task_filter and base_rec.get("task") != task_filter:
            continue
        base_probs[sid] = probe_base.predict_proba(
            icr_vector(base_rec["icr_scores"]).reshape(1, -1))[0, 1]
        rt_probs[sid]   = probe_rt.predict_proba(
            icr_vector(rt_rec["icr_scores"]).reshape(1, -1))[0, 1]

    results = []
    for threshold in thresholds:
        labels_combined = []
        for sid in test_ids:
            base_rec = base_all.get(sid)
            rt_rec   = rt_all.get(sid)
            if base_rec is None or rt_rec is None:
                continue
            if task_filter and base_rec.get("task") != task_filter:
                continue
            if sid not in base_probs:
                continue

            rt_correct = rt_rec["is_correct_strict"]
            rt_refusal = rt_rec["is_refusal"]
            probe_flag = rt_probs[sid] >= threshold

            if combination == "or":
                combined = rt_refusal or bool(probe_flag)
            else:
                combined = rt_refusal and bool(probe_flag)

            labels_combined.append(derive_label(rt_correct, combined))

        stats = compute_stats(labels_combined)
        stats["threshold"] = threshold
        results.append(stats)
    return results


def baseline_stats(base_all, rt_all, test_ids, task_filter=None):
    """Compute fixed baselines (base, refuse_only, icr_only at t=0.5)."""
    lb, lr, li = [], [], []
    # Need probe_base for icr_only — reuse probe from caller via closure not possible,
    # so return raw counts instead and let caller fill icr_only separately.
    for sid in test_ids:
        base_rec = base_all.get(sid)
        rt_rec   = rt_all.get(sid)
        if base_rec is None or rt_rec is None:
            continue
        if task_filter and base_rec.get("task") != task_filter:
            continue
        lb.append(derive_label(base_rec["is_correct_strict"], False))
        lr.append(derive_label(rt_rec["is_correct_strict"], rt_rec["is_refusal"]))
    return compute_stats(lb), compute_stats(lr)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "or":          "#E53935",   # red
    "and":         "#1E88E5",   # blue
    "refuse_only": "#FB8C00",   # orange (dashed baseline)
    "icr_only":    "#43A047",   # green  (dashed baseline)
    "base":        "#9E9E9E",   # grey   (dashed baseline)
}


def plot_sweep(sweep_or, sweep_and, baselines, thresholds,
               metric, ylabel, title, output_path, task_label="All"):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Baseline horizontals
    for name, val, ls in [
        ("base",        baselines["base"].get(metric, 0),        ":"),
        ("refuse_only", baselines["refuse_only"].get(metric, 0), "--"),
        ("icr_only",    baselines["icr_only"].get(metric, 0),    "-."),
    ]:
        ax.axhline(val, color=COLORS[name], linestyle=ls, linewidth=1.5,
                   label=f"{name} ({val:.3f})", alpha=0.8)

    # Sweep curves
    or_vals  = [s.get(metric, 0) for s in sweep_or]
    and_vals = [s.get(metric, 0) for s in sweep_and]
    ax.plot(thresholds, or_vals,  color=COLORS["or"],  marker="o", markersize=5,
            linewidth=2, label="OR combination")
    ax.plot(thresholds, and_vals, color=COLORS["and"], marker="s", markersize=5,
            linewidth=2, label="AND combination")

    # Annotate best AND point
    best_idx = int(np.argmax(and_vals))
    ax.annotate(f"{and_vals[best_idx]:.3f}",
                xy=(thresholds[best_idx], and_vals[best_idx]),
                xytext=(thresholds[best_idx] + 0.03, and_vals[best_idx] + 0.008),
                fontsize=8, color=COLORS["and"],
                arrowprops=dict(arrowstyle="->", color=COLORS["and"], lw=1))

    ax.set_xlabel("Probe Threshold", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{title}  [{task_label}]", fontsize=12)
    ax.set_xlim(thresholds[0] - 0.02, thresholds[-1] + 0.02)
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_multi_task(all_sweep_or, all_sweep_and, all_baselines,
                    thresholds, task_labels, metric, ylabel, title, output_path):
    """One subplot per task."""
    n = len(task_labels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, task_label, sweep_or, sweep_and, baselines in zip(
            axes, task_labels, all_sweep_or, all_sweep_and, all_baselines):

        for name, val, ls in [
            ("base",        baselines["base"].get(metric, 0),        ":"),
            ("refuse_only", baselines["refuse_only"].get(metric, 0), "--"),
            ("icr_only",    baselines["icr_only"].get(metric, 0),    "-."),
        ]:
            ax.axhline(val, color=COLORS[name], linestyle=ls, linewidth=1.3,
                       label=f"{name} ({val:.3f})", alpha=0.8)

        or_vals  = [s.get(metric, 0) for s in sweep_or]
        and_vals = [s.get(metric, 0) for s in sweep_and]
        ax.plot(thresholds, or_vals,  color=COLORS["or"],  marker="o", markersize=4,
                linewidth=2, label="OR")
        ax.plot(thresholds, and_vals, color=COLORS["and"], marker="s", markersize=4,
                linewidth=2, label="AND")

        best_idx = int(np.argmax(and_vals))
        ax.annotate(f"{and_vals[best_idx]:.3f}",
                    xy=(thresholds[best_idx], and_vals[best_idx]),
                    xytext=(thresholds[best_idx] + 0.04, and_vals[best_idx] + 0.008),
                    fontsize=7.5, color=COLORS["and"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["and"], lw=0.8))

        ax.set_title(task_label.capitalize(), fontsize=11)
        ax.set_xlabel("Threshold", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=6.5, loc="best")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--icr_path_base",    required=True)
    p.add_argument("--icr_path_rtuning", required=True)
    p.add_argument("--output_dir", default="icr_analysis/outputs")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--test_size",  type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir  = Path(args.output_dir)
    fig_dir  = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    base_all = load_records(Path(args.icr_path_base))
    rt_all   = load_records(Path(args.icr_path_rtuning))

    common_ids  = sorted(set(base_all) & set(rt_all))
    base_labels = [base_all[sid]["label"] for sid in common_ids]
    train_ids, test_ids = train_test_split(
        common_ids, test_size=args.test_size,
        stratify=base_labels, random_state=args.seed,
    )
    print(f"Common: {len(common_ids)}  Train: {len(train_ids)}  Test: {len(test_ids)}")

    # Train probes
    X_base, y_base = build_dataset(base_all, train_ids,
                                   pos_labels=["hallucination"],
                                   neg_labels=["correct_confident"])
    X_rt, y_rt     = build_dataset(rt_all,   train_ids,
                                   pos_labels=["hallucination", "correct_refusal"],
                                   neg_labels=["correct_confident", "false_refusal"])
    probe_base = train_probe(X_base, y_base, args.seed)
    probe_rt   = train_probe(X_rt,   y_rt,   args.seed)
    print("Probes trained.")

    thresholds = [round(t, 2) for t in np.arange(0.1, 0.96, 0.05)]

    tasks = [
        ("all",      None),
        ("hotpotqa", "hotpotqa"),
        ("pararel",  "pararel"),
    ]

    # Compute icr_only baseline with t=0.5
    def icr_only_stats(task_filter):
        labels = []
        for sid in test_ids:
            base_rec = base_all.get(sid)
            rt_rec   = rt_all.get(sid)
            if base_rec is None or rt_rec is None:
                continue
            if task_filter and base_rec.get("task") != task_filter:
                continue
            prob = probe_base.predict_proba(
                icr_vector(base_rec["icr_scores"]).reshape(1, -1))[0, 1]
            labels.append(derive_label(base_rec["is_correct_strict"], prob >= 0.5))
        return compute_stats(labels)

    all_results = {}
    all_sweep_or, all_sweep_and, all_baselines_list = [], [], []
    task_labels_list = []

    for task_label, task_filter in tasks:
        print(f"\nSweeping task={task_label} ...")
        s_or  = sweep(base_all, rt_all, probe_base, probe_rt,
                      test_ids, thresholds, "or",  task_filter)
        s_and = sweep(base_all, rt_all, probe_base, probe_rt,
                      test_ids, thresholds, "and", task_filter)

        base_stat, refonly_stat = baseline_stats(base_all, rt_all, test_ids, task_filter)
        icr_stat = icr_only_stats(task_filter)

        baselines = {
            "base":        base_stat,
            "refuse_only": refonly_stat,
            "icr_only":    icr_stat,
        }

        all_results[task_label] = {
            "baselines": {k: v for k, v in baselines.items()},
            "or":  s_or,
            "and": s_and,
        }
        all_sweep_or.append(s_or)
        all_sweep_and.append(s_and)
        all_baselines_list.append(baselines)
        task_labels_list.append(task_label)

        # Best AND threshold
        best_and_idx = int(np.argmax([s.get("reliability", 0) for s in s_and]))
        best_and_t   = thresholds[best_and_idx]
        best_and_rel = s_and[best_and_idx].get("reliability", 0)
        ref_rel      = refonly_stat.get("reliability", 0)
        delta        = best_and_rel - ref_rel
        print(f"  [{task_label}] Best AND: threshold={best_and_t}  "
              f"reliability={best_and_rel:.3f}  "
              f"(refuse_only={ref_rel:.3f}, Δ={delta:+.3f})")

    # Save JSON
    json_path = out_dir / "threshold_sweep_results.json"
    json_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResults saved: {json_path}")

    # Plot: reliability, all tasks combined (3 subplots)
    plot_multi_task(
        all_sweep_or, all_sweep_and, all_baselines_list,
        thresholds, task_labels_list,
        metric="reliability",
        ylabel="Reliability",
        title="Threshold Sweep: OR vs AND — Reliability",
        output_path=fig_dir / "threshold_sweep_reliability.png",
    )

    # Plot: hallucination rate
    plot_multi_task(
        all_sweep_or, all_sweep_and, all_baselines_list,
        thresholds, task_labels_list,
        metric="hallucination_rate",
        ylabel="Hallucination Rate",
        title="Threshold Sweep: OR vs AND — Hallucination Rate",
        output_path=fig_dir / "threshold_sweep_hallu.png",
    )

    # Single "all" reliability plot (cleaner version for thesis)
    plot_sweep(
        all_results["all"]["or"], all_results["all"]["and"],
        all_results["all"]["baselines"],
        thresholds,
        metric="reliability",
        ylabel="Reliability",
        title="OR vs AND Threshold Sweep — Reliability",
        output_path=fig_dir / "threshold_sweep_all.png",
        task_label="All tasks",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
