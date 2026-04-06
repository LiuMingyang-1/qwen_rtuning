#!/usr/bin/env python3
"""Visualize ICR score distributions and base vs R-Tuning comparison.

Two modes:
  Single-file mode (--icr_path):
    - Per-layer violin distributions by label
    - Per-layer AUROC for hallucination detection & false-refusal detection
    - PCA scatter by label

  Comparison mode (--icr_path_base + --icr_path_rtuning):
    - Delta ICR per layer: how much each label's mean score shifted after R-Tuning
    - Side-by-side per-layer AUROC (base vs rtuning)
    - PCA comparison for hallucination samples

Example
-------
# single-file
python icr_analysis/analyze.py \
    --icr_path icr_analysis/outputs/icr_scores_rtuning.jsonl \
    --output_dir icr_analysis/outputs/figures

# comparison
python icr_analysis/analyze.py \
    --icr_path_base    icr_analysis/outputs/icr_scores_base.jsonl \
    --icr_path_rtuning icr_analysis/outputs/icr_scores_rtuning.jsonl \
    --output_dir icr_analysis/outputs/figures
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
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

LABEL_COLORS = {
    "correct_confident": "#2196F3",
    "correct_refusal":   "#4CAF50",
    "false_refusal":     "#FF9800",
    "hallucination":     "#F44336",
}
LABEL_ORDER = ["correct_confident", "correct_refusal", "false_refusal", "hallucination"]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_icr_data(
    icr_path: Path,
    task_filter: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Return dict label → feature matrix [N, num_layers] (mean over tokens)."""
    buckets: Dict[str, List[np.ndarray]] = {k: [] for k in LABEL_ORDER}
    with icr_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if task_filter and rec.get("task") != task_filter:
                continue
            label = rec.get("label")
            if label not in buckets:
                continue
            vec = np.array([np.mean(layer) for layer in rec["icr_scores"]], dtype=np.float32)
            buckets[label].append(vec)
    return {k: np.array(v) for k, v in buckets.items() if v}


def load_icr_by_id(
    icr_path: Path,
    task_filter: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Return dict sample_id → feature vector [num_layers]."""
    result = {}
    with icr_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if task_filter and rec.get("task") != task_filter:
                continue
            vec = np.array([np.mean(layer) for layer in rec["icr_scores"]], dtype=np.float32)
            result[rec["id"]] = {"vec": vec, "label": rec.get("label"), "task": rec.get("task")}
    return result


# -----------------------------------------------------------------------------
# Single-file plots
# -----------------------------------------------------------------------------

def plot_layer_distributions(
    data: Dict[str, np.ndarray],
    output_path: Path,
    labels_to_plot: Optional[List[str]] = None,
    title: str = "ICR Score per Layer",
) -> None:
    if labels_to_plot is None:
        labels_to_plot = [k for k in LABEL_ORDER if k in data]

    num_layers = next(iter(data.values())).shape[1]
    fig, ax = plt.subplots(figsize=(14, 4))

    n = len(labels_to_plot)
    group_width = 0.8
    label_width = group_width / n

    for li, label in enumerate(labels_to_plot):
        if label not in data or len(data[label]) == 0:
            continue
        matrix = data[label]
        color = LABEL_COLORS.get(label, "gray")
        offset = (li - (n - 1) / 2) * label_width
        positions = [l + 1 + offset for l in range(num_layers)]

        parts = ax.violinplot(
            [matrix[:, l] for l in range(num_layers)],
            positions=positions,
            widths=label_width * 0.9,
            showmedians=True,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color(color)
        parts["cmedians"].set_linewidth(1.5)

    ax.set_xlabel("Layer")
    ax.set_ylabel("ICR Score (mean over tokens)")
    ax.set_title(title)
    ax.set_xticks(range(1, num_layers + 1))
    ax.set_xticklabels([str(i) for i in range(1, num_layers + 1)], fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=LABEL_COLORS.get(l, "gray"), alpha=0.6)
               for l in labels_to_plot if l in data]
    ax.legend(handles, [l for l in labels_to_plot if l in data], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_layer_auroc(
    data: Dict[str, np.ndarray],
    output_path: Path,
    pos_label: str = "hallucination",
    neg_label: str = "correct_confident",
    title: Optional[str] = None,
) -> None:
    if pos_label not in data or neg_label not in data:
        print(f"[WARN] Missing {pos_label} or {neg_label}, skipping AUROC plot")
        return

    pos, neg = data[pos_label], data[neg_label]
    num_layers = pos.shape[1]
    X = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

    aurocs = []
    for l in range(num_layers):
        try:
            a = roc_auc_score(y, X[:, l])
            aurocs.append(max(a, 1 - a))
        except Exception:
            aurocs.append(0.5)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(1, num_layers + 1), aurocs, color="#2196F3", alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(title or f"Per-layer AUROC: {pos_label} vs {neg_label}")
    ax.set_xticks(range(1, num_layers + 1))
    ax.set_xticklabels([str(i) for i in range(1, num_layers + 1)], fontsize=7)
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_pca(
    data: Dict[str, np.ndarray],
    output_path: Path,
    labels_to_plot: Optional[List[str]] = None,
    title: str = "PCA of ICR Feature Vectors",
) -> None:
    if labels_to_plot is None:
        labels_to_plot = [k for k in LABEL_ORDER if k in data]

    arrays = [data[l] for l in labels_to_plot if l in data and len(data[l]) > 0]
    used = [l for l in labels_to_plot if l in data and len(data[l]) > 0]
    if not arrays:
        return

    X = np.concatenate(arrays, axis=0)
    group_ids = np.concatenate([np.full(len(data[l]), i) for i, l in enumerate(used)])

    X_2d = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, label in enumerate(used):
        mask = group_ids == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=LABEL_COLORS.get(label, "gray"), label=label, alpha=0.4, s=10)
    ax.set_title(title)
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# -----------------------------------------------------------------------------
# Comparison plots (base vs rtuning)
# -----------------------------------------------------------------------------

def _compute_layer_aurocs(
    data: Dict[str, np.ndarray],
    pos_label: str,
    neg_label: str,
) -> Optional[List[float]]:
    if pos_label not in data or neg_label not in data:
        return None
    pos, neg = data[pos_label], data[neg_label]
    num_layers = pos.shape[1]
    X = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    aurocs = []
    for l in range(num_layers):
        try:
            a = roc_auc_score(y, X[:, l])
            aurocs.append(max(a, 1 - a))
        except Exception:
            aurocs.append(0.5)
    return aurocs


def plot_auroc_comparison(
    data_base: Dict[str, np.ndarray],
    data_rtuning: Dict[str, np.ndarray],
    output_path: Path,
    pos_label: str,
    neg_label: str,
    title: Optional[str] = None,
) -> None:
    aurocs_base = _compute_layer_aurocs(data_base, pos_label, neg_label)
    aurocs_rt = _compute_layer_aurocs(data_rtuning, pos_label, neg_label)
    if aurocs_base is None or aurocs_rt is None:
        print(f"[WARN] Missing data for comparison: {pos_label} vs {neg_label}")
        return

    num_layers = len(aurocs_base)
    x = np.arange(1, num_layers + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(x - width / 2, aurocs_base, width, label="Base", color="#90CAF9", alpha=0.9)
    ax.bar(x + width / 2, aurocs_rt,   width, label="R-Tuning", color="#1565C0", alpha=0.9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(title or f"Per-layer AUROC Comparison: {pos_label} vs {neg_label}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=7)
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_delta_icr(
    data_base: Dict[str, np.ndarray],
    data_rtuning: Dict[str, np.ndarray],
    output_path: Path,
    labels_to_plot: Optional[List[str]] = None,
    title: str = "ICR Score Delta (R-Tuning − Base) per Layer",
) -> None:
    """For each label, plot mean(rtuning) - mean(base) per layer."""
    if labels_to_plot is None:
        labels_to_plot = [k for k in LABEL_ORDER if k in data_base and k in data_rtuning]

    available = [l for l in labels_to_plot if l in data_base and l in data_rtuning]
    if not available:
        return

    num_layers = next(iter(data_base.values())).shape[1]
    fig, ax = plt.subplots(figsize=(12, 4))

    for label in available:
        mean_base = data_base[label].mean(axis=0)
        mean_rt = data_rtuning[label].mean(axis=0)
        delta = mean_rt - mean_base
        ax.plot(range(1, num_layers + 1), delta,
                marker="o", markersize=4,
                color=LABEL_COLORS.get(label, "gray"),
                label=label)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("ΔICR Score")
    ax.set_title(title)
    ax.set_xticks(range(1, num_layers + 1))
    ax.set_xticklabels([str(i) for i in range(1, num_layers + 1)], fontsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_pca_comparison(
    data_base: Dict[str, np.ndarray],
    data_rtuning: Dict[str, np.ndarray],
    output_path: Path,
    label: str = "hallucination",
    title: Optional[str] = None,
) -> None:
    """PCA on hallucination samples from both models, coloured by model."""
    if label not in data_base or label not in data_rtuning:
        print(f"[WARN] Label '{label}' missing in one of the datasets, skipping PCA comparison")
        return

    base_vecs = data_base[label]
    rt_vecs = data_rtuning[label]
    X = np.concatenate([base_vecs, rt_vecs], axis=0)
    groups = ["Base"] * len(base_vecs) + ["R-Tuning"] * len(rt_vecs)

    X_2d = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, color in [("Base", "#FF9800"), ("R-Tuning", "#1565C0")]:
        mask = np.array(groups) == name
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, label=name, alpha=0.4, s=10)
    ax.set_title(title or f"PCA of '{label}' samples: Base vs R-Tuning")
    ax.legend(fontsize=9, markerscale=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize ICR score analysis.")
    p.add_argument("--icr_path", default=None, help="Single-file mode: path to icr_scores.jsonl")
    p.add_argument("--icr_path_base", default=None, help="Comparison mode: base model ICR scores")
    p.add_argument("--icr_path_rtuning", default=None, help="Comparison mode: R-Tuning model ICR scores")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--task", default=None, help="Filter to a specific task")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.task}" if args.task else "_all"

    comparison_mode = args.icr_path_base and args.icr_path_rtuning
    single_mode = args.icr_path

    if not comparison_mode and not single_mode:
        raise ValueError("Provide either --icr_path or both --icr_path_base and --icr_path_rtuning")

    # ---- Single-file mode -----------------------------------------------
    if single_mode:
        data = load_icr_data(Path(args.icr_path), task_filter=args.task)
        print(f"Loaded: { {k: len(v) for k, v in data.items()} }")

        plot_layer_distributions(data, output_dir / f"layer_dist{suffix}.png",
            title=f"ICR Score Distributions per Layer ({args.task or 'all'})")

        plot_layer_distributions(data, output_dir / f"layer_dist_hallu_correct{suffix}.png",
            labels_to_plot=["hallucination", "correct_confident"],
            title=f"ICR: Hallucination vs Correct+Confident ({args.task or 'all'})")

        plot_layer_auroc(data, output_dir / f"layer_auroc_hallucination{suffix}.png",
            pos_label="hallucination", neg_label="correct_confident",
            title=f"Per-layer AUROC: Hallucination Detection ({args.task or 'all'})")

        plot_layer_auroc(data, output_dir / f"layer_auroc_false_refusal{suffix}.png",
            pos_label="false_refusal", neg_label="correct_refusal",
            title=f"Per-layer AUROC: False Refusal Detection ({args.task or 'all'})")

        plot_pca(data, output_dir / f"pca{suffix}.png",
            title=f"PCA of ICR Feature Vectors ({args.task or 'all'})")

    # ---- Comparison mode ------------------------------------------------
    if comparison_mode:
        data_base = load_icr_data(Path(args.icr_path_base), task_filter=args.task)
        data_rt = load_icr_data(Path(args.icr_path_rtuning), task_filter=args.task)
        print(f"Base:    { {k: len(v) for k, v in data_base.items()} }")
        print(f"R-Tuning:{ {k: len(v) for k, v in data_rt.items()} }")

        # Delta ICR per layer (main comparison figure)
        plot_delta_icr(data_base, data_rt,
            output_dir / f"delta_icr{suffix}.png",
            title=f"ICR Score Delta (R-Tuning − Base) per Layer ({args.task or 'all'})")

        # Side-by-side AUROC: hallucination detection
        plot_auroc_comparison(data_base, data_rt,
            output_dir / f"auroc_comparison_hallucination{suffix}.png",
            pos_label="hallucination", neg_label="correct_confident",
            title=f"Per-layer AUROC: Hallucination Detection — Base vs R-Tuning ({args.task or 'all'})")

        # Side-by-side AUROC: false refusal
        plot_auroc_comparison(data_base, data_rt,
            output_dir / f"auroc_comparison_false_refusal{suffix}.png",
            pos_label="false_refusal", neg_label="correct_refusal",
            title=f"Per-layer AUROC: False Refusal Detection — Base vs R-Tuning ({args.task or 'all'})")

        # Layer distributions for both models combined (hallucination only)
        plot_delta_icr(data_base, data_rt,
            output_dir / f"delta_icr_hallu_refusal{suffix}.png",
            labels_to_plot=["hallucination", "false_refusal"],
            title=f"ICR Score Delta: Hallucination & False Refusal ({args.task or 'all'})")

        # PCA comparison: hallucination samples base vs rtuning
        plot_pca_comparison(data_base, data_rt,
            output_dir / f"pca_comparison_hallucination{suffix}.png",
            label="hallucination",
            title=f"PCA: Hallucination Samples — Base vs R-Tuning ({args.task or 'all'})")


if __name__ == "__main__":
    main()
