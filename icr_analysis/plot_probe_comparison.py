"""Plot probe AUROC comparison: base vs rtuning."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

results_path = Path("icr_analysis/outputs/probe_results.json")
output_dir = Path("icr_analysis/outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

with open(results_path) as f:
    results = json.load(f)

tasks = [
    ("hallucination_detection", "Hallucination Detection\n(hallucination vs correct_confident)"),
    ("refusal_calibration",     "Refusal Calibration\n(false_refusal vs correct_refusal)"),
]
datasets = ["all", "hotpotqa", "pararel"]
models   = ["base", "rtuning"]
colors   = {"base": "#5B9BD5", "rtuning": "#ED7D31"}
labels   = {"base": "Base", "rtuning": "Refusal Fine-tuned"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (task_key, task_title) in zip(axes, tasks):
    x = np.arange(len(datasets))
    width = 0.35

    for i, model in enumerate(models):
        aurocs = []
        errs   = []
        for ds in datasets:
            key = f"{model}/{task_key}/{ds}"
            aurocs.append(results[key]["auroc_mean"])
            errs.append(results[key]["auroc_std"])
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, aurocs, width, yerr=errs, capsize=4,
                      label=labels[model], color=colors[model], alpha=0.85)
        for bar, val in zip(bars, aurocs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(task_title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(["All", "HotpotQA", "ParaRel"])
    ax.set_ylabel("AUROC (5-fold CV)")
    ax.set_ylim(0.4, 0.85)
    ax.legend()

fig.suptitle("ICR Probe AUROC: Base vs Refusal Fine-tuned", fontsize=13, fontweight="bold")
fig.tight_layout()

out = output_dir / "probe_auroc_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
