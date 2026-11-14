"""Regenerate only corrected Figure 2 from frozen Phase 6 repeat metrics."""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "FIGURES"
data = pd.read_csv(ROOT / "results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10, "figure.dpi": 180, "savefig.dpi": 300,
})
rows = []
for repeat in (1, 2, 3):
    selection = data[(data["repeat"] == repeat) & (data["model_id"] == "selection_aware_procedure")].iloc[0]
    comparator = data[(data["repeat"] == repeat) & (data["model_id"] == "frozen_v3_logistic_balanced")].iloc[0]
    rows.append((repeat, selection.roc_auc, comparator.roc_auc))

fig, ax = plt.subplots(figsize=(7.8, 4.8))
for label, column, color, marker, yoff in [
    ("Selection-aware procedure", 1, "#176B87", "o", 9),
    ("V3-style comparator", 2, "#D97706", "s", -17),
]:
    xs = [r[0] for r in rows]
    ys = [r[column] for r in rows]
    ax.plot(xs, ys, marker=marker, color=color, lw=2.1, ms=7, label=label)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, yoff), textcoords="offset points",
                    ha="center", fontsize=8, color=color)
ax.set_xticks([1, 2, 3], ["Repeat 1", "Repeat 2", "Repeat 3"])
ax.set_xlim(0.8, 3.2)
ax.set_ylim(0.0, 1.0)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylabel("Complete-repeat out-of-fold ROC-AUC (0–1)")
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, loc="lower right")
ax.set_title("Paired ZAS repeat-level discrimination")
fig.text(0.5, 0.01,
         "Each repeat predicts all 303 source rows once. Repeats overlap; points are not independent samples.",
         ha="center", fontsize=8)
fig.tight_layout(rect=[0, 0.05, 1, 1])
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "FIGURE2_PHASE6_REPEAT_AUC.png", bbox_inches="tight")
fig.savefig(OUT / "FIGURE2_PHASE6_REPEAT_AUC.pdf", bbox_inches="tight")
plt.close(fig)
