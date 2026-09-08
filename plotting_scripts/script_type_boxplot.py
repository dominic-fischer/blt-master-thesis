#!/usr/bin/env python3
"""
script_type_boxplot.py
Box plot of premium by script type, for both monotonicity and normalisation.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SCRIPT_COLORS
from correlate_w_cc_pages import load_data, build_matched, MASTER_CSV

PREMIUM_COLS = {
    "monotonicity":  "raw_monotonicity_t_0.3662_pps_premium",
    "normalisation": "norm_entropy_t_0.5293_pps_premium",
}

MIN_GROUP_SIZE = 3
SCRIPT_ORDER = ["Latin", "Arabic", "Cyrillic", "Devanagari"]


def make_boxplot(premium_col, mode_label, out_path):
    data = load_data(MASTER_CSV, premium_col)
    matched, unmatched = build_matched(data)
    df = pd.DataFrame(matched)

    script_counts = df['script'].value_counts()
    kept_scripts = [s for s in SCRIPT_ORDER if script_counts.get(s, 0) >= MIN_GROUP_SIZE]
    df = df[df['script'].isin(kept_scripts)]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    box_data = [df[df['script'] == s]['premium'].values for s in kept_scripts]
    colors = [SCRIPT_COLORS.get(s, "#888888") for s in kept_scripts]

    bp = ax.boxplot(box_data, labels=kept_scripts, patch_artist=True, widths=0.5,
                     showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white",
                                                 "markeredgecolor": "black", "markersize": 6},
                     medianprops={"color": "black"})
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    rng = np.random.default_rng(0)
    for i, s in enumerate(kept_scripts):
        ys = df[df['script'] == s]['premium'].values
        xs = rng.normal(i + 1, 0.06, size=len(ys))
        ax.scatter(xs, ys, color=colors[i], edgecolors="white", linewidths=0.4,
                   s=30, alpha=0.8, zorder=3)

    n_per_group = [len(g) for g in box_data]
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(kept_scripts, n_per_group)])

    ax.set_ylabel(f"BLT Patch Premium vs. English ({mode_label.capitalize()})", fontsize=11)
    ax.set_title(f"{mode_label.capitalize()} Premium by Script Type", fontsize=13, fontweight="bold")
    ax.grid(axis="y", color="#cccccc", linewidth=0.5, linestyle=":", alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#aaaaaa")

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


Path("charts").mkdir(parents=True, exist_ok=True)
for mode_label, premium_col in PREMIUM_COLS.items():
    out_path = Path("charts") / f"script_type_boxplot_{mode_label}.png"
    make_boxplot(premium_col, mode_label, out_path)