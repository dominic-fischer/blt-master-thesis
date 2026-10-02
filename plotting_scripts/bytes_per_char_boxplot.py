#!/usr/bin/env python3
"""
bytes_per_char_boxplot.py
Box plot of premium grouped by UTF-8 bytes per character of each language's
script (1 = Latin, 2 = e.g. Cyrillic/Arabic/Hebrew, 3 = e.g. Devanagari/CJK/
Thai, 4 = supplementary-plane scripts), for both monotonicity and normalisation.
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from correlate_w_cc_pages import load_data, build_matched, MASTER_CSV

PREMIUM_COLS = {
    "monotonicity":  "raw_monotonicity_t_0.3662_pps_premium",
    "normalisation": "norm_entropy_t_0.5293_pps_premium",
}

MIN_GROUP_SIZE = 3

# UTF-8 bytes per character for the main letters of each ISO 15924 script.
# Covers the FLORES+ scripts; anything missing is reported and skipped.
SCRIPT_BYTES = {
    # 1 byte (ASCII letters)
    "Latn": 1,
    # 2 bytes (U+0080 - U+07FF)
    "Arab": 2, "Armn": 2, "Cyrl": 2, "Grek": 2, "Hebr": 2,
    "Nkoo": 2, "Syrc": 2, "Thaa": 2,
    # 3 bytes (U+0800 - U+FFFF)
    "Beng": 3, "Deva": 3, "Ethi": 3, "Geor": 3, "Gujr": 3, "Guru": 3,
    "Hang": 3, "Hans": 3, "Hant": 3, "Jpan": 3, "Khmr": 3, "Knda": 3,
    "Laoo": 3, "Mlym": 3, "Mtei": 3, "Mymr": 3, "Olck": 3, "Orya": 3,
    "Sinh": 3, "Taml": 3, "Telu": 3, "Tfng": 3, "Thai": 3, "Tibt": 3,
    # 4 bytes (supplementary planes)
    "Adlm": 4,
}

# ── Style (matches the "Premium vs. Entropy mean" charts) ─────────────────────

BYTE_COLORS = {
    1: "#4C72B0",   # blue
    2: "#55A868",   # green
    3: "#CCB974",   # tan
    4: "#8172B3",   # purple (not in the reference chart; only if 4-byte scripts appear)
}
EDGE_COLOR    = "#333333"
SUBTITLE_GREY = "#555555"
GRID_COLOR    = "#dddddd"


def get_script_tag(row):
    tag = row.get("script_tag", "")
    if not tag and "_" in row["code"]:
        tag = row["code"].split("_")[1]
    return tag


def make_boxplot(premium_col, mode_label, out_path):
    data = load_data(MASTER_CSV, premium_col)
    matched, unmatched = build_matched(data)
    df = pd.DataFrame(matched)

    df["script_tag"] = [get_script_tag(r) for r in matched]
    df["bytes"] = df["script_tag"].map(SCRIPT_BYTES)

    unknown = df[df["bytes"].isna()]
    if not unknown.empty:
        print("No bytes-per-char mapping for these scripts (skipped): "
              + ", ".join(sorted(unknown["script_tag"].unique())))
    df = df.dropna(subset=["bytes"])
    df["bytes"] = df["bytes"].astype(int)

    for b, grp in df.groupby("bytes"):
        print(f"  {b} byte(s): {len(grp):3d} languages, scripts = "
              + ", ".join(sorted(grp["script_tag"].unique())))

    counts = df["bytes"].value_counts()
    kept = [b for b in sorted(counts.index) if counts[b] >= MIN_GROUP_SIZE]
    df = df[df["bytes"].isin(kept)]

    box_data = [df[df["bytes"] == b]["premium"].values for b in kept]
    colors   = [BYTE_COLORS.get(b, "#888888") for b in kept]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bp = ax.boxplot(
        box_data, patch_artist=True, widths=0.5,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 7},
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": EDGE_COLOR}, capprops={"color": EDGE_COLOR},
        flierprops={"marker": ""},   # individual points are drawn below anyway
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(EDGE_COLOR)

    rng = np.random.default_rng(0)
    for i, (ys, color) in enumerate(zip(box_data, colors)):
        xs = rng.normal(i + 1, 0.06, size=len(ys))
        ax.scatter(xs, ys, color=color, edgecolors=EDGE_COLOR, linewidths=0.8,
                   s=60, zorder=3)

    ax.set_xticks(range(1, len(kept) + 1))
    ax.set_xticklabels([f"{b} byte{'s' if b > 1 else ''}\n(n={len(g)})"
                        for b, g in zip(kept, box_data)])

    # titles
    m = re.search(r"_t_([\d.]+)", premium_col)
    subtitle = f"{mode_label.capitalize()}" + (f" (t = {m.group(1)})" if m else "")
    ax.set_title(f"{mode_label.capitalize()} Premium by Bytes per Character",
                 fontsize=17, fontweight="bold", pad=34)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=13, color=SUBTITLE_GREY)

    # axes
    ax.set_xlabel("UTF-8 bytes per character", fontsize=15)
    ax.set_ylabel("BLT Patch Premium vs. English", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, axis="both", linestyle="--", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out_path}\n")


Path("charts").mkdir(parents=True, exist_ok=True)
for mode_label, premium_col in PREMIUM_COLS.items():
    print(f"── {mode_label} ──")
    out_path = Path("charts") / f"bytes_per_char_boxplot_{mode_label}.png"
    make_boxplot(premium_col, mode_label, out_path)