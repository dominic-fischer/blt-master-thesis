#!/usr/bin/env python3
"""
plot_cc_correlation.py
Standalone script. Reads results/results_CSV/base_model_results.csv — which
contains BOTH the BLT patch-premium results and the Common Crawl page counts
in a single file — then plots BLT patch premium vs. Common Crawl page count
with per-script colouring and a log-scale x-axis, for both raw entropy and
monotonicity premiums.

Optional: `pip install adjustText` for automatic, non-overlapping point labels
with leader lines. Without it, labels fall back to a fixed offset.
"""

import csv
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pathlib import Path

# specify parent directory of this script to import from
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SCRIPT_LABELS, SCRIPT_COLORS

# ── Config ────────────────────────────────────────────────────────────────────

MASTER_CSV = Path("results/results_CSV/base_model_results.csv")
OUT_DIR    = Path("charts/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cosmetic label for the x-axis only (the master CSV no longer tags the crawl).
CC_CRAWL = "CC-MAIN-2026-17"

# Which premium columns to plot. Each produces its own chart.
PREMIUM_COLS = {
    "raw_entropy":      "raw_entropy_t_1.3340_pps_premium",
    "raw_monotonicity": "raw_monotonicity_t_0.3662_pps_premium",
    "norm_entropy":      "norm_entropy_t_0.5293_pps_premium",
}

TITLE_LABELS = {
    "raw_entropy":      "Raw Entropy",
    "raw_monotonicity": "Monotonicity",
    "norm_entropy":      "Normalisation",
}

# ── Style (matches the "Premium vs. Entropy mean" charts) ─────────────────────

FIT_COLOR     = "#e8352b"
EDGE_COLOR    = "#333333"
LEADER_COLOR  = "#999999"
SUBTITLE_GREY = "#555555"
GRID_COLOR    = "#dddddd"

# With many languages, full-size labels and markers would swamp the plot, so
# they shrink once the point count passes this threshold.
DENSE_THRESHOLD = 40
SHOW_LABELS = False   # set True to label every point with its language name
# ── 1. Load the master CSV ────────────────────────────────────────────────────

def load_data(path, premium_col):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if premium_col not in reader.fieldnames:
            available = [c for c in reader.fieldnames if c.endswith("_pps_premium")]
            raise SystemExit(
                f"Column '{premium_col}' not found in {path}.\n"
                "Available premium columns:\n  " + "\n  ".join(available)
            )
        for row in reader:
            # Code_Orig is the script-tagged code (e.g. 'ace_Arab'); Code is the bare
            # ISO (e.g. 'ace'). Prefer the tagged form so script colouring still works.
            code = (row.get("Code_Orig") or row.get("Code") or "").strip()
            if not code or code.lower().startswith("eng"):
                continue  # English is the baseline the premium is measured against

            prem_raw = (row.get(premium_col) or "").strip()
            if not prem_raw:
                continue
            try:
                premium = float(prem_raw)
            except ValueError:
                continue

            rows.append({
                "language":     (row.get("Name") or "").strip(),
                "code":         code,
                "script_tag":   (row.get("Script") or "").strip(),
                "premium":      premium,
                "cc_pages_raw": (row.get("CC_Pages") or "").strip(),
            })
    return rows

# ── 2. Assign script ──────────────────────────────────────────────────────────

def get_script(row):
    # Prefer the explicit Script column; fall back to the tag in Code_Orig.
    s = row["script_tag"]
    if not s and "_" in row["code"]:
        s = row["code"].split("_")[1]
    for tag, label in SCRIPT_LABELS.items():
        if s.startswith(tag):
            return label
    return "Other"

# ── 3. Build matched dataset (rows that actually have a CC page count) ─────────

def build_matched(data):
    matched, unmatched = [], []
    for row in data:
        raw = row["cc_pages_raw"]
        if not raw:
            unmatched.append(f"  {row['language']} ({row['code']}) — no CC_Pages")
            continue
        try:
            pages = int(float(raw.replace(",", "")))
        except ValueError:
            unmatched.append(f"  {row['language']} ({row['code']}) — bad CC_Pages={raw!r}")
            continue
        if pages <= 0:  # log scale needs strictly positive values
            unmatched.append(f"  {row['language']} ({row['code']}) — CC_Pages={pages}")
            continue
        row["cc_pages"] = pages
        row["script"]   = get_script(row)
        matched.append(row)
    return matched, unmatched

# ── 4 & 5. Regression + plot, per premium mode ─────────────────────────────────

def _label_points(ax, matched, fontsize):
    """Point labels with thin grey leader lines. Uses adjustText if installed;
    otherwise falls back to a fixed offset."""
    leader = dict(arrowstyle="-", color=LEADER_COLOR, lw=0.6)
    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None

    if adjust_text is not None:
        texts = [
            ax.text(r["cc_pages"], r["premium"], r["language"],
                    fontsize=fontsize, color="#222222", zorder=5)
            for r in matched
        ]
        adjust_text(texts,
                    x=np.array([r["cc_pages"] for r in matched]),
                    y=np.array([r["premium"] for r in matched]),
                    ax=ax, arrowprops=leader)
    else:
        for r in matched:
            ax.annotate(r["language"], xy=(r["cc_pages"], r["premium"]),
                        xytext=(8, 6), textcoords="offset points",
                        fontsize=fontsize, color="#222222", zorder=5,
                        arrowprops=leader)


def make_chart(matched, mode_title, subtitle, out_path):
    log_x  = np.log10([r["cc_pages"] for r in matched])
    y      = np.array([r["premium"]  for r in matched])
    slope, intercept, r_val, p_val, _ = stats.linregress(log_x, y)
    print(f"Log regression ({mode_title}): r={r_val:.3f}, p={p_val:.4f}, slope={slope:.3f}")

    scripts_present = sorted(set(r["script"] for r in matched))
    PALETTE         = plt.cm.tab20.colors
    fallback        = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(scripts_present)}

    def get_color(script):
        return SCRIPT_COLORS.get(script, fallback.get(script, "#888888"))

    dense       = len(matched) > DENSE_THRESHOLD
    marker_size = 70 if dense else 220
    label_size  = 6.5 if dense else 11

    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # points
    for script in scripts_present:
        pts = [r for r in matched if r["script"] == script]
        ax.scatter(
            [r["cc_pages"] for r in pts],
            [r["premium"]  for r in pts],
            color=get_color(script),
            s=marker_size, zorder=4,
            edgecolors=EDGE_COLOR, linewidths=0.8 if dense else 1.0,
        )
    if SHOW_LABELS:
        _label_points(ax, matched, label_size)

    # fit line
    x_range = np.linspace(log_x.min(), log_x.max(), 300)
    fit_label = f"Log fit: r = {r_val:.2f}, r$^2$ = {r_val**2:.2f}, p = {p_val:.3f}"
    ax.plot(10**x_range, intercept + slope * x_range,
            color=FIT_COLOR, linewidth=2.5, linestyle="--", zorder=3)

    # legend: one entry per script with counts, then the fit line
    handles = []
    for script in scripts_present:
        n = sum(1 for r in matched if r["script"] == script)
        handles.append(plt.Line2D(
            [0], [0], marker="o", linestyle="None",
            markerfacecolor=get_color(script), markeredgecolor=EDGE_COLOR,
            markeredgewidth=1.0, markersize=11, label=f"{script} (n={n})",
        ))
    handles.append(plt.Line2D([0], [0], color=FIT_COLOR, linewidth=2.5,
                              linestyle="--", label=fit_label))
    n_cols = 2 if len(scripts_present) > 8 else 1
    leg = ax.legend(handles=handles, title="Script", fontsize=10 if n_cols == 2 else 12,
                    title_fontsize=12, loc="best", ncol=n_cols, frameon=True,
                    framealpha=0.95, edgecolor="#cccccc", fancybox=True,
                    borderpad=0.8, labelspacing=0.6)
    leg._legend_box.align = "left"

    # titles
    ax.set_title("BLT Patch Premium vs. Common Crawl Presence",
                 fontsize=17, fontweight="bold", pad=34)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=13, color=SUBTITLE_GREY)

    # axes
    ax.set_xscale("log")
    ax.set_xlabel(f"Pages in Common Crawl ({CC_CRAWL}, log scale)", fontsize=15)
    ax.set_ylabel("BLT Patch Premium vs. English", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)

    ax.grid(True, which="major", linestyle="--", color=GRID_COLOR, linewidth=0.8)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out_path}\n")


for mode_label, premium_col in PREMIUM_COLS.items():
    data = load_data(MASTER_CSV, premium_col)
    matched, unmatched = build_matched(data)

    print(f"Premium column : {premium_col}")
    print(f"Matched        : {len(matched)}")
    print(f"Unmatched      : {len(unmatched)}")
    if unmatched:
        print("\n".join(unmatched))

    if not matched:
        print(f"WARNING: no rows with a usable CC_Pages value for {premium_col} — skipping.\n")
        continue

    mode_title = TITLE_LABELS.get(mode_label, mode_label)
    m = re.search(r"_t_([\d.]+)", premium_col)
    subtitle = f"{mode_title}" + (f" (t = {m.group(1)})" if m else "")

    out_path = OUT_DIR / f"premium_vs_cc_pages_{premium_col.replace('_pps_premium', '')}.png"
    make_chart(matched, mode_title, subtitle, out_path)