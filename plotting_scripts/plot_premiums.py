#!/usr/bin/env python3
"""
plot_premiums.py
Reads floresplus_MASTER_CSV.csv and produces per-script bar charts
of BLT patch premiums relative to English, plus an overview chart.
Output filenames carry the premium-column tag so different configs
don't overwrite each other.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import csv
from pathlib import Path
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SCRIPT_LABELS

# ── Config ──────────────────────────────────────────────────────────────────

MASTER_CSV = Path("floresplus_MASTER_CSV.csv")

# Which premium column to plot. Same 16 options as the other scripts — set the
# full "<model>_t_<threshold>_pps_premium" name.
PREMIUM_COL = "raw_entropy_t_1.3340_pps_premium"
PREMIUM_TAG = PREMIUM_COL.replace("_pps_premium", "")   # appended to filenames

# ── 1. Load the master CSV ──────────────────────────────────────────────────

def load_data(path: Path, premium_col: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if premium_col not in reader.fieldnames:
            avail = [c for c in reader.fieldnames if c.endswith("_pps_premium")]
            raise SystemExit(
                f"Column '{premium_col}' not found in {path}.\n"
                "Available premium columns:\n  " + "\n  ".join(avail)
            )
        bpp_col = premium_col.replace("_pps_premium", "_bpp")  # matching bpp column
        for row in reader:
            # Code_Orig is the script-tagged code (e.g. 'ace_Arab'); keep it so
            # script colouring and the 'eng_' filter below still work.
            code = (row.get("Code_Orig") or row.get("Code") or "").strip()
            if not code:
                continue
            prem_raw = (row.get(premium_col) or "").strip()
            if not prem_raw:
                continue
            try:
                premium = float(prem_raw)
            except ValueError:
                continue
            try:
                avg_bpp = float(row.get(bpp_col) or "nan")
            except ValueError:
                avg_bpp = float("nan")
            rows.append({
                "language":   (row.get("Name") or "").strip(),
                "code":       code,
                "script_tag": (row.get("Script") or "").strip(),
                "premium":    premium,
                "avg_bpp":    avg_bpp,
            })
    return rows

data = load_data(MASTER_CSV, PREMIUM_COL)

# ── 2. Script detection ──────────────────────────────────────────────────────

def get_script(row) -> str:
    # Prefer the explicit Script column; fall back to the tag in Code_Orig.
    s = row.get("script_tag") or ""
    if not s and "_" in row["code"]:
        s = row["code"].split("_")[1]
    for tag, label in SCRIPT_LABELS.items():
        if s.startswith(tag):
            return label
    return "Other"

for row in data:
    row["script"] = get_script(row)

# Group by script
script_groups = defaultdict(list)
for row in data:
    script_groups[row["script"]].append(row)

# ── Save original groupings for the overview (before singleton merge) ────────
script_groups_full = {s: list(rows) for s, rows in script_groups.items()}

# ── Merge singleton scripts into "Other" (only for per-script bar charts) ────
merged_other = []
singleton_scripts = [s for s, rows in script_groups.items() if len(rows) == 1]
for s in singleton_scripts:
    merged_other.extend(script_groups.pop(s))

if merged_other:
    script_groups["Other"].extend(merged_other)

# Sort within each group by premium ascending
for g in script_groups.values():
    g.sort(key=lambda r: r["premium"])

# Sort script groups by median premium
script_order = sorted(script_groups.keys(),
                      key=lambda s: np.median([r["premium"] for r in script_groups[s]]))

# ── 3. Color palette ─────────────────────────────────────────────────────────

def bar_color(premium, cmap):
    t = min((premium - 1.0) / 4.0, 1.0)
    return cmap(t)

CMAP = plt.cm.RdYlGn_r

# ── 4. Plot each script ───────────────────────────────────────────────────────

OUT_DIR = Path("charts/script_premiums")
OUT_DIR_OVERVIEW = Path("charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_TITLE  = {"fontsize": 13, "fontweight": "bold", "color": "#1a1a2e"}
FONT_AXIS   = {"fontsize": 9,  "color": "#333333"}
FONT_TICK   = {"labelsize": 8, "labelcolor": "#444444"}
PARITY_KW   = dict(color="#2d6a4f", linewidth=1.4, linestyle="--", alpha=0.8)
GRID_KW     = dict(axis="y", color="#cccccc", linewidth=0.6, linestyle=":", alpha=0.7)

saved = []

for script in script_order:
    rows = script_groups[script]
    n = len(rows)
    langs   = [r["language"] for r in rows]
    prems   = [r["premium"]  for r in rows]
    colors  = [bar_color(p, CMAP) for p in prems]

    fig_w = max(7, 0.45 * n + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    x = np.arange(n)
    bars = ax.bar(x, prems, color=colors, width=0.72, zorder=3,
                  edgecolor="white", linewidth=0.4)

    ax.axhline(1.0, **PARITY_KW, label="Parity (1.0)", zorder=4)

    for bar, p in zip(bars, prems):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{p:.2f}", ha="center", va="bottom",
                fontsize=6.5, color="#222222", fontweight="500", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=45, ha="right", **FONT_AXIS)
    ax.tick_params(**FONT_TICK)
    ax.set_ylabel("Premium vs. English", **FONT_AXIS)
    ax.set_title(f"{script} — BLT Patch Premium per Language", **FONT_TITLE, pad=10)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(**GRID_KW, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#aaaaaa")
    ax.set_ylim(0, max(prems) * 1.18)
    ax.legend(fontsize=8, framealpha=0.6)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=1, vmax=5))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.02, pad=0.01, aspect=30)
    cbar.set_label("Premium severity", fontsize=8, color="#444444")
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()

    safe_name = script.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    out_path = OUT_DIR / f"premium_{safe_name}_{PREMIUM_TAG}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_path)
    print(f"  saved -> {out_path}")

# ── 5. Overview chart: uses script_groups_full (singletons NOT collapsed) ────

overview_stats = []
# Sort by median premium, same logic as before but from the full groupings
full_script_order = sorted(
    script_groups_full.keys(),
    key=lambda s: np.median([r["premium"] for r in script_groups_full[s]
                              if not r["code"].startswith("eng_")] or [0])
)

for s in full_script_order:
    prems = [r["premium"] for r in script_groups_full[s]
             if not r["code"].startswith("eng_")]
    if not prems:
        continue
    overview_stats.append({
        "script": s,
        "min":    min(prems),
        "max":    max(prems),
        "median": np.median(prems),
        "n":      len(prems),
    })

fig, ax = plt.subplots(figsize=(max(10, 0.72 * len(overview_stats)), 6))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")

x = np.arange(len(overview_stats))
BOX_W = 0.55

for i, st in enumerate(overview_stats):
    med_color = bar_color(st["median"], CMAP)

    rect = plt.Rectangle(
        (i - BOX_W / 2, st["min"]),
        BOX_W,
        st["max"] - st["min"],
        facecolor=med_color,
        alpha=0.30,
        edgecolor=med_color,
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(rect)

    ax.plot(
        [i - BOX_W / 2, i + BOX_W / 2],
        [st["median"], st["median"]],
        color=med_color,
        linewidth=2.5,
        solid_capstyle="round",
        zorder=4,
    )

    for val in (st["min"], st["max"]):
        ax.plot(
            [i - BOX_W / 4, i + BOX_W / 4],
            [val, val],
            color=med_color,
            linewidth=1.4,
            zorder=4,
        )

    ax.text(i, st["median"] + 0.07, f"{st['median']:.2f}",
            ha="center", va="bottom",
            fontsize=7, color="#111111", fontweight="600", zorder=5)

    ax.annotate(f"n={st['n']}", xy=(i, st["min"]),
                xytext=(0, -6), textcoords="offset points",
                ha="center", va="top",
                fontsize=6, color="#555555", zorder=5,
                style="italic")

ax.axhline(1.0, **PARITY_KW, label="Parity (1.0)", zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([st["script"] for st in overview_stats],
                   rotation=40, ha="right", fontsize=8.5, color="#333333")
ax.tick_params(**FONT_TICK)
ax.set_ylabel("Premium vs. English", **FONT_AXIS)
ax.set_title("BLT Patch Premium by Script Family\n"
             "bar = median  ·  shaded range = min–max",
             **FONT_TITLE, pad=10)
ax.grid(**GRID_KW, zorder=1)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#aaaaaa")
ax.set_xlim(-0.6, len(overview_stats) - 0.4)
ax.set_ylim(0, max(st["max"] for st in overview_stats) * 1.12)
ax.legend(fontsize=8, framealpha=0.6)

sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=1, vmax=5))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                    fraction=0.018, pad=0.01, aspect=30)
cbar.set_label("Median premium severity", fontsize=8, color="#444444")
cbar.ax.tick_params(labelsize=7)

plt.tight_layout()
fig.subplots_adjust(bottom=0.22)
overview_path = OUT_DIR_OVERVIEW / f"premium_overview_by_script_{PREMIUM_TAG}.png"
fig.savefig(overview_path, dpi=160, bbox_inches="tight")
plt.close(fig)
saved.append(overview_path)
print(f"  saved -> {overview_path}")

print(f"\nDone. {len(saved)} charts saved.")