#!/usr/bin/env python3
"""
plot_premiums.py
Reads results/summary.txt and produces per-script bar charts
of BLT patch premiums relative to English.
"""

import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ── 1. Parse the summary file ───────────────────────────────────────────────

SUMMARY = Path("results/summary.txt")

def parse_summary(path: Path):
    rows = []
    # ISO code pattern: e.g. eng_Latn, arb_Arab_sout3123
    CODE_RE = re.compile(r'\b([a-z]{2,3}_[A-Z][a-zA-Z0-9_]+)\b')
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Language") or line.startswith("---"):
                continue
            # Find the ISO code position
            m = CODE_RE.search(line)
            if not m:
                continue
            lang_name = line[:m.start()].strip()
            rest = line[m.end():].strip()
            code = m.group(1)
            nums = rest.split()
            if len(nums) < 6:
                continue
            # sents, total_patches, total_bytes, avg_bpp, avg_pps, premium
            premium = float(nums[5])
            avg_bpp = float(nums[3])
            rows.append({
                "language": lang_name,
                "code": code,
                "premium": premium,
                "avg_bpp": avg_bpp,
            })
    return rows

data = parse_summary(SUMMARY)

# ── 2. Script detection from ISO 15924 script subtag in the language code ───

def get_script(code: str) -> str:
    parts = code.split("_")
    # code format: lang_Script[_variant]
    if len(parts) >= 2:
        s = parts[1]
        # Handle 4-letter script tags that may have been truncated in longer codes
        for tag, label in SCRIPT_LABELS.items():
            if s.startswith(tag):
                return label
    return "Other"

for row in data:
    row["script"] = get_script(row["code"])

# Group by script
from collections import defaultdict
script_groups = defaultdict(list)
for row in data:
    script_groups[row["script"]].append(row)

# Sort within each group by premium ascending
for g in script_groups.values():
    g.sort(key=lambda r: r["premium"])

# Sort script groups by median premium
script_order = sorted(script_groups.keys(),
                      key=lambda s: np.median([r["premium"] for r in script_groups[s]]))

# ── 3. Color palette ─────────────────────────────────────────────────────────

# Parity line at 1.0; color bars by premium severity
def bar_color(premium, cmap):
    # map 1→5+ onto the colormap
    t = min((premium - 1.0) / 4.0, 1.0)
    return cmap(t)

CMAP = plt.cm.RdYlGn_r   # green (low) → red (high)

# ── 4. Plot each script ───────────────────────────────────────────────────────

OUT_DIR = Path("results/figures")
OUT_DIR_OVERVIEW = Path("results")
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

    # Parity line
    ax.axhline(1.0, **PARITY_KW, label="Parity (1.0)", zorder=4)

    # Value labels on bars
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

    # Colorbar legend
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=1, vmax=5))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.02, pad=0.01, aspect=30)
    cbar.set_label("Premium severity", fontsize=8, color="#444444")
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()

    safe_name = script.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    out_path = OUT_DIR / f"premium_{safe_name}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_path)
    print(f"  saved → {out_path}")

# ── 5. Overview chart: range boxes (min–max) with median marker ──────────────

stats = []
for s in script_order:
    prems = [r["premium"] for r in script_groups[s]]
    stats.append({
        "script": s,
        "min":    min(prems),
        "max":    max(prems),
        "median": np.median(prems),
        "n":      len(prems),
    })

fig, ax = plt.subplots(figsize=(max(10, 0.72 * len(stats)), 6))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")

x = np.arange(len(stats))
BOX_W = 0.55

for i, st in enumerate(stats):
    med_color = bar_color(st["median"], CMAP)

    # Filled range rectangle (min → max)
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

    # Median horizontal line
    ax.plot(
        [i - BOX_W / 2, i + BOX_W / 2],
        [st["median"], st["median"]],
        color=med_color,
        linewidth=2.5,
        solid_capstyle="round",
        zorder=4,
    )

    # Min / max tick marks
    for val in (st["min"], st["max"]):
        ax.plot(
            [i - BOX_W / 4, i + BOX_W / 4],
            [val, val],
            color=med_color,
            linewidth=1.4,
            zorder=4,
        )

    # Median value label
    ax.text(i, st["median"] + 0.07, f"{st['median']:.2f}",
            ha="center", va="bottom",
            fontsize=7, color="#111111", fontweight="600", zorder=5)

    # n= label just below the bottom of the box
    ax.annotate(f"n={st['n']}", xy=(i, st["min"]),
                xytext=(0, -6), textcoords="offset points",
                ha="center", va="top",
                fontsize=6, color="#555555", zorder=5,
                style="italic")

ax.axhline(1.0, **PARITY_KW, label="Parity (1.0)", zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([st["script"] for st in stats],
                   rotation=40, ha="right", fontsize=8.5, color="#333333")
ax.tick_params(**FONT_TICK)
ax.set_ylabel("Premium vs. English", **FONT_AXIS)
ax.set_title("BLT Patch Premium by Script Family\n"
             "bar = median  ·  shaded range = min–max",
             **FONT_TITLE, pad=10)
ax.grid(**GRID_KW, zorder=1)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#aaaaaa")
ax.set_xlim(-0.6, len(stats) - 0.4)
ax.set_ylim(0, max(st["max"] for st in stats) * 1.12)
ax.legend(fontsize=8, framealpha=0.6)

sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=1, vmax=5))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                    fraction=0.018, pad=0.01, aspect=30)
cbar.set_label("Median premium severity", fontsize=8, color="#444444")
cbar.ax.tick_params(labelsize=7)

plt.tight_layout()
fig.subplots_adjust(bottom=0.22)
overview_path = OUT_DIR_OVERVIEW / "premium_overview_by_script.png"
fig.savefig(overview_path, dpi=160, bbox_inches="tight")
plt.close(fig)
saved.append(overview_path)
print(f"  saved → {overview_path}")

print(f"\nDone. {len(saved)} charts saved to {OUT_DIR}/")