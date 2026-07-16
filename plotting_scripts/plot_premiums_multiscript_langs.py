#!/usr/bin/env python3
"""
plot_multiscript.py
Reads floresplus_MASTER.csv and plots grouped bars for languages that
appear in multiple distinct scripts (matched on the ISO 639 part before the
first underscore of Code_Orig; regional variants like lld_Latn_gard1241 are
collapsed into lld + Latn).
Output: charts/premium_multiscript.png
"""
import re
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

MASTER_CSV = Path("floresplus_MASTER.csv")

# Which premium column to read. Same 16 options as the other scripts — set the
# full "<model>_t_<threshold>_pps_premium" name.
PREMIUM_COL = "raw_entropy_t_1.3340_pps_premium"

# ── 1. Parse ──────────────────────────────────────────────────────────────────
# Match the script-tagged code (e.g. 'ace_Arab', 'lld_Latn_gard1241'),
# capturing the ISO 639 part and the 4-letter script, dropping any suffix.
CODE_RE = re.compile(r'^([a-z]{2,3})_([A-Z][a-z]{3})(?:_[a-zA-Z0-9]+)?$')

rows = []
with open(MASTER_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    if PREMIUM_COL not in reader.fieldnames:
        avail = [c for c in reader.fieldnames if c.endswith("_pps_premium")]
        raise SystemExit(
            f"Column '{PREMIUM_COL}' not found in {MASTER_CSV}.\n"
            "Available premium columns:\n  " + "\n  ".join(avail)
        )
    for row in reader:
        code = (row.get("Code_Orig") or "").strip()
        m = CODE_RE.match(code)
        if not m:
            continue
        prem_raw = (row.get(PREMIUM_COL) or "").strip()
        if not prem_raw:
            continue
        try:
            premium = float(prem_raw)
        except ValueError:
            continue
        rows.append({
            "language": (row.get("Name") or "").strip(),
            "lang_iso": m.group(1),
            "script":   m.group(2),
            "premium":  premium,
        })

# ── 2. Collapse regional variants, keep only multi-script languages ───────────
# For each (lang_iso, script) keep the row with the shortest/simplest name
best = {}
for r in rows:
    key = (r["lang_iso"], r["script"])
    if key not in best or len(r["language"]) < len(best[key]["language"]):
        best[key] = r

by_iso = defaultdict(list)
for r in best.values():
    by_iso[r["lang_iso"]].append(r)

multiscript = {
    iso: sorted(entries, key=lambda e: e["premium"])
    for iso, entries in by_iso.items()
    if len(entries) > 1
}

def base_name(entries):
    return min(entries, key=lambda e: (e["language"].count("("), len(e["language"])))["language"]

groups = sorted(
    [{"iso": iso, "name": base_name(e), "entries": e}
     for iso, e in multiscript.items()],
    key=lambda g: g["name"]
)

if not groups:
    raise SystemExit("No languages found in more than one script — nothing to plot.")

# ── 3. Script colors & labels ─────────────────────────────────────────────────
SCRIPT_DISPLAY = {
    "Latn": "Latin",   "Arab": "Arabic",  "Deva": "Devanagari",
    "Hans": "Han (Simp.)", "Hant": "Han (Trad.)",
    "Beng": "Bengali", "Mtei": "Meitei Mayek", "Tfng": "Tifinagh",
}
SCRIPT_COLORS = {
    "Latn": "#4c9be8", "Arab": "#e5c07b", "Deva": "#98c379",
    "Hans": "#61afef", "Hant": "#abb2bf", "Beng": "#56b6c2",
    "Mtei": "#d19a66", "Tfng": "#e06c75",
}
def slabel(s): return SCRIPT_DISPLAY.get(s, s)
def scolor(s): return SCRIPT_COLORS.get(s, "#aaaaaa")

all_scripts = sorted({e["script"] for g in groups for e in g["entries"]})

# ── 4. Plot ───────────────────────────────────────────────────────────────────
BAR_W     = 0.32
GROUP_GAP = 0.45

# Compute x-centres for each group
centres = []
cursor = 0.0
for g in groups:
    n = len(g["entries"])
    centres.append(cursor + (n * BAR_W) / 2)
    cursor += n * BAR_W + GROUP_GAP

fig_w = max(11, cursor + 1.0)
fig, ax = plt.subplots(figsize=(fig_w, 6.0))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")

for centre, group in zip(centres, groups):
    entries = group["entries"]
    n = len(entries)
    offsets = np.linspace(-(n - 1) * BAR_W / 2, (n - 1) * BAR_W / 2, n)
    for e, offset in zip(entries, offsets):
        x     = centre + offset
        color = scolor(e["script"])
        ax.bar(x, e["premium"], width=BAR_W * 0.88,
               color=color, zorder=3, edgecolor="white", linewidth=0.5)
        # Premium value above bar
        ax.text(x, e["premium"] + 0.07, f"{e['premium']:.2f}",
                ha="center", va="bottom",
                fontsize=7.5, color="#222222", fontweight="500", zorder=5)
        # Script name inside bar (rotated)
        ax.text(x, 0.12, slabel(e["script"]),
                ha="center", va="bottom",
                fontsize=6.5, color="white", fontweight="600",
                rotation=90, zorder=5)

# ── 5. Two-line italic x-axis labels (language name + script name) ────────────
def strip_brackets(s):
    return re.sub(r'\s*\(.*?\)', '', s).strip()

xtick_pos   = centres
xtick_label = []
for group in groups:
    entries = group["entries"]
    xtick_label.append(f"{strip_brackets(group['name'])}")

ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_label, rotation=15, ha="center",
                   fontsize=8.5, color="#333333", style="italic")

# Parity line
ax.axhline(1.0, color="#2d6a4f", linewidth=1.4, linestyle="--",
           alpha=0.8, label="Parity (1.0)", zorder=2)

ax.tick_params(labelsize=8, labelcolor="#444444")
ax.set_ylabel("Premium vs. English", fontsize=9, color="#333333")
ax.set_title("BLT Patch Premium — Languages Written in Multiple Scripts",
             fontsize=13, fontweight="bold", color="#1a1a2e", pad=10)
ax.grid(axis="y", color="#cccccc", linewidth=0.6, linestyle=":", alpha=0.7, zorder=1)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#aaaaaa")
ax.set_xlim(-0.5, cursor - GROUP_GAP + 0.5)
ax.set_ylim(0, max(e["premium"] for g in groups for e in g["entries"]) * 1.18)

# Legend
legend_handles = [
    mpatches.Patch(color=scolor(s), label=slabel(s))
    for s in all_scripts
] + [
    plt.Line2D([0], [0], color="#2d6a4f", linewidth=1.4,
               linestyle="--", label="Parity (1.0)")
]
ax.legend(handles=legend_handles, fontsize=8, framealpha=0.6, loc="upper left")

plt.tight_layout()
OUT = Path(f"charts/premium_multiscript_{PREMIUM_COL.replace('_pps_premium', '')}.png")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}")