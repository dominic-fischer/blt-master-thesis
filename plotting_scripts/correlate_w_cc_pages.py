#!/usr/bin/env python3
"""
plot_cc_correlation.py
Standalone script. Reads results/summary.txt and commoncrawl_stats.csv,
then plots BLT patch premium vs. Common Crawl page count (most recent crawl)
with per-script colouring and a log-scale x-axis.
"""

import re
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pathlib import Path

from config import SCRIPT_LABELS, SCRIPT_COLORS

# ── Paths ────────────────────────────────────────────────────────────────────

SUMMARY  = Path("results/summary.txt")
CC_CSV   = Path("commoncrawl_stats.csv")
CC_CRAWL = "CC-MAIN-2026-17"
OUT_DIR  = Path("results/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────

FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "color": "#1a1a2e"}
FONT_AXIS  = {"fontsize": 9,  "color": "#333333"}
FONT_TICK  = {"labelsize": 8, "labelcolor": "#444444"}

# ── 1. Parse summary.txt ──────────────────────────────────────────────────────

def parse_summary(path):
    rows = []
    CODE_RE = re.compile(r'\b([a-z]{2,3}_[A-Z][a-zA-Z0-9_]+)\b')
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Language") or line.startswith("---"):
                continue
            m = CODE_RE.search(line)
            if not m:
                continue
            lang_name = line[:m.start()].strip()
            rest      = line[m.end():].strip()
            code      = m.group(1)
            nums      = rest.split()
            if len(nums) < 6:
                continue
            rows.append({
                "language": lang_name,
                "code":     code,
                "premium":  float(nums[5]),
                "avg_bpp":  float(nums[3]),
            })
    return rows

data = parse_summary(SUMMARY)

# ── 2. Assign script ──────────────────────────────────────────────────────────

def get_script(code):
    parts = code.split("_")
    if len(parts) >= 2:
        s = parts[1]
        for tag, label in SCRIPT_LABELS.items():
            if s.startswith(tag):
                return label
    return "Other"

for row in data:
    row["script"] = get_script(row["code"])

# ── 3. Load CC page counts ────────────────────────────────────────────────────

cc_pages = {}
with open(CC_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["crawl"] != CC_CRAWL:
            continue
        lang = row["primary_language"].strip()
        if lang == "<unknown>":
            continue
        try:
            cc_pages[lang] = int(row["pages"])
        except ValueError:
            pass

# ── 4. ISO 639-3 mapping ──────────────────────────────────────────────────────

OVERRIDE = {
    "cmn": "zho",   # Mandarin
    "arb": "ara",   # Modern Standard Arabic
    "pes": "fas",   # Western Persian
    "zsm": "msa",   # Standard Malay
    "nob": "nor",   # Norwegian Bokmal
    "ekk": "est",   # Estonian
    "lvs": "lav",   # Standard Latvian
    "plt": "mlg",   # Plateau Malagasy
    "swh": "swa",   # Swahili
    "fuv": "ful",   # Nigerian Fulfulde
    "uzn": "uzb",   # Northern Uzbek
    "uzs": "uzb",   # Southern Uzbek
    "pbt": "pus",   # Southern Pashto
    "ory": "ori",   # Odia
    "gaz": "orm",   # West Central Oromo
    "kmr": "kur",   # Northern Kurdish
    "ckb": "kur",   # Central Kurdish
    "azj": "aze",   # North Azerbaijani
    "azb": "aze",   # South Azerbaijani
    "als": "sqi",   # Tosk Albanian
}

def code_to_iso3(code):
    return OVERRIDE.get(code.split("_")[0], code.split("_")[0])

# ── 5. Build matched dataset ──────────────────────────────────────────────────

matched   = []
unmatched = []
for row in data:
    if row["code"].startswith("eng_"):
        continue
    iso3  = code_to_iso3(row["code"])
    pages = cc_pages.get(iso3)
    if pages is None:
        unmatched.append(f"  {row['language']} ({row['code']} -> {iso3})")
        continue
    matched.append({**row, "cc_pages": pages})

print(f"Matched   : {len(matched)}")
print(f"Unmatched : {len(unmatched)}")
if unmatched:
    print("\n".join(unmatched))

# ── 6. Log regression ────────────────────────────────────────────────────────

log_x  = np.log10([r["cc_pages"] for r in matched])
y      = np.array([r["premium"]  for r in matched])
slope, intercept, r_val, p_val, _ = stats.linregress(log_x, y)
print(f"\nLog regression: r={r_val:.3f}, p={p_val:.4f}, slope={slope:.3f}")

# ── 7. Plot ───────────────────────────────────────────────────────────────────

scripts_present = sorted(set(r["script"] for r in matched))
PALETTE         = plt.cm.tab20.colors
fallback        = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(scripts_present)}

def get_color(script):
    return SCRIPT_COLORS.get(script, fallback.get(script, "#888888"))

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")

for script in scripts_present:
    pts = [r for r in matched if r["script"] == script]
    ax.scatter(
        [r["cc_pages"] for r in pts],
        [r["premium"]  for r in pts],
        color=get_color(script), label=script,
        s=55, zorder=4, alpha=0.85,
        edgecolors="white", linewidths=0.4,
    )

for r in matched:
    ax.annotate(
        r["language"],
        xy=(r["cc_pages"], r["premium"]),
        xytext=(4, 3), textcoords="offset points",
        fontsize=5, color="#444444", zorder=5,
    )

x_range = np.linspace(log_x.min(), log_x.max(), 300)
ax.plot(
    10**x_range, intercept + slope * x_range,
    color="#c0392b", linewidth=1.6, linestyle="--", zorder=3,
    label=f"Log fit  r={r_val:.2f}, p={p_val:.3f}",
)

ax.set_xscale("log")
ax.set_xlabel(f"Pages in Common Crawl ({CC_CRAWL}, log scale)", **FONT_AXIS)
ax.set_ylabel("BLT Patch Premium vs. English", **FONT_AXIS)
ax.set_title("BLT Patch Premium vs. Common Crawl Presence", **FONT_TITLE, pad=10)
ax.grid(axis="y", color="#cccccc", linewidth=0.6, linestyle=":", alpha=0.7, zorder=1)
ax.grid(axis="x", color="#cccccc", linewidth=0.4, linestyle=":", alpha=0.5, zorder=1)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#aaaaaa")
ax.tick_params(**FONT_TICK)
ax.legend(fontsize=7, framealpha=0.6, ncol=2, loc="lower left",
          title="Script", title_fontsize=7)

plt.tight_layout()
out_path = OUT_DIR / "premium_vs_cc_pages.png"
fig.savefig(out_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved -> {out_path}")