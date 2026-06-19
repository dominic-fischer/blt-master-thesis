#!/usr/bin/env python3
"""
plot_cc_correlation.py
Standalone script. Reads florespluis_MASTER_CSV.csv — which now contains BOTH the
BLT patch-premium results and the Common Crawl page counts in a single file — then
plots BLT patch premium vs. Common Crawl page count with per-script colouring and a
log-scale x-axis.

(Previously this joined results/summary.txt against commoncrawl_stats.csv via an
ISO 639-3 remapping. The master CSV already carries CC_Pages per row, so that whole
join — the second CSV, the OVERRIDE table, and code_to_iso3 — is gone.)
"""

import csv
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

MASTER_CSV = Path("floresplus_MASTER_CSV.csv")
OUT_DIR    = Path("charts/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cosmetic label for the x-axis only (the master CSV no longer tags the crawl).
CC_CRAWL = "CC-MAIN-2026-17"

# Which premium column to plot on the y-axis. Every option below is a
# "<model>_t_<threshold>_pps_premium" column present in the CSV:
#
#   combined_t_0.0000          combined_t_0.2578          combined_t_0.4688          combined_t_0.9375
#   norm_entropy_t_0.4072      norm_entropy_t_0.5293      norm_entropy_t_0.7222      norm_entropy_t_1.0371
#   raw_entropy_t_1.1904       raw_entropy_t_1.3340       raw_entropy_t_1.4946       raw_entropy_t_1.7988
#   raw_monotonicity_t_0.2359  raw_monotonicity_t_0.3662  raw_monotonicity_t_0.5928  raw_monotonicity_t_0.9496
#
# Set the full column name (i.e. with the "_pps_premium" suffix).
PREMIUM_COL = "raw_entropy_t_1.3340_pps_premium"

# ── Style ─────────────────────────────────────────────────────────────────────

FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "color": "#1a1a2e"}
FONT_AXIS  = {"fontsize": 9,  "color": "#333333"}
FONT_TICK  = {"labelsize": 8, "labelcolor": "#444444"}

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

data = load_data(MASTER_CSV, PREMIUM_COL)

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

print(f"Premium column : {PREMIUM_COL}")
print(f"Matched        : {len(matched)}")
print(f"Unmatched      : {len(unmatched)}")
if unmatched:
    print("\n".join(unmatched))

if not matched:
    raise SystemExit("No rows with a usable CC_Pages value — nothing to plot.")

# ── 4. Log regression ─────────────────────────────────────────────────────────

log_x  = np.log10([r["cc_pages"] for r in matched])
y      = np.array([r["premium"]  for r in matched])
slope, intercept, r_val, p_val, _ = stats.linregress(log_x, y)
print(f"\nLog regression: r={r_val:.3f}, p={p_val:.4f}, slope={slope:.3f}")

# ── 5. Plot ───────────────────────────────────────────────────────────────────

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
out_path = OUT_DIR / f"premium_vs_cc_pages_{PREMIUM_COL.replace('_pps_premium', '')}.png"
fig.savefig(out_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved -> {out_path}")