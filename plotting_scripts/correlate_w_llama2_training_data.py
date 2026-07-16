#!/usr/bin/env python3
"""
plot_llama2_correlation.py
Plots BLT patch premium vs. Llama 2 training-data volume (billions of tokens),
with per-script colouring and a log-scale x-axis.

The Llama 2 token counts are not in the data file, so they stay hardcoded below.
The only thing that now comes from florespluis_MASTER_CSV.csv is the premium,
matched per language via the Code_Orig column (e.g. 'deu_Latn').
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SCRIPT_LABELS, SCRIPT_COLORS

# ── Config ────────────────────────────────────────────────────────────────────

MASTER_CSV = Path("floresplus_MASTER.csv")

# Which premium column to plot. Same 16 options as the CC script — set the full
# "<model>_t_<threshold>_pps_premium" name. Keep this in sync with that script if
# you want the two charts to describe the same premium.
PREMIUM_COL = "raw_entropy_t_1.3340_pps_premium"

# ── Llama 2 training data (billions of tokens), from the Llama 2 paper ────────

llama2_tokens = {
    'de': 3.4, 'fr': 3.2, 'sv': 3.0, 'zh': 2.6, 'es': 2.6,
    'ru': 2.6, 'nl': 2.4, 'it': 2.2, 'ja': 2.0, 'pl': 1.8,
    'pt': 1.8, 'vi': 1.6, 'uk': 1.4, 'ko': 1.2, 'ca': 0.8,
    'sr': 0.8, 'id': 0.6, 'cs': 0.6, 'fi': 0.6, 'hu': 0.6,
    'no': 0.6, 'ro': 0.6, 'bg': 0.4, 'da': 0.4, 'sl': 0.2, 'hr': 0.2,
}

# Code_Orig (as it appears in the CSV)  ->  (llama2 key, script tag)
code_map = {
    'deu_Latn': ('de', 'Latn'), 'fra_Latn': ('fr', 'Latn'),
    'swe_Latn': ('sv', 'Latn'), 'cmn_Hans': ('zh', 'Hans'),
    'spa_Latn': ('es', 'Latn'), 'rus_Cyrl': ('ru', 'Cyrl'),
    'nld_Latn': ('nl', 'Latn'), 'ita_Latn': ('it', 'Latn'),
    'jpn_Jpan': ('ja', 'Jpan'), 'pol_Latn': ('pl', 'Latn'),
    'por_Latn': ('pt', 'Latn'), 'vie_Latn': ('vi', 'Latn'),
    'ukr_Cyrl': ('uk', 'Cyrl'), 'kor_Hang': ('ko', 'Hang'),
    'cat_Latn': ('ca', 'Latn'), 'srp_Cyrl': ('sr', 'Cyrl'),
    'ind_Latn': ('id', 'Latn'), 'ces_Latn': ('cs', 'Latn'),
    'fin_Latn': ('fi', 'Latn'), 'hun_Latn': ('hu', 'Latn'),
    'nob_Latn': ('no', 'Latn'), 'ron_Latn': ('ro', 'Latn'),
    'bul_Cyrl': ('bg', 'Cyrl'), 'dan_Latn': ('da', 'Latn'),
    'slv_Latn': ('sl', 'Latn'), 'hrv_Latn': ('hr', 'Latn'),
}

full_names = {
    'de': 'German', 'fr': 'French', 'sv': 'Swedish', 'zh': 'Chinese',
    'es': 'Spanish', 'ru': 'Russian', 'nl': 'Dutch', 'it': 'Italian',
    'ja': 'Japanese', 'pl': 'Polish', 'pt': 'Portuguese', 'vi': 'Vietnamese',
    'uk': 'Ukrainian', 'ko': 'Korean', 'ca': 'Catalan', 'sr': 'Serbian',
    'id': 'Indonesian', 'cs': 'Czech', 'fi': 'Finnish', 'hu': 'Hungarian',
    'no': 'Norwegian', 'ro': 'Romanian', 'bg': 'Bulgarian', 'da': 'Danish',
    'sl': 'Slovenian', 'hr': 'Croatian',
}

# ── Load premiums from the master CSV ─────────────────────────────────────────

master = pd.read_csv(MASTER_CSV)
if PREMIUM_COL not in master.columns:
    avail = [c for c in master.columns if c.endswith("_pps_premium")]
    raise SystemExit(
        f"Column '{PREMIUM_COL}' not found in {MASTER_CSV}.\n"
        "Available premium columns:\n  " + "\n  ".join(avail)
    )

# Code_Orig -> premium
prem_by_code = dict(
    zip(master["Code_Orig"].astype(str).str.strip(), master[PREMIUM_COL])
)

rows, missing = [], []
for code_orig, (iso, script_key) in code_map.items():
    tokens = llama2_tokens.get(iso)
    if tokens is None:
        continue
    premium = prem_by_code.get(code_orig)
    if premium is None or pd.isna(premium):
        missing.append(f"{full_names.get(iso, iso)} ({code_orig})")
        continue
    script_label = SCRIPT_LABELS.get(script_key, script_key)
    rows.append({
        'language': full_names.get(iso, iso),
        'code':     iso,
        'tokens_B': tokens,
        'premium':  float(premium),
        'script':   script_label,
        'color':    SCRIPT_COLORS.get(script_label, '#333333'),
    })

if missing:
    # Most often a code mismatch — e.g. the CSV may use 'zho_Hans' where code_map
    # has 'cmn_Hans'. Fix the code_map key if a language you expected is listed here.
    print("Not found in CSV (check Code_Orig spelling):\n  " + "\n  ".join(missing))

df = pd.DataFrame(rows).drop_duplicates(subset='code').sort_values('tokens_B')
if df.empty:
    raise SystemExit("No languages matched between code_map and the CSV — nothing to plot.")

# ── Log regression (premium ~ a*ln(tokens) + b) ───────────────────────────────

log_tokens = np.log(df['tokens_B'])
slope, intercept, r, p, se = stats.linregress(log_tokens, df['premium'])
x_fit = np.linspace(df['tokens_B'].min(), df['tokens_B'].max(), 200)
y_fit = slope * np.log(x_fit) + intercept

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")

for _, row in df.iterrows():
    ax.scatter(row['tokens_B'], row['premium'],
               color=row['color'], edgecolors='white',
               linewidths=0.6, s=75, zorder=3, alpha=0.9)
    ax.annotate(row['language'], (row['tokens_B'], row['premium']),
                fontsize=6.5, ha='left', va='bottom',
                xytext=(4, 3), textcoords='offset points', color='#333')

ax.plot(x_fit, y_fit, color='tomato', linewidth=1.8,
        linestyle='--', label=f'Log fit  (r={r:.2f}, p={p:.3f})', zorder=2)

# script colour legend
present_scripts = df['script'].unique()
script_handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=SCRIPT_COLORS[s], markersize=8, label=s)
    for s in present_scripts if s in SCRIPT_COLORS
]
script_handles.append(
    plt.Line2D([0], [1], color='tomato', linewidth=1.8,
               linestyle='--', label=f'Log fit (r={r:.2f}, p={p:.3f})')
)
ax.legend(handles=script_handles, title='Script', fontsize=8,
          title_fontsize=9, loc='lower left', framealpha=0.9)

ax.set_xlabel('Training data in Llama 2 (billions of tokens)', fontsize=11)
ax.set_ylabel('Tokenisation premium vs. English', fontsize=11)
ax.set_title('BLT Patch Premium vs. Llama 2 Training Data',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xscale('log')
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#aaaaaa")

plt.tight_layout()
Path('charts').mkdir(parents=True, exist_ok=True)
plt.savefig(f'charts/premium_vs_llama2_data_{PREMIUM_COL.replace("_pps_premium", "")}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Pearson r (log tokens vs premium): {r:.3f}, p={p:.4f}, N={len(df)}")