#!/usr/bin/env python3
"""
plot_llama2_correlation.py
Plots BLT patch premium vs. Llama 2 training-data volume, in two variants:
  1) raw reported token counts (as published in Touvron et al. Table 10)
  2) content-adjusted token counts, correcting for the fact that different
     languages need different numbers of Llama 2 tokens to encode the same
     amount of content

Step 0: tokenise each language's FLORES+ dev-set text (reusing the
per-language restructured result JSONs under RESULTS_DIR, which already store
the original "text" field per sentence) with the actual Llama 2 tokenizer.
Since FLORES+ is multiparallel, num_tokens_flores[lang] / num_tokens_flores[en]
gives a "token inflation factor": how many times more tokens a language needs,
relative to English, to encode the same content. Dividing each language's
reported Llama 2 token count by its inflation factor converts it into
English-content-equivalent units, making the reported percentages comparable
across languages regardless of how verbose their tokenisation happens to be.

Adjusted percentages (for the LaTeX table) are saved to ADJUSTED_OUT_CSV.
Charts are saved to charts/, one raw-token and one content-adjusted chart per
premium mode listed in PREMIUM_COLS.
"""

import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SCRIPT_LABELS, SCRIPT_COLORS

# ── Config ────────────────────────────────────────────────────────────────────

MASTER_CSV       = Path("results/results_CSV/base_model_results.csv")
RESULTS_DIR      = Path("results/base_model")   # same convention as analyse_patching_methods.py
ADJUSTED_OUT_CSV = Path("charts/llama2_adjusted_shares.csv")

TOKENIZER_NAME = "meta-llama/Llama-2-7b-hf"

# Which premium columns to correlate against. Add/remove entries as needed;
# each produces its own pair of charts (raw tokens + content-adjusted).
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

# ── Llama 2 training data (billions of tokens), from the Llama 2 paper ────────

llama2_tokens = {
    'en': 1794.0,
    'de': 3.4, 'fr': 3.2, 'sv': 3.0, 'zh': 2.6, 'es': 2.6,
    'ru': 2.6, 'nl': 2.4, 'it': 2.2, 'ja': 2.0, 'pl': 1.8,
    'pt': 1.8, 'vi': 1.6, 'uk': 1.4, 'ko': 1.2, 'ca': 0.8,
    'sr': 0.8, 'id': 0.6, 'cs': 0.6, 'fi': 0.6, 'hu': 0.6,
    'no': 0.6, 'ro': 0.6, 'bg': 0.4, 'da': 0.4, 'sl': 0.2, 'hr': 0.2,
}

# "Unknown" (incl. code), 8.38% / 167.6B in Touvron et al. Table 10. Not a
# real language, so it has no FLORES+ text and cannot be content-adjusted --
# its token count is carried through UNCHANGED into the adjusted total, so
# that adjusted_share_pct stays on a comparable 100%-sum basis to the raw
# Tok% column instead of silently redistributing Unknown's mass elsewhere.
UNKNOWN_TOKENS_B = 167.6

# Code_Orig (as it appears in the CSV / results/base_model/*.json filenames)
# -> (llama2 key, script tag)
code_map = {
    'eng_Latn': ('en', 'Latn'),
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
    'en': 'English',
    'de': 'German', 'fr': 'French', 'sv': 'Swedish', 'zh': 'Chinese',
    'es': 'Spanish', 'ru': 'Russian', 'nl': 'Dutch', 'it': 'Italian',
    'ja': 'Japanese', 'pl': 'Polish', 'pt': 'Portuguese', 'vi': 'Vietnamese',
    'uk': 'Ukrainian', 'ko': 'Korean', 'ca': 'Catalan', 'sr': 'Serbian',
    'id': 'Indonesian', 'cs': 'Czech', 'fi': 'Finnish', 'hu': 'Hungarian',
    'no': 'Norwegian', 'ro': 'Romanian', 'bg': 'Bulgarian', 'da': 'Danish',
    'sl': 'Slovenian', 'hr': 'Croatian',
}

# ── Step 0: tokenise FLORES+ per language, derive token counts ────────────────

def load_flores_text(code_orig):
    """Concatenate every sentence's 'text' field for a language from the
    restructured results JSON (same files used by inspect_results.py /
    analyse_patching_methods.py). Returns None if the file is missing."""
    fpath = RESULTS_DIR / f"{code_orig}.json"
    if not fpath.exists():
        return None
    with open(fpath, encoding="utf-8") as f:
        sentences = json.load(f)
    return "\n".join(s["text"] for s in sentences if "text" in s)


def compute_token_counts(code_map, tokenizer_name):
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    rows = []
    for code_orig, (iso, script_key) in code_map.items():
        text = load_flores_text(code_orig)
        if text is None:
            print(f"  SKIP {code_orig}: no results file at {RESULTS_DIR / (code_orig + '.json')}")
            continue

        ids = tok.encode(text, add_special_tokens=False)
        num_tokens = len(ids)
        if num_tokens == 0:
            print(f"  SKIP {code_orig}: tokenised to 0 tokens")
            continue

        rows.append({
            "code_orig":         code_orig,
            "iso":               iso,
            "language":          full_names.get(iso, iso),
            "num_tokens_flores": num_tokens,
        })
        print(f"  {code_orig:12s} ({full_names.get(iso, iso):12s})  "
              f"tokens_flores={num_tokens:>8,}")

    return pd.DataFrame(rows)


counts_df = compute_token_counts(code_map, TOKENIZER_NAME)
tokens_flores_by_iso = dict(zip(counts_df["iso"], counts_df["num_tokens_flores"]))

if "en" not in tokens_flores_by_iso:
    raise SystemExit("English ('en'/'eng_Latn') is required as the baseline for the "
                      "inflation factor, but no FLORES+ token count was found for it.")

en_tokens_flores = tokens_flores_by_iso["en"]

# inflation[lang] = how many times more Llama 2 tokens a language needs,
# relative to English, to encode the same (parallel) FLORES+ content
inflation_by_iso = {
    iso: n / en_tokens_flores for iso, n in tokens_flores_by_iso.items()
}

missing_inflation = [iso for iso in llama2_tokens if iso not in inflation_by_iso]
if missing_inflation:
    print(f"\nNo inflation factor available for: {missing_inflation} "
          f"(no matching FLORES+ results file found) -- excluded from the "
          f"content-adjusted chart and table.")

# content-adjusted token volume, in English-content-equivalent units
adjusted_tokens_B = {
    iso: tokens_B / inflation_by_iso[iso]
    for iso, tokens_B in llama2_tokens.items()
    if iso in inflation_by_iso
}

# renormalised percentage shares. Unknown is included in the total, unchanged,
# so both Tok% and True% sum to ~100% on the same basis (cf. discussion: this
# is "Option B" -- Unknown can't be content-adjusted, so it's left as-is
# rather than excluded from the denominator, which would otherwise silently
# inflate every other language's adjusted share).
total_adjusted = sum(adjusted_tokens_B.values()) + UNKNOWN_TOKENS_B
adjusted_share_pct = {
    iso: v / total_adjusted * 100 for iso, v in adjusted_tokens_B.items()
}
unknown_share_pct = UNKNOWN_TOKENS_B / total_adjusted * 100

print("\nContent-adjusted shares (renormalised, Unknown included unchanged):")
for iso, pct in sorted(adjusted_share_pct.items(), key=lambda x: -x[1]):
    print(f"  {full_names.get(iso, iso):12s} {pct:7.3f}%   "
          f"(inflation={inflation_by_iso[iso]:.3f}, adj_tokens_B={adjusted_tokens_B[iso]:.2f})")
print(f"  {'Unknown':12s} {unknown_share_pct:7.3f}%   (unchanged, not content-adjustable)")

# ── Save adjusted table (this is what feeds the LaTeX table's Percent' column) ─

# ── Save adjusted table (this is what feeds the LaTeX table's Percent' column) ─

adj_rows = []
for iso, tokens_B in llama2_tokens.items():
    adj_rows.append({
        "iso":                iso,
        "language":           full_names.get(iso, iso),
        "tokens_B_reported":  tokens_B,
        "inflation_factor":   inflation_by_iso.get(iso),
        "adjusted_tokens_B":  adjusted_tokens_B.get(iso),
        "adjusted_share_pct": adjusted_share_pct.get(iso),
    })

# Unknown: not a real language, no inflation factor, token count carried
# through unchanged (cf. UNKNOWN_TOKENS_B comment above).
adj_rows.append({
    "iso":                "unk",
    "language":           "Unknown",
    "tokens_B_reported":  UNKNOWN_TOKENS_B,
    "inflation_factor":   None,
    "adjusted_tokens_B":  UNKNOWN_TOKENS_B,
    "adjusted_share_pct": unknown_share_pct,
})

adj_out = pd.DataFrame(adj_rows).sort_values("adjusted_share_pct", ascending=False)
ADJUSTED_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
adj_out.to_csv(ADJUSTED_OUT_CSV, index=False)
print(f"\nSaved adjusted shares -> {ADJUSTED_OUT_CSV}")

# ── Load premiums from the master CSV ─────────────────────────────────────────

master = pd.read_csv(MASTER_CSV)
for label, col in PREMIUM_COLS.items():
    if col not in master.columns:
        avail = [c for c in master.columns if c.endswith("_pps_premium")]
        raise SystemExit(
            f"Column '{col}' (for '{label}') not found in {MASTER_CSV}.\n"
            "Available premium columns:\n  " + "\n  ".join(avail)
        )

# ── Build a comparable DataFrame for a given volume dict + premium column ─────

def build_df(volume_by_iso, volume_col, prem_by_code):
    rows, missing = [], []
    for code_orig, (iso, script_key) in code_map.items():
        if iso == "en":
            continue  # English is the baseline the premium is measured against
        vol = volume_by_iso.get(iso)
        if vol is None:
            continue
        premium = prem_by_code.get(code_orig)
        if premium is None or pd.isna(premium):
            missing.append(f"{full_names.get(iso, iso)} ({code_orig})")
            continue
        script_label = SCRIPT_LABELS.get(script_key, script_key)
        rows.append({
            'language':  full_names.get(iso, iso),
            'code':      iso,
            volume_col:  vol,
            'premium':   float(premium),
            'script':    script_label,
            'color':     SCRIPT_COLORS.get(script_label, '#333333'),
        })
    if missing:
        print("Not found in CSV (check Code_Orig spelling):\n  " + "\n  ".join(missing))
    df = pd.DataFrame(rows).drop_duplicates(subset='code').sort_values(volume_col)
    if df.empty:
        raise SystemExit("No languages matched between code_map and the CSV — nothing to plot.")
    return df


# ── Chart helper ────────────────────────────────────────────────────────────────

def make_chart(df, volume_col, xlabel, title, out_path):
    log_vol = np.log(df[volume_col])
    slope, intercept, r, p, se = stats.linregress(log_vol, df['premium'])
    x_fit = np.linspace(df[volume_col].min(), df[volume_col].max(), 200)
    y_fit = slope * np.log(x_fit) + intercept

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    for _, row in df.iterrows():
        ax.scatter(row[volume_col], row['premium'],
                   color=row['color'], edgecolors='white',
                   linewidths=0.6, s=75, zorder=3, alpha=0.9)
        ax.annotate(row['language'], (row[volume_col], row['premium']),
                    fontsize=6.5, ha='left', va='bottom',
                    xytext=(4, 3), textcoords='offset points', color='#333')

    ax.plot(x_fit, y_fit, color='tomato', linewidth=1.8,
            linestyle='--', label=f'Log fit  (r={r:.2f}, p={p:.3f})', zorder=2)

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

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Tokenisation premium vs. English', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#aaaaaa")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Pearson r ({volume_col}, log): {r:.3f}, p={p:.4f}, N={len(df)}  -> {out_path}")


# ── Generate raw-token and content-adjusted charts for each premium mode ──────

Path('charts').mkdir(parents=True, exist_ok=True)

for mode_label, prem_col in PREMIUM_COLS.items():
    prem_by_code = dict(
        zip(master["Code_Orig"].astype(str).str.strip(), master[prem_col])
    )
    premium_stem = prem_col.replace("_pps_premium", "")
    mode_title   = TITLE_LABELS.get(mode_label, mode_label)

    df_raw = build_df(llama2_tokens, 'tokens_B', prem_by_code)
    df_adj = build_df(adjusted_tokens_B, 'adj_tokens_B', prem_by_code)

    make_chart(
        df_raw, 'tokens_B',
        xlabel='Training data in Llama 2 (billions of tokens)',
        title=f'BLT Patch Premium vs. Llama 2 Training Data (Raw Tokens, {mode_title})',
        out_path=f'charts/premium_vs_llama2_data_{premium_stem}.png',
    )

    make_chart(
        df_adj, 'adj_tokens_B',
        xlabel='Estimated training data in Llama 2 (billions of tokens, content-adjusted)',
        title=f'BLT Patch Premium vs. Llama 2 Training Data (Content-Adjusted, {mode_title})',
        out_path=f'charts/premium_vs_llama2_data_adjusted_{premium_stem}.png',
    )