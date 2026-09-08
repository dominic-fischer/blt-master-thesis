#!/usr/bin/env python3
"""
script_type_anova.py
Tests whether script type explains variance in the monotonicity premium,
and compares that to how much the log(CC pages) regression explains --
i.e., does script type account for more of the premium's variance than
training-data volume does?

Run after correlate_w_cc_pages.py-style data loading (reuses load_data,
build_matched, get_script from that script).
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SCRIPT_LABELS, SCRIPT_COLORS

# Reuse the same loading logic as correlate_w_cc_pages.py
from correlate_w_cc_pages import load_data, build_matched, get_script, MASTER_CSV

PREMIUM_COL = "raw_monotonicity_t_0.3662_pps_premium"
MIN_GROUP_SIZE = 3  # drop script groups with too few points for a meaningful comparison

# ── Load data ───────────────────────────────────────────────────────────────

data = load_data(MASTER_CSV, PREMIUM_COL)
matched, unmatched = build_matched(data)
df = pd.DataFrame(matched)

print(f"Premium column : {PREMIUM_COL}")
print(f"Total matched  : {len(df)}")

# ── One-way ANOVA: does script type explain variance in premium? ──────────────

script_counts = df['script'].value_counts()
kept_scripts = script_counts[script_counts >= MIN_GROUP_SIZE].index
dropped_scripts = script_counts[script_counts < MIN_GROUP_SIZE]

if len(dropped_scripts) > 0:
    print(f"\nDropped {len(dropped_scripts)} script group(s) with < {MIN_GROUP_SIZE} "
          f"languages (too small for group comparison):")
    for s, n in dropped_scripts.items():
        print(f"  {s}: n={n}")

df_anova = df[df['script'].isin(kept_scripts)]
groups = [g['premium'].values for _, g in df_anova.groupby('script')]

f_stat, p_val = stats.f_oneway(*groups)

# eta-squared: proportion of total variance explained by script group membership
grand_mean = df_anova['premium'].mean()
ss_between = sum(
    len(g) * (g['premium'].mean() - grand_mean) ** 2
    for _, g in df_anova.groupby('script')
)
ss_total = ((df_anova['premium'] - grand_mean) ** 2).sum()
eta_sq = ss_between / ss_total

# omega-squared: bias-corrected version of eta-squared, penalizing for the
# number of groups (eta² alone can overstate explained variance simply
# because a categorical predictor with more groups has more free parameters)
k = len(kept_scripts)
n = len(df_anova)
ms_within = (ss_total - ss_between) / (n - k)
omega_sq = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)

print(f"\n--- One-way ANOVA: premium ~ script ---")
print(f"n = {len(df_anova)} (across {len(kept_scripts)} script groups, "
      f"{len(df) - len(df_anova)} languages excluded for small group size)")
print(f"F({len(kept_scripts)-1}, {len(df_anova)-len(kept_scripts)}) = {f_stat:.3f}")
print(f"p = {p_val:.4g}")
print(f"eta² = {eta_sq:.4f}  (i.e. script type explains {eta_sq*100:.1f}% of premium variance)")
print(f"omega² = {omega_sq:.4f}  (bias-corrected estimate: {omega_sq*100:.1f}%)")

# ── Per-group means, for context / a table in the thesis ──────────────────────

print("\nPer-script group means (sorted by mean premium):")
group_stats = (
    df_anova.groupby('script')['premium']
    .agg(['mean', 'std', 'count'])
    .sort_values('mean', ascending=False)
)
print(group_stats.round(3).to_string())

# ── For comparison: r² from the log(CC pages) regression on the SAME subset ───
# (so eta² and r² are computed over the identical set of languages, making the
# "which explains more" comparison apples-to-apples)

log_cc = np.log10(df_anova['cc_pages'])
slope, intercept, r_val, p_val_reg, se = stats.linregress(log_cc, df_anova['premium'])
r_sq = r_val ** 2

print(f"\n--- For comparison: log(CC pages) regression on the same {len(df_anova)} languages ---")
print(f"r = {r_val:.3f}, p = {p_val_reg:.4g}, r² = {r_sq:.4f}  "
      f"(training data explains {r_sq*100:.1f}% of premium variance)")

print(f"\n=== Summary ===")
print(f"Script type explains {eta_sq*100:.1f}% of variance (eta², p={p_val:.4g})")
print(f"Training data (log CC pages) explains {r_sq*100:.1f}% of variance (r², p={p_val_reg:.4g})")