#!/usr/bin/env python3
"""
script_type_anova.py
One-way ANOVA of premium on script type, for both monotonicity and
normalisation, to test whether script structure explains variance in the
premium better than training-data volume does.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from correlate_w_cc_pages import load_data, build_matched, MASTER_CSV

PREMIUM_COLS = {
    "monotonicity":  "raw_monotonicity_t_0.3662_pps_premium",
    "normalisation": "norm_entropy_t_0.5293_pps_premium",
}

MIN_GROUP_SIZE = 3


def run_anova(premium_col, mode_label):
    data = load_data(MASTER_CSV, premium_col)
    matched, unmatched = build_matched(data)
    df = pd.DataFrame(matched)

    script_counts = df['script'].value_counts()
    kept_scripts = script_counts[script_counts >= MIN_GROUP_SIZE].index
    dropped_scripts = script_counts[script_counts < MIN_GROUP_SIZE]

    df_anova = df[df['script'].isin(kept_scripts)]
    groups = [g['premium'].values for _, g in df_anova.groupby('script')]

    f_stat, p_val = stats.f_oneway(*groups)

    grand_mean = df_anova['premium'].mean()
    ss_between = sum(
        len(g) * (g['premium'].mean() - grand_mean) ** 2
        for _, g in df_anova.groupby('script')
    )
    ss_total = ((df_anova['premium'] - grand_mean) ** 2).sum()
    eta_sq = ss_between / ss_total

    k = len(kept_scripts)
    n = len(df_anova)
    ms_within = (ss_total - ss_between) / (n - k)
    omega_sq = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)

    print(f"\n{'='*70}\n{mode_label.upper()}\n{'='*70}")
    print(f"Premium column: {premium_col}")
    print(f"Total matched: {len(df)}")
    print(f"Dropped {len(dropped_scripts)} script group(s) with < {MIN_GROUP_SIZE} languages: "
          f"{dict(dropped_scripts)}")

    print(f"\n--- One-way ANOVA: premium ~ script ---")
    print(f"n = {n} (across {k} script groups, {len(df) - n} languages excluded)")
    print(f"F({k-1}, {n-k}) = {f_stat:.3f}")
    print(f"p = {p_val:.4g}")
    print(f"eta² = {eta_sq:.4f}  ({eta_sq*100:.1f}% of variance)")
    print(f"omega² = {omega_sq:.4f}  ({omega_sq*100:.1f}% bias-corrected)")

    print("\nPer-script group means (sorted by mean premium):")
    group_stats = (
        df_anova.groupby('script')['premium']
        .agg(['mean', 'std', 'count'])
        .sort_values('mean')
    )
    print(group_stats.round(3).to_string())

    log_cc = np.log10(df_anova['cc_pages'])
    slope, intercept, r_val, p_val_reg, se = stats.linregress(log_cc, df_anova['premium'])
    r_sq = r_val ** 2
    print(f"\n--- For comparison: log(CC pages) regression on the same {n} languages ---")
    print(f"r = {r_val:.3f}, p = {p_val_reg:.4g}, r² = {r_sq:.4f}  "
          f"({r_sq*100:.1f}% explained)")

    return {
        "mode": mode_label, "n": n, "k": k,
        "f_stat": f_stat, "p_val": p_val,
        "eta_sq": eta_sq, "omega_sq": omega_sq,
        "r_val": r_val, "p_val_reg": p_val_reg, "r_sq": r_sq,
        "kept_scripts": list(kept_scripts),
        "group_stats": group_stats,
    }


results = {}
for mode_label, premium_col in PREMIUM_COLS.items():
    results[mode_label] = run_anova(premium_col, mode_label)

print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
for mode_label, r in results.items():
    print(f"{mode_label:15s}  eta²={r['eta_sq']:.3f}  omega²={r['omega_sq']:.3f}  "
          f"F={r['f_stat']:.1f}  p={r['p_val']:.3g}  |  "
          f"CC r²={r['r_sq']:.3f}  p={r['p_val_reg']:.3g}")