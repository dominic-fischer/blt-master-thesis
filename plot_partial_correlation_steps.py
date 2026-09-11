#!/usr/bin/env python3
"""
plot_partial_correlation_steps.py -- Visual step-by-step walkthrough of
computing the partial correlation between spread and premium, controlling
for total entropy budget.

Produces a single 6-panel figure:

  (a) Step 1a: spread vs budget, with the fitted line used to predict
      spread FROM budget.
  (b) Step 2a: the leftover (spread minus its budget-predicted value),
      plotted against budget -- should look flat/patternless, since by
      construction the residual no longer correlates with budget.
  (c) Reference only, NOT part of the pipeline: the RAW spread vs premium
      relationship, shown here (top-right) so it can be compared directly
      against the final budget-controlled panel below it (bottom-right).
  (d) Step 1b: premium vs budget, with the fitted line used to predict
      premium FROM budget.
  (e) Step 2b: the leftover (premium minus its budget-predicted value),
      plotted against budget -- also flat/patternless by construction.
  (f) Step 3 (the payoff): the two leftovers from (b) and (e) plotted
      against each other. Whatever relationship remains here can't be
      coming from budget anymore, since both axes have already had their
      shared relationship with budget removed.

USAGE
    python3 plot_partial_correlation_steps.py Balanced_raw_entropy_t_1.9458_premiums_sorted.txt

    python3 plot_partial_correlation_steps.py --spread-json byte_position_stats.json \\
        --out steps.png Balanced_raw_entropy_t_1.9458_premiums_sorted.txt

Requires: matplotlib, numpy
"""

import argparse
import json
import math
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CODE_TO_LANG_NAME = {
    "eng_Latn": "English", "cmn_Hans": "Mandarin Chinese", "deu_Latn": "German",
    "jpn_Jpan": "Japanese", "spa_Latn": "Spanish", "fra_Latn": "French",
    "ita_Latn": "Italian", "vie_Latn": "Vietnamese", "arb_Arab": "Arabic",
    "tha_Thai": "Thai", "kor_Hang": "Korean", "ron_Latn": "Romanian",
    "fin_Latn": "Finnish", "heb_Hebr": "Hebrew", "tam_Taml": "Tamil",
    "hrv_Latn": "Croatian", "srp_Cyrl": "Serbian", "kat_Geor": "Georgian",
    "amh_Ethi": "Amharic", "nya_Latn": "Chichewa",
}
LENGTH_COLORS = {1: "#4C72B0", 2: "#DD8452", 3: "#55A868", 4: "#C44E52"}


def parse_premium_table(path):
    data = {}
    header_seen = False
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not header_seen:
                header_seen = True
                continue
            if len(parts) < 6:
                continue
            try:
                data[parts[0]] = float(parts[1])  # just premium, index 1
            except ValueError:
                continue
    return data


def load_spread_and_budget(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for lang, entry in raw.items():
        spread = entry.get("spread")
        budget = entry.get("main_length_entropy_sum")
        if budget is None and entry.get("main_length_entropies") is not None:
            budget = sum(entry["main_length_entropies"])
        out[lang] = (spread, entry.get("main_length"), budget)
    return out


def pearson_r_p(x, y, extra_df_used=0):
    """Pearson r and its two-tailed p-value (H0: no correlation).

    extra_df_used: pass 1 when x/y are RESIDUALS from a regression on a
    third (control) variable -- the correct significance test for a
    partial correlation uses n-3 degrees of freedom, not the usual n-2,
    since that regression already used one up. Only the final
    leftover-vs-leftover panel needs this; the others are direct
    (non-partial) correlations.

    Uses scipy if available (via the t-distribution with the correct
    degrees of freedom); falls back to a Fisher z-transformation normal
    approximation otherwise."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1]) if n >= 2 else float("nan")
    df = n - 2 - extra_df_used
    try:
        from scipy import stats as _stats
        if df <= 0 or abs(r) >= 1:
            return r, float("nan")
        t_stat = r * math.sqrt(df / (1 - r ** 2))
        p = 2 * _stats.t.sf(abs(t_stat), df)
        return r, float(p)
    except ImportError:
        if n < 4 or abs(r) >= 1:
            return r, float("nan")
        z = math.atanh(r)
        se = 1.0 / math.sqrt(n - 3 - extra_df_used)
        zscore = z / se
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(zscore) / math.sqrt(2))))
        return r, p


def scatter_with_fit(ax, x, y, colors, labels, xlabel, ylabel, title,
                      fit_line=True, extra_df_used=0):
    ax.scatter(x, y, c=colors, s=70, edgecolors="#333333", linewidths=0.7, zorder=3)
    r, p = pearson_r_p(x, y, extra_df_used=extra_df_used)
    if fit_line and len(x) >= 2:
        coefs = np.polyfit(x, y, 1)
        xs_line = np.linspace(min(x), max(x), 50)
        p_label = f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"
        fit_handle, = ax.plot(xs_line, np.polyval(coefs, xs_line), linestyle="--",
                               color="red", linewidth=1.6, zorder=2,
                               label=f"Linear fit (r={r:.2f}, p={p_label})")
        ax.legend(handles=[fit_handle], loc="lower left", fontsize=7.5, framealpha=0.9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.tick_params(labelsize=8)
    return r, p


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("premium_file")
    parser.add_argument("--spread-json", default="byte_position_stats.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    premiums = parse_premium_table(args.premium_file)
    spread_data = load_spread_and_budget(args.spread_json)

    langs = sorted(set(premiums) & set(spread_data))
    xs_spread, ys_premium, colors, labels, budgets = [], [], [], [], []
    for lang in langs:
        spread, main_length, budget = spread_data[lang]
        if budget is None:
            continue
        if spread is None:
            spread = 0.0  # 1-byte languages: treated as zero spread, same
                           # convention as plot_spread_vs_premium.py, since
                           # there's no "across positions" to spread over
        xs_spread.append(spread)
        ys_premium.append(premiums[lang])
        budgets.append(budget)
        colors.append(LENGTH_COLORS.get(main_length, "#888888"))
        labels.append(CODE_TO_LANG_NAME.get(lang, lang))

    xs_spread = np.array(xs_spread)
    ys_premium = np.array(ys_premium)
    budgets = np.array(budgets)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) step 1a: predict spread from budget
    scatter_with_fit(axes[0, 0], budgets, xs_spread, colors, labels,
                      "Budget", "Spread", "(a) Step 1a: fit spread ~ budget")

    # (b) step 2a: spread residual (leftover after removing budget's effect)
    spread_fit = np.polyfit(budgets, xs_spread, 1)
    spread_resid = xs_spread - np.polyval(spread_fit, budgets)
    scatter_with_fit(axes[0, 1], budgets, spread_resid, colors, labels,
                      "Budget", "Spread residual",
                      "(b) Step 2a: leftover spread\n(should look flat vs. budget)")
    axes[0, 1].axhline(0, color="#999999", lw=0.8, linestyle=":")

    # (c) reference only, NOT part of the pipeline -- placed here so it
    # sits directly above the final panel (f) for an easy before/after read
    r_raw, p_raw = scatter_with_fit(axes[0, 2], xs_spread, ys_premium, colors, labels,
                                     "Spread", "Premium", "(c) Reference only: RAW spread vs. premium")
    for spine in axes[0, 2].spines.values():
        spine.set_linestyle((0, (4, 3)))
        spine.set_color("#999999")
    axes[0, 2].set_facecolor("#f5f5f5")

    # (d) step 1b: predict premium from budget
    scatter_with_fit(axes[1, 0], budgets, ys_premium, colors, labels,
                      "Budget", "Premium", "(d) Step 1b: fit premium ~ budget")

    # (e) step 2b: premium residual
    premium_fit = np.polyfit(budgets, ys_premium, 1)
    premium_resid = ys_premium - np.polyval(premium_fit, budgets)
    scatter_with_fit(axes[1, 1], budgets, premium_resid, colors, labels,
                      "Budget", "Premium residual",
                      "(e) Step 2b: leftover premium\n(should look flat vs. budget)")
    axes[1, 1].axhline(0, color="#999999", lw=0.8, linestyle=":")

    # (f) step 3: the payoff -- correlate the two leftovers. extra_df_used=1
    # since these are residuals from a regression on budget (the correct
    # significance test for a partial correlation).
    r_partial, p_partial = scatter_with_fit(
        axes[1, 2], spread_resid, premium_resid, colors, labels,
        "Spread residual", "Premium residual",
        "(f) Step 3: leftover vs. leftover\n= partial correlation",
        extra_df_used=1)
    axes[1, 2].axhline(0, color="#999999", lw=0.8, linestyle=":")
    axes[1, 2].axvline(0, color="#999999", lw=0.8, linestyle=":")

    # legend for main_length color coding, shared across all panels
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=LENGTH_COLORS[l],
                   markeredgecolor="#333333", markersize=8, label=f"{l}-byte")
        for l in sorted(set(ml for _, ml, _ in spread_data.values() if ml in LENGTH_COLORS))
    ]
    fig.legend(handles=handles, title="main_length", loc="upper center",
               ncol=len(handles), fontsize=8.5, title_fontsize=9,
               bbox_to_anchor=(0.5, 1.02))

    prefix = os.path.splitext(os.path.basename(args.premium_file))[0].split("_")[0]
    fig.suptitle(f"{prefix}: partial correlation walkthrough  --  "
                 f"raw r = {r_raw:.3f}  |  budget-controlled r = {r_partial:.3f}",
                 fontsize=13, fontweight="bold", y=1.06)

    plt.tight_layout()

    out_path = args.out
    if out_path is None:
        out_dir = os.path.dirname(os.path.abspath(args.premium_file))
        out_path = os.path.join(out_dir, f"{prefix}_partial_correlation_steps.png")

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved {out_path}")
    p_raw_str = f"{p_raw:.4f}" if p_raw >= 0.0001 else f"{p_raw:.2e}"
    p_partial_str = f"{p_partial:.4f}" if p_partial >= 0.0001 else f"{p_partial:.2e}"
    print(f"raw r(spread, premium) = {r_raw:.4f}  p = {p_raw_str}")
    print(f"partial r(spread, premium | budget) = {r_partial:.4f}  p = {p_partial_str}")


if __name__ == "__main__":
    main()