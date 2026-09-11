#!/usr/bin/env python3
"""
plot_spread_vs_premium.py -- Scatter plot of structural entropy "spread"
(from byte_position_stats.py's output) against premium (from a
premiums_sorted.txt file), one point per language.

WHY TWO SEPARATE INPUTS
    Premium is model- and run-specific (it changes with checkpoint,
    training-data balance, threshold, etc.) -- that's the file you pass
    as the main argument, e.g. a Balanced_raw_entropy_..._premiums_sorted.txt.

    Spread is NOT model-specific: byte_position_stats.py computes it
    purely from byte VALUES in the corpus text (bytes_entropies[i][0]),
    never from any model's entropy predictions. So the same
    byte_position_stats.json is valid input regardless of which
    premium run you're plotting against -- it only needs to be
    regenerated if the underlying TEXT samples change, not when you
    switch checkpoints, thresholds, or training-data balance. Hence it
    defaults to a fixed filename rather than being a required argument.

USAGE
    python3 plot_spread_vs_premium.py Balanced_raw_entropy_t_1.9458_premiums_sorted.txt

    # explicit spread-stats path if not using the default
    python3 plot_spread_vs_premium.py --spread-json path/to/byte_position_stats.json \\
        Balanced_raw_entropy_t_1.9458_premiums_sorted.txt

    # custom output path (default: saved next to the premium file)
    python3 plot_spread_vs_premium.py --out spread_vs_premium.png \\
        Balanced_raw_entropy_t_1.9458_premiums_sorted.txt

NOTE ON 1-BYTE LANGUAGES
    spread is undefined (null) for 1-byte languages in the JSON -- there's
    no "across positions" to spread across with only one position. For
    THIS PLOT specifically, null is treated as 0.0 (the same value
    Georgian/Hebrew-style fully-concentrated multi-byte languages get),
    since a 1-byte language is the limiting case of "no spread beyond a
    single byte" rather than a genuinely different quantity. If you want
    them excluded instead, pass --exclude-1byte.

Requires: matplotlib, numpy
    pip install matplotlib numpy
"""

import argparse
import json
import math
import os
import re
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

# consistent color per main_length across the whole project's charts
LENGTH_COLORS = {1: "#4C72B0", 2: "#DD8452", 3: "#55A868", 4: "#C44E52"}


def parse_premium_table(path):
    """Parse a whitespace-separated premium table file (same format used
    throughout this project) into lang_code -> (premium, mean, var)."""
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
            lang = parts[0]
            try:
                premium = float(parts[1])
                mean = float(parts[4])
                var = float(parts[5])
            except ValueError:
                continue
            data[lang] = (premium, mean, var)
    if not data:
        raise ValueError(f"No data rows parsed from {path}")
    return data


def load_spread_stats(path):
    """Load byte_position_stats.py's output -> lang_code -> (spread, main_length, budget).
    budget = main_length_entropy_sum, the total per-character identity
    entropy for that language's dominant byte-length -- what a language's
    character DENSITY looks like, independent of how that total gets
    split across positions."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    result = {}
    for lang, entry in raw.items():
        budget = entry.get("main_length_entropy_sum")
        if budget is None and entry.get("main_length_entropies") is not None:
            budget = sum(entry["main_length_entropies"])
        result[lang] = (entry.get("spread"), entry.get("main_length"), budget)
    return result


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def pearson_r_p(xs, ys, extra_df_used=0):
    """Pearson r and its two-tailed p-value (H0: no correlation).

    extra_df_used: pass 1 here when xs/ys are RESIDUALS from a regression
    on a third (control) variable -- e.g. the --control-for-budget mode.
    That regression already used up a degree of freedom, so the correct
    significance test for a partial correlation uses n-3 degrees of
    freedom, not the usual n-2. Passing this ensures the p-value isn't
    silently computed as if these were two independent raw variables.

    Uses scipy if available (via the t-distribution with the correct
    degrees of freedom); falls back to a Fisher z-transformation normal
    approximation otherwise (standard, reasonable for these sample
    sizes, but an approximation rather than the exact t-test, and the
    df correction above isn't meaningful for that approximation)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    r = float(np.corrcoef(xs, ys)[0, 1])
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


def get_prefix(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_")[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("premium_file", help="A *_premiums_sorted.txt file")
    parser.add_argument("--spread-json", default="byte_position_stats.json",
                         help="Path to byte_position_stats.py's output (default: byte_position_stats.json)")
    parser.add_argument("--out", default=None,
                         help="Output PNG path (default: saved next to premium_file)")
    parser.add_argument("--exclude-1byte", action="store_true",
                         help="Exclude 1-byte languages instead of treating their spread as 0.0")
    parser.add_argument("--control-for-budget", action="store_true",
                         help="Plot the PARTIAL relationship between spread and premium, "
                              "controlling for total entropy budget (main_length_entropy_sum). "
                              "Both spread and premium are residualized against budget before "
                              "plotting/correlating -- use this when budget and spread are "
                              "themselves correlated (e.g. CJK languages are both the highest-"
                              "budget AND the most-spread), which can otherwise mask or inflate "
                              "the raw spread-premium relationship.")
    args = parser.parse_args()

    premiums = parse_premium_table(args.premium_file)
    spreads = load_spread_stats(args.spread_json)

    langs = sorted(set(premiums) & set(spreads))
    missing = set(premiums) - set(spreads)
    if missing:
        print(f"Warning: no spread stats for {sorted(missing)}; excluded from plot.", file=sys.stderr)

    xs, ys, colors, labels, budgets = [], [], [], [], []
    for lang in langs:
        spread, main_length, budget = spreads[lang]
        if spread is None:
            if args.exclude_1byte:
                continue
            spread = 0.0
        premium, mean, var = premiums[lang]
        xs.append(spread)
        ys.append(premium)
        budgets.append(budget)
        colors.append(LENGTH_COLORS.get(main_length, "#888888"))
        labels.append(CODE_TO_LANG_NAME.get(lang, lang))

    if len(xs) < 2:
        print("Not enough matched languages with spread data to plot.", file=sys.stderr)
        sys.exit(1)

    if args.control_for_budget:
        if any(b is None for b in budgets):
            print("Warning: some languages are missing main_length_entropy_sum; "
                  "cannot control for budget.", file=sys.stderr)
            sys.exit(1)
        xs_arr, ys_arr, b_arr = np.array(xs), np.array(ys), np.array(budgets)
        # residualize BOTH spread and premium against budget -- the true
        # partial-correlation construction, not just adjusting premium alone
        spread_fit = np.polyfit(b_arr, xs_arr, 1)
        premium_fit = np.polyfit(b_arr, ys_arr, 1)
        xs = list(xs_arr - np.polyval(spread_fit, b_arr))
        ys = list(ys_arr - np.polyval(premium_fit, b_arr))
        x_label = "Spread residual (after removing budget's linear effect)"
        y_label = "Premium residual (after removing budget's linear effect)"
        title_note = "budget-controlled (partial correlation)"
    else:
        x_label = "Spread (0 = concentrated in one byte, 1 = evenly spread)"
        y_label = "Premium"
        title_note = "raw"

    r, p = pearson_r_p(xs, ys, extra_df_used=1 if args.control_for_budget else 0)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(xs, ys, c=colors, s=90, edgecolors="#333333", linewidths=0.8, zorder=3)
    if args.control_for_budget:
        ax.axhline(0, color="#999999", lw=0.8, linestyle=":", zorder=1)
        ax.axvline(0, color="#999999", lw=0.8, linestyle=":", zorder=1)

    # Fitted regression line, with its r/p folded into the legend label
    # (rather than the title) -- same "Log fit (r=..., p=...)"-style
    # legend entry used elsewhere in this project.
    xs_arr, ys_arr = np.array(xs), np.array(ys)
    fit_coefs = np.polyfit(xs_arr, ys_arr, 1)
    x_line = np.linspace(xs_arr.min(), xs_arr.max(), 100)
    y_line = np.polyval(fit_coefs, x_line)
    p_label = f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"
    fit_handle, = ax.plot(x_line, y_line, linestyle="--", color="red", linewidth=1.6,
                           zorder=2, label=f"Linear fit (r={r:.2f}, p={p_label})")

    # Points that share (nearly) the same x -- most commonly all the
    # 1-byte languages sitting at spread=0 in the RAW (non-controlled) plot
    # -- get their text labels staggered diagonally so they don't stack on
    # top of each other, regardless of how close their y-values happen to be.
    from collections import defaultdict
    x_groups = defaultdict(list)
    for i, x in enumerate(xs):
        x_groups[round(x, 2)].append(i)

    for x_key, idxs in x_groups.items():
        idxs.sort(key=lambda i: ys[i], reverse=True)
        for rank, i in enumerate(idxs):
            ax.annotate(labels[i], (xs[i], ys[i]), textcoords="offset points",
                        xytext=(8, 6 + rank * 11), fontsize=8.5, color="#333333",
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6,
                                         shrinkA=0, shrinkB=3))

    ax.set_xlabel(x_label, fontsize=10.5)
    ax.set_ylabel(y_label, fontsize=10.5)
    prefix = get_prefix(args.premium_file)
    ax.set_title(f"{prefix}: spread vs. premium ({title_note})",
                 fontsize=11.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    # legend: main_length color key + the fit line's r/p, all in one boxed
    # legend in the bottom-left corner
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=LENGTH_COLORS[l],
                   markeredgecolor="#333333", markersize=9, label=f"{l}-byte")
        for l in sorted(set(main_length for _, main_length, _ in spreads.values() if main_length in LENGTH_COLORS))
    ]
    handles.append(fit_handle)
    ax.legend(handles=handles, title="main_length", loc="lower left",
              fontsize=9, title_fontsize=9.5, framealpha=0.9)

    plt.tight_layout()

    out_path = args.out
    if out_path is None:
        out_dir = os.path.dirname(os.path.abspath(args.premium_file))
        suffix = "_spread_vs_premium_controlled.png" if args.control_for_budget else "_spread_vs_premium.png"
        out_path = os.path.join(out_dir, f"{prefix}{suffix}")

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {out_path}")
    label = "r(spread, premium | budget)" if args.control_for_budget else "r(spread, premium)"
    p_str = f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"
    print(f"Pearson {label} = {r:.4f}  p = {p_str}  (n={len(xs)})")


if __name__ == "__main__":
    main()