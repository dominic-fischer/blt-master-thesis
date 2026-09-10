#!/usr/bin/env python3
"""
Premium color-chart generator.

Takes three input files:
  1. An entropy-based ranking table
  2. A monotonicity-based ranking table
  3. A per-language training-data CSV (e.g. training_setup/langs/langs_chosen.csv)

...and visualizes how each language's "premium" changes between the two
rankings, using color instead of raw numbers:

  - Premium:              green (=min in that column) -> red (=max in that column)
                          EACH COLUMN GETS ITS OWN SCALE (see note below).
  - Entropy Mean / Var:   light grey (=min) -> black (=max, across both files)
                          (shown once per language - these don't change
                          between rankings)

Training-data volume isn't color-coded; instead each language's rank by
training-data size (1 = most bytes) is shown in brackets next to its name,
e.g. "English (#1)", "Mandarin Chinese (#2)".

PREMIUM COLOR SCALE (per-column, not shared):
    The two Premium columns each get their OWN green->red scale, based on
    that column's own min/max, rather than a scale shared across both
    columns. This matters because the two rankings can have different
    minimums - e.g. under one ranking English's premium of 1.00 might be
    the lowest value in that column (and should render as pure green),
    while under the other ranking some other language dips below 1.00.
    A shared scale would use the lower of the two columns' minimums for
    BOTH columns, making a column's own true "best" value not render as
    pure green. Scoping each column to its own range fixes that.

COLUMN ORDER (left to right):
    Language (full name, with training-data rank) | Premium -> Premium | Mean | Var

Row order follows file 1's row order as-is (no re-sorting is done) - so
whatever order your entropy-based ranking file is sorted in is what the
chart displays top-to-bottom.

Expected format for files 1 & 2 (same as before):

    Language  Premium  PPS       BPP     EntropyMean  EntropyVar
    amh_Ethi  5.7097   186.6921  1.1733  2.7971       0.7736
    ...
    # comment lines starting with '#' are ignored

Expected format for file 3 (the training-data CSV): a header row plus one
row per language. The script auto-detects:
  - a language column: either a language-code column (values that look like
    "eng_Latn") or a full-name column (values like "English") - if only
    full names are present, they're mapped to codes using the built-in
    LANG_NAME_TO_CODE table below (covers the 20 languages seen so far;
    extend it if you add languages).
  - a bytes column: any column whose (normalized) name contains
    "imbalanced", "allocation" and "bytes" - falls back to any column
    containing "bytes" if that exact combination isn't found.
  Numbers may contain thousands-separator commas; those are stripped.

USAGE
    python3 txt_premiums_to_chart_alternative.py <entropy_file> <monotonicity_file> [lang_data_csv]

    lang_data_csv defaults to "training_setup/langs/langs_chosen.csv" if
    omitted.

The two ranking filenames must share the same prefix - the part of the
filename before the first underscore, e.g.:

    global_raw_entropy.txt   global_raw_monotonicity.txt
    ^^^^^^                   ^^^^^^
    (same prefix "global" -> output: global_premium_color_chart.png)

The output chart is saved in the same directory as the two ranking files,
named "<prefix>_premium_color_chart.png".

Requires: matplotlib, numpy
    pip install matplotlib numpy
"""

import argparse
import csv
import os
import re
import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Built-in language-name <-> code map. Extend this if you add languages
# that aren't covered yet.
LANG_NAME_TO_CODE = {
    "english": "eng_Latn",
    "mandarin chinese": "cmn_Hans",
    "chinese": "cmn_Hans",
    "german": "deu_Latn",
    "japanese": "jpn_Jpan",
    "spanish": "spa_Latn",
    "french": "fra_Latn",
    "italian": "ita_Latn",
    "vietnamese": "vie_Latn",
    "arabic": "arb_Arab",
    "thai": "tha_Thai",
    "korean": "kor_Hang",
    "romanian": "ron_Latn",
    "finnish": "fin_Latn",
    "hebrew": "heb_Hebr",
    "tamil": "tam_Taml",
    "croatian": "hrv_Latn",
    "serbian": "srp_Cyrl",
    "georgian": "kat_Geor",
    "amharic": "amh_Ethi",
    "chichewa": "nya_Latn",
    "nyanja": "nya_Latn",
}
CODE_TO_LANG_NAME = {
    "eng_Latn": "English", "cmn_Hans": "Mandarin Chinese", "deu_Latn": "German",
    "jpn_Jpan": "Japanese", "spa_Latn": "Spanish", "fra_Latn": "French",
    "ita_Latn": "Italian", "vie_Latn": "Vietnamese", "arb_Arab": "Arabic",
    "tha_Thai": "Thai", "kor_Hang": "Korean", "ron_Latn": "Romanian",
    "fin_Latn": "Finnish", "heb_Hebr": "Hebrew", "tam_Taml": "Tamil",
    "hrv_Latn": "Croatian", "srp_Cyrl": "Serbian", "kat_Geor": "Georgian",
    "amh_Ethi": "Amharic", "nya_Latn": "Chichewa",
}

CODE_PATTERN = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")


# ---------------------------------------------------------------------------
# 1. PARSING
# ---------------------------------------------------------------------------

def parse_table(path):
    """Parse a whitespace-separated premium table file into an ordered dict:
    lang_code -> (premium, entropy_mean, entropy_var).
    Skips blank lines, '#' comments, and the header row.
    """
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


def _normalize_colname(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _find_column(fieldnames, keyword_sets):
    """keyword_sets: list of tuples of keywords; returns first column whose
    normalized name contains ALL keywords in the first tuple that matches,
    trying tuples in order (most specific first)."""
    normed = {fn: _normalize_colname(fn) for fn in fieldnames}
    for keywords in keyword_sets:
        for fn, norm in normed.items():
            if all(kw in norm for kw in keywords):
                return fn
    return None


def parse_training_data(path):
    """Parse the training-data CSV into an ordered dict:
    lang_code -> imbalanced_allocation_bytes (float).
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows parsed from {path}")

    bytes_col = _find_column(fieldnames, [
        ("imbalanced", "allocation", "bytes"),
        ("allocation", "bytes"),
        ("bytes",),
    ])
    if bytes_col is None:
        raise ValueError(
            f"Could not find a bytes column in {path}. "
            f"Available columns: {fieldnames}"
        )

    # Prefer an explicit language-code column (values like 'eng_Latn').
    code_col = None
    for fn in fieldnames:
        sample_vals = [r[fn].strip() for r in rows[:5] if r.get(fn)]
        if sample_vals and all(CODE_PATTERN.match(v) for v in sample_vals):
            code_col = fn
            break

    name_col = None
    if code_col is None:
        name_col = _find_column(fieldnames, [("language",), ("lang", "name")])
        if name_col is None:
            raise ValueError(
                f"Could not find a language-code or language-name column in "
                f"{path}. Available columns: {fieldnames}"
            )

    data = {}
    unmapped = []
    for row in rows:
        raw_bytes = row[bytes_col].replace(",", "").strip()
        if not raw_bytes:
            continue
        try:
            n_bytes = float(raw_bytes)
        except ValueError:
            continue

        if code_col is not None:
            code = row[code_col].strip()
        else:
            name = row[name_col].strip()
            code = LANG_NAME_TO_CODE.get(name.lower())
            if code is None:
                unmapped.append(name)
                continue
        data[code] = n_bytes

    if unmapped:
        print(f"Warning: no code mapping for language name(s) {unmapped} in "
              f"{path}; add them to LANG_NAME_TO_CODE. Skipped.", file=sys.stderr)
    if not data:
        raise ValueError(f"No usable training-data rows parsed from {path}")
    return data


def get_prefix(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_")[0]


def get_label(path):
    """Human-readable label for the column header, derived from the part
    of the filename after the prefix, e.g. 'global_raw_entropy.txt' ->
    'Raw Entropy'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")[1:]
    if not parts:
        return stem
    return " ".join(p.capitalize() for p in parts)


def format_bytes(v):
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.1f}K"
    return f"{v:.0f}"


# ---------------------------------------------------------------------------
# 2. COLOR SCALES
# ---------------------------------------------------------------------------

def green_red_gradient(t):
    """t=0 -> green, t=0.5 -> yellow, t=1 -> red."""
    t = np.clip(t, 0, 1)
    if t < 0.5:
        tt = t / 0.5
        r = 0.13 + tt * (0.95 - 0.13)
        g = 0.62 + tt * (0.75 - 0.62)
        b = 0.20 + tt * (0.15 - 0.20)
    else:
        tt = (t - 0.5) / 0.5
        r = 0.95 + tt * (0.75 - 0.95)
        g = 0.75 + tt * (0.10 - 0.75)
        b = 0.15 + tt * (0.10 - 0.15)
    return (r, g, b)


def premium_color(p, p_min, p_max):
    """Low premium (column min) = green, high premium (column max) = red."""
    t = (p - p_min) / (p_max - p_min) if p_max > p_min else 0
    return green_red_gradient(t)


def grey_color(v, vmin, vmax):
    """Light grey (low) -> black (high)."""
    t = np.clip((v - vmin) / (vmax - vmin), 0, 1) if vmax > vmin else 0
    g = 0.88 - t * (0.88 - 0.05)
    return (g, g, g)


def text_color_for(bg):
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return "black" if lum > 0.55 else "white"


# ---------------------------------------------------------------------------
# 3. DRAWING
# ---------------------------------------------------------------------------

def draw_chart(table1, table2, training_data, label1, label2, out_path):
    langs = list(table1.keys())

    missing2 = [l for l in langs if l not in table2]
    if missing2:
        print(f"Warning: {missing2} present in '{label1}' file but missing "
              f"from '{label2}' file; skipped.", file=sys.stderr)
        langs = [l for l in langs if l in table2]

    missing_td = [l for l in langs if l not in training_data]
    if missing_td:
        print(f"Warning: no training-data entry for {missing_td}; "
              f"no rank shown for these.", file=sys.stderr)

    # rank languages by training-data size, most bytes = rank #1
    rank_by_lang = {
        lang: i + 1
        for i, (lang, _) in enumerate(
            sorted(training_data.items(), key=lambda kv: -kv[1])
        )
    }

    all_means = [table1[l][1] for l in langs] + [table2[l][1] for l in langs]
    all_vars = [table1[l][2] for l in langs] + [table2[l][2] for l in langs]

    # Premium: separate scale PER COLUMN (see module docstring). table1's
    # own min/max drives the left Premium column's color; table2's own
    # min/max drives the right one. NOT shared across both columns.
    p1_vals = [table1[l][0] for l in langs]
    p2_vals = [table2[l][0] for l in langs]
    p1_min, p1_max = min(p1_vals), max(p1_vals)
    p2_min, p2_max = min(p2_vals), max(p2_vals)

    m_min, m_max = min(all_means), max(all_means)
    v_min, v_max = min(all_vars), max(all_vars)

    # Wrap long group-title labels so they don't overlap each other.
    wrap_width = 16
    label1_wrapped = textwrap.fill(label1, wrap_width)
    label2_wrapped = textwrap.fill(label2, wrap_width)
    title_lines = max(label1_wrapped.count("\n"), label2_wrapped.count("\n")) + 1
    header_line_h = 0.4
    extra_top = header_line_h * (title_lines - 1)

    n = len(langs)
    fig_h = n * 0.5 + 3.2 + extra_top   # +3.2 to fit the 2-row legend below
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.set_xlim(0, 7.6)
    ax.set_ylim(-3.4, n + 1.5 + extra_top)
    ax.axis("off")

    sq = 0.8

    # column order: Language (name + training-data rank) | Premium -> Premium | Mean | Var
    x_lang = 0.2
    x1_p = 2.3
    x_arrow = 3.4
    x2_p = 4.4
    x_mean = 5.6
    x_var = 6.5

    def draw_square(x, y, color, val, fmt="{:.2f}"):
        rect = patches.FancyBboxPatch(
            (x, y), sq, sq,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0.6, edgecolor="#333333", facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + sq / 2, y + sq / 2, fmt.format(val) if not isinstance(val, str) else val,
                ha="center", va="center", fontsize=7.2,
                color=text_color_for(color), fontweight="bold")

    ytop = n + 0.6 + extra_top
    ax.text(x_lang, ytop, "Language", fontsize=10, fontweight="bold", va="bottom")
    ax.text(x1_p + sq / 2, ytop, "Premium", fontsize=9, fontweight="bold", ha="center", va="bottom")
    ax.text(x2_p + sq / 2, ytop, "Premium", fontsize=9, fontweight="bold", ha="center", va="bottom")
    ax.text(x_mean + sq / 2, ytop, "Mean", fontsize=9, fontweight="bold", ha="center", va="bottom")
    ax.text(x_var + sq / 2, ytop, "Var", fontsize=9, fontweight="bold", ha="center", va="bottom")

    ax.text(x1_p + sq / 2, ytop + 0.55, label1_wrapped, fontsize=10.5, fontweight="bold",
            ha="center", va="bottom", multialignment="center", color="#444444")
    ax.text(x2_p + sq / 2, ytop + 0.55, label2_wrapped, fontsize=10.5, fontweight="bold",
            ha="center", va="bottom", multialignment="center", color="#444444")

    for i, lang in enumerate(langs):
        y = n - 1 - i + 0.3
        p1, mean, var = table1[lang]
        p2, _, _ = table2[lang]
        lang_name = CODE_TO_LANG_NAME.get(lang, lang)
        if lang in rank_by_lang:
            lang_label = f"{lang_name} (#{rank_by_lang[lang]})"
        else:
            lang_label = lang_name

        ax.text(x_lang, y + sq / 2, lang_label, fontsize=9.5, va="center", fontweight="medium")

        draw_square(x1_p, y, premium_color(p1, p1_min, p1_max), p1)
        ax.annotate("", xy=(x_arrow + 0.6, y + sq / 2), xytext=(x_arrow - 0.15, y + sq / 2),
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.3))
        draw_square(x2_p, y, premium_color(p2, p2_min, p2_max), p2)

        draw_square(x_mean, y, grey_color(mean, m_min, m_max), mean)
        draw_square(x_var, y, grey_color(var, v_min, v_max), var)

    # ---- legend (2 rows: premium x2 on top, mean/var below) ----
    grad_w = 2.0
    n_steps = 60

    def draw_gradient_legend(x0, y0, vmin, vmax, color_fn, label, label_formatter=None):
        if label_formatter is None:
            label_formatter = lambda v: f"{v:.2f}"
        for k in range(n_steps):
            t0 = k / n_steps
            val = vmin + t0 * (vmax - vmin)
            ax.add_patch(patches.Rectangle(
                (x0 + t0 * grad_w, y0), grad_w / n_steps + 0.001, 0.25,
                color=color_fn(val), linewidth=0,
            ))
        ax.text(x0, y0 - 0.3, label_formatter(vmin), fontsize=8, ha="left")
        ax.text(x0 + grad_w, y0 - 0.3, label_formatter(vmax), fontsize=8, ha="right")
        ax.text(x0 + grad_w / 2, y0 + 0.5, label, fontsize=8.5, ha="center",
                multialignment="center")

    row1_y, row2_y = -1.2, -2.6
    draw_gradient_legend(0.2, row1_y, p1_min, p1_max,
                          lambda v: premium_color(v, p1_min, p1_max),
                          "Left-col Premium\n(green=min\u2192red=max)")
    draw_gradient_legend(3.6, row1_y, p2_min, p2_max,
                          lambda v: premium_color(v, p2_min, p2_max),
                          "Right-col Premium\n(green=min\u2192red=max)")
    draw_gradient_legend(0.2, row2_y, m_min, m_max,
                          lambda v: grey_color(v, m_min, m_max),
                          "Entropy Mean (light\u2192dark)")
    draw_gradient_legend(3.6, row2_y, v_min, v_max,
                          lambda v: grey_color(v, v_min, v_max),
                          "Entropy Variance (light\u2192dark)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate a premium color chart from two ranking files and a training-data CSV.")
    parser.add_argument("file1", help="First ranking table file (e.g. entropy-based)")
    parser.add_argument("file2", help="Second ranking table file (e.g. monotonicity-based)")
    parser.add_argument("lang_data_csv", nargs="?",
                         default="training_setup/langs/langs_chosen.csv",
                         help="Training-data CSV (default: training_setup/langs/langs_chosen.csv)")
    args = parser.parse_args()

    prefix1 = get_prefix(args.file1)
    prefix2 = get_prefix(args.file2)
    assert prefix1 == prefix2, (
        f"Ranking filenames must share the same prefix (part before the first "
        f"underscore). Got '{prefix1}' from '{args.file1}' and '{prefix2}' "
        f"from '{args.file2}'."
    )

    table1 = parse_table(args.file1)
    table2 = parse_table(args.file2)
    training_data = parse_training_data(args.lang_data_csv)
    label1 = get_label(args.file1)
    label2 = get_label(args.file2)

    out_dir = os.path.dirname(os.path.abspath(args.file1))
    out_path = os.path.join(out_dir, f"{prefix1}_premium_color_chart.png")
    draw_chart(table1, table2, training_data, label1, label2, out_path)


if __name__ == "__main__":
    main()