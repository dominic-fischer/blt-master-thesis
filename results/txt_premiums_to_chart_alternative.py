#!/usr/bin/env python3
"""
Premium color-chart generator.

Takes three input files:
  1. An entropy-based ranking table
  2. A monotonicity-based ranking table
  3. A per-language training-data CSV (e.g. training_setup/langs/langs_chosen.csv)

...and visualizes how each language's "premium" changes between the two
rankings, using color instead of raw numbers.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

COLUMN_ALIASES = {
    "pps": "PPS",
    "bpp": "BPP",
    "mean": "EntropyMean", "entropymean": "EntropyMean",
    "var": "EntropyVar", "variance": "EntropyVar", "entropyvar": "EntropyVar",
    "skew": "EntropySkew", "skewness": "EntropySkew", "entropyskew": "EntropySkew",
    "kurtosis": "EntropyKurtosis", "kurt": "EntropyKurtosis", "entropykurtosis": "EntropyKurtosis",
    "autocorr": "EntropyAutocorr1", "autocorrelation": "EntropyAutocorr1",
    "autocorr1": "EntropyAutocorr1", "entropyautocorr1": "EntropyAutocorr1",
    "volatility": "EntropyVolatility", "vol": "EntropyVolatility", "entropyvolatility": "EntropyVolatility",
}

SPREAD_ALIASES = {
    "spread": "spread", "entropy_spread": "spread",
    "spread_customenc": "spread_customenc", "custom_spread": "spread_customenc",
    "spread_custom_encoding": "spread_customenc",
    "spread_sum": "main_length_entropy_sum", "entropy_sum": "main_length_entropy_sum",
    "spread_customenc_sum": "main_length_entropy_sum_customenc",
}

SPREAD_FIELD_SOURCE = {
    "spread": ("spread_json", "spread"),
    "main_length_entropy_sum": ("spread_json", "main_length_entropy_sum"),
    "spread_customenc": ("spread_customenc_json", "spread"),
    "main_length_entropy_sum_customenc": ("spread_customenc_json", "main_length_entropy_sum"),
}


def load_spread_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_col_spec(name, header_fields):
    key = name.strip().lower()
    if key in SPREAD_ALIASES:
        return "spread", SPREAD_ALIASES[key]
    resolved = COLUMN_ALIASES.get(key)
    if resolved is None:
        for h in header_fields:
            if h.lower() == key:
                resolved = h
                break
    if resolved is None or resolved not in header_fields:
        raise ValueError(
            f"Column '{name}' not found in header {header_fields}, and not "
            f"a recognized spread alias either."
        )
    return "file", resolved


def parse_table(path, file_extra_cols, spread_cols, spread_sources):
    """Parses table file and injects external spread columns if requested."""
    data = {}
    header_fields = None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if header_fields is None:
                header_fields = parts
                continue
            if len(parts) < len(header_fields):
                continue
            row = dict(zip(header_fields, parts))
            lang = row.get("Language", parts[0])
            try:
                premium = float(row["Premium"])
                extras = {c: float(row[c]) for c in file_extra_cols}
            except (KeyError, ValueError):
                continue

            # Resolve external spread values
            missing_spread = False
            for col in spread_cols:
                src_dict_name, field_name = SPREAD_FIELD_SOURCE[col]
                src_dict = spread_sources.get(src_dict_name, {})
                val = src_dict.get(lang, {}).get(field_name)
                if val is None:
                    missing_spread = True
                    break
                extras[col] = float(val)

            if not missing_spread:
                data[lang] = (premium, extras)

    if not data:
        raise ValueError(f"No valid data rows parsed from {path}")
    return data


def peek_header(path):
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            return line.split()
    raise ValueError(f"No header line found in {path}")


def _normalize_colname(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _find_column(fieldnames, keyword_sets):
    normed = {fn: _normalize_colname(fn) for fn in fieldnames}
    for keywords in keyword_sets:
        for fn, norm in normed.items():
            if all(kw in norm for kw in keywords):
                return fn
    return None


def parse_training_data(path, is_balanced=False):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows parsed from {path}")

    if is_balanced:
        bytes_col = _find_column(fieldnames, [
            ("balanced", "allocation", "bytes"),
            ("balanced", "bytes"),
            ("allocation", "bytes"),
            ("bytes",),
        ])
    else:
        bytes_col = _find_column(fieldnames, [
            ("imbalanced", "allocation", "bytes"),
            ("imbalanced", "bytes"),
            ("allocation", "bytes"),
            ("bytes",),
        ])

    if bytes_col is None:
        raise ValueError(f"Could not find a matching bytes column in {path}.")

    code_col = None
    for fn in fieldnames:
        sample_vals = [r[fn].strip() for r in rows[:5] if r.get(fn)]
        if sample_vals and all(CODE_PATTERN.match(v) for v in sample_vals):
            code_col = fn
            break

    name_col = None
    if code_col is None:
        name_col = _find_column(fieldnames, [("language",), ("lang", "name")])

    data = {}
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
                continue
        data[code] = n_bytes

    return data


def get_prefix(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_")[0]


def get_label(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")[1:]
    if not parts:
        return stem
    return " ".join(p.capitalize() for p in parts)


def green_red_gradient(t):
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
    t = (p - p_min) / (p_max - p_min) if p_max > p_min else 0
    return green_red_gradient(t)


def grey_color(v, vmin, vmax):
    t = np.clip((v - vmin) / (vmax - vmin), 0, 1) if vmax > vmin else 0
    g = 0.88 - t * (0.88 - 0.05)
    return (g, g, g)


def text_color_for(bg):
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return "black" if lum > 0.55 else "white"


def draw_chart(table1, table2, training_data, label1, label2, display_cols, out_path):
    langs = list(table1.keys())
    langs = [l for l in langs if l in table2]

    rank_by_lang = {
        lang: i + 1
        for i, (lang, _) in enumerate(
            sorted(training_data.items(), key=lambda kv: -kv[1])
        )
    }

    extra_minmax = {}
    for col in display_cols:
        vals = [table1[l][1][col] for l in langs] + [table2[l][1][col] for l in langs]
        extra_minmax[col] = (min(vals), max(vals))

    p1_vals = [table1[l][0] for l in langs]
    p2_vals = [table2[l][0] for l in langs]
    p1_min, p1_max = min(p1_vals), max(p1_vals)
    p2_min, p2_max = min(p2_vals), max(p2_vals)

    mins_equal = math.isclose(p1_min, p2_min, rel_tol=1e-9, abs_tol=1e-6)

    if mins_equal:
        shared_min = p1_min
        shared_max = max(p1_max, p2_max)
        scale1_min, scale1_max = shared_min, shared_max
        scale2_min, scale2_max = shared_min, shared_max
    else:
        span1 = p1_max - p1_min
        span2 = p2_max - p2_min
        if span1 >= span2:
            scale1_min, scale1_max = p1_min, p1_max
            scale2_min, scale2_max = p2_min, p2_min + span1
        else:
            scale1_min, scale1_max = p1_min, p1_min + span2
            scale2_min, scale2_max = p2_min, p2_max

    wrap_width = 16
    label1_wrapped = textwrap.fill(label1, wrap_width)
    label2_wrapped = textwrap.fill(label2, wrap_width)
    title_lines = max(label1_wrapped.count("\n"), label2_wrapped.count("\n")) + 1
    header_line_h = 0.4
    extra_top = header_line_h * (title_lines - 1)

    n = len(langs)
    fig_h = n * 0.5 + 2.8 + extra_top
    sq = 0.8

    x_lang = 0.2
    x1_p = 2.3
    x_arrow = 3.4
    x2_p = 4.4
    x_extra_start = 5.6
    col_spacing = 1.0
    n_extra = max(len(display_cols), 1)
    x_extra = {col: x_extra_start + i * col_spacing for i, col in enumerate(display_cols)}
    fig_w = x_extra_start + n_extra * col_spacing + 0.6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(-2.7, n + 1.5 + extra_top)
    ax.axis("off")

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
    for col in display_cols:
        ax.text(x_extra[col] + sq / 2, ytop, col, fontsize=9, fontweight="bold", ha="center", va="bottom")

    ax.text(x1_p + sq / 2, ytop + 0.55, label1_wrapped, fontsize=10.5, fontweight="bold",
            ha="center", va="bottom", multialignment="center", color="#444444")
    ax.text(x2_p + sq / 2, ytop + 0.55, label2_wrapped, fontsize=10.5, fontweight="bold",
            ha="center", va="bottom", multialignment="center", color="#444444")

    for i, lang in enumerate(langs):
        y = n - 1 - i + 0.3
        p1, extras1 = table1[lang]
        p2, extras2 = table2[lang]
        lang_name = CODE_TO_LANG_NAME.get(lang, lang)
        lang_label = f"{lang_name} (#{rank_by_lang[lang]})" if lang in rank_by_lang else lang_name

        ax.text(x_lang, y + sq / 2, lang_label, fontsize=9.5, va="center", fontweight="medium")

        draw_square(x1_p, y, premium_color(p1, scale1_min, scale1_max), p1)
        ax.annotate("", xy=(x_arrow + 0.6, y + sq / 2), xytext=(x_arrow - 0.15, y + sq / 2),
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.3))
        draw_square(x2_p, y, premium_color(p2, scale2_min, scale2_max), p2)

        for col in display_cols:
            vmin, vmax = extra_minmax[col]
            draw_square(x_extra[col], y, grey_color(extras1[col], vmin, vmax), extras1[col])

    # Legend rendering
    grad_w = 2.0
    n_steps = 60

    def draw_gradient_legend(x0, y0, vmin, vmax, color_fn, label):
        for k in range(n_steps):
            t0 = k / n_steps
            val = vmin + t0 * (vmax - vmin)
            ax.add_patch(patches.Rectangle(
                (x0 + t0 * grad_w, y0), grad_w / n_steps + 0.001, 0.25,
                color=color_fn(val), linewidth=0,
            ))
        ax.text(x0, y0 - 0.3, f"{vmin:.2f}", fontsize=8, ha="left")
        ax.text(x0 + grad_w, y0 - 0.3, f"{vmax:.2f}", fontsize=8, ha="right")
        ax.text(x0 + grad_w / 2, y0 + 0.35, label, fontsize=8.5, ha="center", multialignment="center")

    def draw_premium_legend(x0, y0, c1_min, c1_max, s1_min, s1_max, c2_min, c2_max, s2_min, s2_max):
        for k in range(n_steps):
            t0 = k / n_steps
            ax.add_patch(patches.Rectangle(
                (x0 + t0 * grad_w, y0), grad_w / n_steps + 0.001, 0.25,
                color=green_red_gradient(t0), linewidth=0,
            ))
        ax.text(x0 + grad_w / 2, y0 + 0.35, "Premium\n(green=min\u2192red=max)",
                fontsize=8.5, ha="center", va="bottom", multialignment="center")

        def place_tick(val, ref_min, ref_max, y_offset):
            rel_pos = np.clip((val - ref_min) / (ref_max - ref_min), 0, 1) if ref_max > ref_min else 0
            x_pos = x0 + rel_pos * grad_w
            ha = "left" if rel_pos < 0.1 else ("right" if rel_pos > 0.9 else "center")
            ax.text(x_pos, y0 + y_offset, f"{val:.2f}", fontsize=8, ha=ha, color="#111111", fontweight="semibold")

        def place_span(c_min, c_max, ref_min, ref_max, y_offset):
            rel_min = np.clip((c_min - ref_min) / (ref_max - ref_min), 0, 1) if ref_max > ref_min else 0
            rel_max = np.clip((c_max - ref_min) / (ref_max - ref_min), 0, 1) if ref_max > ref_min else 0
            span_rel = rel_max - rel_min
            delta = c_max - c_min
            pos_x = x0 + rel_max * grad_w + 0.28 if span_rel < 0.35 else x0 + ((rel_min + rel_max) / 2.0) * grad_w
            ha = "left" if span_rel < 0.35 else "center"
            ax.text(pos_x, y0 + y_offset, f"(\u0394 {delta:.2f})", fontsize=7.5, ha=ha, color="#555555", fontstyle="italic")

        place_tick(c1_min, s1_min, s1_max, -0.28)
        place_span(c1_min, c1_max, s1_min, s1_max, -0.28)
        place_tick(c1_max, s1_min, s1_max, -0.28)

        place_tick(c2_min, s2_min, s2_max, -0.65)
        place_span(c2_min, c2_max, s2_min, s2_max, -0.65)
        place_tick(c2_max, s2_min, s2_max, -0.65)

    leg_y = -1.2
    draw_premium_legend(0.2, leg_y, p1_min, p1_max, scale1_min, scale1_max, p2_min, p2_max, scale2_min, scale2_max)

    legend_x0 = 2.8
    legend_spacing = 2.6
    for i, col in enumerate(display_cols):
        vmin, vmax = extra_minmax[col]
        draw_gradient_legend(
            legend_x0 + i * legend_spacing, leg_y, vmin, vmax,
            lambda v, vmin=vmin, vmax=vmax: grey_color(v, vmin, vmax),
            f"{col} (light\u2192dark)",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a premium color chart.")
    parser.add_argument("file1", help="First ranking table file")
    parser.add_argument("file2", help="Second ranking table file")
    parser.add_argument("lang_data_csv", nargs="?", default="training_setup/langs/langs_chosen.csv")
    parser.add_argument("--cols", default="mean,var", help="Comma-separated extra columns to show")
    parser.add_argument("--spread-json", help="Path to byte position stats JSON file for standard spread")
    parser.add_argument("--spread-customenc-json", help="Path to byte position stats JSON file for custom encoding spread")
    args = parser.parse_args()

    prefix1 = get_prefix(args.file1)
    prefix2 = get_prefix(args.file2)
    assert prefix1 == prefix2, f"Ranking filenames must share the same prefix."

    def is_balanced_path(path):
        folder_and_file = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path)).lower()
        tokens = re.split(r'[^a-z0-9]+', folder_and_file)
        return "balanced" in tokens and "imbalanced" not in tokens

    is_balanced = is_balanced_path(args.file1) or is_balanced_path(args.file2)

    header_fields = peek_header(args.file1)
    col_specs = [c.strip() for c in args.cols.split(",") if c.strip()]
    if not col_specs:
        sys.exit(1)

    file_extra_cols = []
    spread_cols = []
    display_cols = []

    for c in col_specs:
        source, resolved_name = resolve_col_spec(c, header_fields)
        display_cols.append(resolved_name)
        if source == "file":
            file_extra_cols.append(resolved_name)
        elif source == "spread":
            spread_cols.append(resolved_name)

    spread_sources = {
        "spread_json": load_spread_json(args.spread_json),
        "spread_customenc_json": load_spread_json(args.spread_customenc_json),
    }

    table1 = parse_table(args.file1, file_extra_cols, spread_cols, spread_sources)
    table2 = parse_table(args.file2, file_extra_cols, spread_cols, spread_sources)
    training_data = parse_training_data(args.lang_data_csv, is_balanced=is_balanced)

    label1 = get_label(args.file1)
    label2 = get_label(args.file2)

    out_dir = os.path.dirname(os.path.abspath(args.file1))
    out_path = os.path.join(out_dir, f"{prefix1}_premium_color_chart.png")
    draw_chart(table1, table2, training_data, label1, label2, display_cols, out_path)


if __name__ == "__main__":
    main()