#!/usr/bin/env python3
"""
Premium color-chart generator.

Produces two separate charts from a pair of ranking files:
  * <prefix>_global_premium_chart.png
  * <prefix>_monotonicity_premium_chart.png

Each chart lays the languages out in two side-by-side halves and uses its own
premium color scale (green = that chart's min, red = that chart's max).

Can be run in two modes:
  1. Auto-resolution mode: pass a setup mode ('balanced', 'imbalanced',
     'balanced-custom') or a folder, and the global / monotonicity ranking
     files are found automatically.
  2. Manual mode: pass the global file and the monotonicity file directly
     (in that order).

Examples:
  python premium_color_chart.py balanced --global-cols mean,var --mono-cols mean,spread
  python premium_color_chart.py imbalanced --cols mean,var,log_bytes
  python premium_color_chart.py balanced --cols mean,var,byte_ratio   # ratio_vs_english from --lang-csv
"""

import argparse
import csv
import json
import math
import os
import re
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

TRAIN_DATA_ALIASES = {
    "train_bytes": "TrainBytes",
    "training_bytes": "TrainBytes",
    "bytes": "TrainBytes",
    "log_train_bytes": "LogTrainBytes",
    "log_bytes": "LogTrainBytes",
    "training_data": "LogTrainBytes",
}

# Columns pulled per-language from --lang-csv (matched by language code).
# Any other literal CSV header name also works.
CSV_ALIASES = {
    "byte_ratio": "ratio_vs_english", "ratio": "ratio_vs_english",
    "ratio_vs_english": "ratio_vs_english",
    "ratio_vs_english_imbalanced": "ratio_vs_english_imbalanced",
    "utf8_bytes": "utf8_bytes",
    "documents": "documents", "n_documents": "documents",
}

# Nicer header text for some columns (anything not listed is shown as-is).
DISPLAY_NAMES = {
    "ratio_vs_english": "ByteRatio",
    "ratio_vs_english_imbalanced": "ByteRatio (imb.)",
}

# Header names in the order results_to_txt_premiums.py writes them. The
# premiums file is FIXED-WIDTH, so rows are sliced at these header offsets
# instead of whitespace-split (a blank field would otherwise shift columns).
KNOWN_HEADERS = ["Language", "Premium", "PPS", "BPP", "EntropyMean",
                 "EntropyVar", "EntropySkew", "EntropyKurtosis",
                 "EntropyAutocorr1", "EntropyVolatility"]

SPREAD_FIELD_SOURCE = {
    "spread": ("spread_json", "spread"),
    "main_length_entropy_sum": ("spread_json", "main_length_entropy_sum"),
    "spread_customenc": ("spread_customenc_json", "spread"),
    "main_length_entropy_sum_customenc": ("spread_customenc_json", "main_length_entropy_sum"),
}

PRESET_DIRS = {
    "balanced": "results/txt_premiums/t_anchor/Balanced/step_0000007200",
    "imbalanced": "results/txt_premiums/t_anchor/Imbalanced/step_0000002600",
    "balanced-custom": "results/txt_premiums/t_anchor/Balanced_customenc/step_0000006400",
}

DEFAULT_COLS = "mean,var"


# --------------------------------------------------------------------------- #
# Input resolution & parsing
# --------------------------------------------------------------------------- #

def resolve_inputs(args_inputs):
    """Returns (global_file, mono_file)."""
    if len(args_inputs) == 1:
        target = args_inputs[0]
        folder = PRESET_DIRS.get(target.lower(), target)
        if not os.path.isdir(folder):
            raise ValueError(f"Target '{target}' is neither a known preset mode nor a valid directory.")

        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith("_premiums_sorted.txt")]

        def base(f):
            return os.path.basename(f).lower()

        mono_file = next((f for f in files if "mono" in base(f)), None)
        global_file = next(
            (f for f in files if f != mono_file and ("global" in base(f) or "entropy" in base(f))),
            None,
        )

        if not global_file or not mono_file:
            if len(files) == 2:
                global_file, mono_file = files[0], files[1]
            else:
                raise ValueError(f"Could not automatically locate global and monotonicity ranking files in {folder}")

        print(f"Resolved '{target}' ->\n  Global:       {global_file}\n  Monotonicity: {mono_file}")
        return global_file, mono_file

    elif len(args_inputs) == 2:
        return args_inputs[0], args_inputs[1]
    else:
        raise ValueError("Please provide either a single preset mode/directory or two file paths (global, monotonicity).")


def load_spread_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_col_spec(name, header_fields, csv_fields=()):
    key = name.strip().lower()
    if key in CSV_ALIASES:
        col = CSV_ALIASES[key]
        if col not in csv_fields:
            raise ValueError(f"Column '{name}' -> '{col}' is not in --lang-csv (columns: {list(csv_fields)}).")
        return "csv", col
    if key in TRAIN_DATA_ALIASES:
        return "train_data", TRAIN_DATA_ALIASES[key]
    if key in SPREAD_ALIASES:
        return "spread", SPREAD_ALIASES[key]
    resolved = COLUMN_ALIASES.get(key)
    if resolved is None:
        for h in header_fields:
            if h.lower() == key:
                resolved = h
                break
    if resolved is None or resolved not in header_fields:
        for h in csv_fields:
            if h.lower() == key:
                return "csv", h
        raise ValueError(
            f"Column '{name}' not found in header {header_fields} or --lang-csv "
            f"columns {list(csv_fields)}, and not a recognized spread or training data alias."
        )
    return "file", resolved


def resolve_col_list(cols_arg, header_fields, csv_fields):
    """Turns 'mean,var,byte_ratio' into (display_cols, {source: [cols]})."""
    specs = [c.strip() for c in cols_arg.split(",") if c.strip()]
    display_cols = []
    by_source = {"file": [], "spread": [], "train_data": [], "csv": []}
    for c in specs:
        source, resolved = resolve_col_spec(c, header_fields, csv_fields)
        if resolved in display_cols:
            continue
        display_cols.append(resolved)
        by_source[source].append(resolved)
    return display_cols, by_source


def _to_float(raw):
    try:
        return float(str(raw).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def read_premium_file(path):
    """Returns (present_headers, rows) with rows as {header: string}, in file
    order. Slices each row at the header's character offsets, so blank
    fields in the fixed-width file stay in their own column."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = next((i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#")), None)
    if header_idx is None:
        raise ValueError(f"No header line found in {path}")
    header_line = lines[header_idx].rstrip("\n")

    present, positions, search_from = [], [], 0
    for h in KNOWN_HEADERS:
        idx = header_line.find(h, search_from)
        if idx == -1:
            continue
        present.append(h)
        positions.append(idx)
        search_from = idx + len(h)
    if "Premium" not in present:
        raise ValueError(f"No 'Premium' column found in header of {path}")
    bounds = list(zip(positions, positions[1:] + [None]))

    rows = []
    for line in lines[header_idx + 1:]:
        if not line.strip():
            break
        if line.lstrip().startswith("#"):
            break
        line = line.rstrip("\n")
        rows.append({h: (line[a:b] if b is not None else line[a:]).strip()
                     for h, (a, b) in zip(present, bounds)})
    return present, rows


def peek_header(path):
    return read_premium_file(path)[0]


def parse_table(path, cols_by_source, spread_sources, training_data, langs_csv):
    """Returns an ordered dict {lang: (premium, extras)} in file order.
    Missing / blank extra values are stored as None (drawn as N/A)."""
    _, rows = read_premium_file(path)
    data = {}
    for row in rows:
        lang = row.get("Language", "")
        premium = _to_float(row.get("Premium"))
        if not lang or premium is None:
            continue

        extras = {c: _to_float(row.get(c)) for c in cols_by_source["file"]}

        for col in cols_by_source["spread"]:
            src_dict_name, field_name = SPREAD_FIELD_SOURCE[col]
            val = spread_sources.get(src_dict_name, {}).get(lang, {}).get(field_name)
            extras[col] = _to_float(val)

        for col in cols_by_source["train_data"]:
            bytes_val = training_data.get(lang)
            if bytes_val is None or bytes_val <= 0:
                extras[col] = None
            elif col == "TrainBytes":
                extras[col] = float(bytes_val)
            elif col == "LogTrainBytes":
                extras[col] = math.log10(bytes_val)

        for col in cols_by_source["csv"]:
            extras[col] = _to_float(langs_csv.get(lang, {}).get(col))

        data[lang] = (premium, extras)

    if not data:
        raise ValueError(f"No valid data rows parsed from {path}")
    return data


def _normalize_colname(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _find_column(fieldnames, keyword_sets):
    normed = {fn: _normalize_colname(fn) for fn in fieldnames}
    for keywords in keyword_sets:
        for fn, norm in normed.items():
            if all(kw in norm for kw in keywords):
                return fn
    return None


def load_langs_csv(path):
    """Returns (fieldnames, {language_code: row_dict})."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError(f"No data rows parsed from {path}")

    code_col = "language_code" if "language_code" in fieldnames else None
    if code_col is None:
        for fn in fieldnames:
            sample_vals = [r[fn].strip() for r in rows[:5] if r.get(fn)]
            if sample_vals and all(CODE_PATTERN.match(v) for v in sample_vals):
                code_col = fn
                break
    name_col = None
    if code_col is None:
        name_col = _find_column(fieldnames, [("language",), ("lang", "name")])

    by_code = {}
    for row in rows:
        if code_col is not None:
            code = (row.get(code_col) or "").strip()
        else:
            code = LANG_NAME_TO_CODE.get((row.get(name_col) or "").strip().lower())
        if code:
            by_code[code] = row
    return fieldnames, by_code


def parse_training_data(fieldnames, langs_csv, is_balanced=False):
    """Returns {language_code: allocation_bytes} for the run's setup
    (used by the train_bytes / log_bytes columns)."""
    setup = "balanced" if is_balanced else "imbalanced"
    bytes_col = _find_column(fieldnames, [
        (setup, "allocation", "bytes"),
        (setup, "bytes"),
        ("allocation", "bytes"),
        ("bytes",),
    ])
    if bytes_col is None:
        raise ValueError("Could not find a matching bytes column in --lang-csv.")
    data = {}
    for code, row in langs_csv.items():
        v = _to_float(row.get(bytes_col))
        if v is not None:
            data[code] = v
    return data


def get_prefix(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_")[0]


# --------------------------------------------------------------------------- #
# Colors & formatting
# --------------------------------------------------------------------------- #

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


def format_cell_value(col, val):
    if val is None:
        return "N/A"
    if col == "TrainBytes":
        if val >= 1e9:
            return f"{val/1e9:.1f}G"
        elif val >= 1e6:
            return f"{val/1e6:.1f}M"
        elif val >= 1e3:
            return f"{val/1e3:.1f}K"
        return f"{int(val)}"
    elif col == "LogTrainBytes":
        return f"10^{val:.1f}"
    return f"{val:.2f}"


def format_legend_value(col, val):
    if col == "LogTrainBytes":
        return f"10^{val:.1f}"
    if col == "TrainBytes":
        return format_cell_value(col, val)
    return f"{val:.2f}"


def text_color_for(bg):
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return "black" if lum > 0.55 else "white"


def wrap_col_header(col, width=9):
    """'EntropyMean' -> 'Entropy\\nMean' so neighbouring headers don't collide."""
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z][a-z])", " ", col).replace("_", " ")
    return textwrap.fill(words, width)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

# Font sizes (fixed -- only column widths adapt to the number of columns)
FS_TITLE = 13
FS_LANG_HEADER = 10
FS_COL_HEADER = 9
FS_LANG = 9.5
FS_CELL = 9.2
FS_LEGEND_LABEL = 8.5
FS_LEGEND_TICK = 8

DEFAULT_FIG_WIDTH = 14.0   # inches; fixed total width of every chart


def text_width_in(text, fontsize, fontweight="normal"):
    """Rendered width of a (possibly multi-line) string, in inches."""
    fig = plt.figure()
    t = fig.text(0, 0, text, fontsize=fontsize, fontweight=fontweight)
    fig.canvas.draw()
    w = t.get_window_extent().width / fig.dpi
    plt.close(fig)
    return w


def draw_chart(table, title, display_cols, out_path, fig_width=DEFAULT_FIG_WIDTH):
    langs = list(table.keys())  # file order (already sorted by premium)
    n = len(langs)
    half = (n + 1) // 2
    halves = [langs[:half], langs[half:]]

    # --- Scales: each chart uses its own premium range ---------------------
    prem_vals = [table[l][0] for l in langs]
    p_min, p_max = min(prem_vals), max(prem_vals)

    extra_minmax = {}
    for col in display_cols:
        vals = [table[l][1][col] for l in langs if table[l][1][col] is not None]
        extra_minmax[col] = (min(vals), max(vals)) if vals else (0.0, 1.0)

    # --- Labels -------------------------------------------------------------
    labels = {l: CODE_TO_LANG_NAME.get(l, l) for l in langs}

    # --- Geometry (x in inches, y in row units) -----------------------------
    # Horizontal layout: the figure has a fixed width. In each half, the
    # language column is as wide as the longest language name; the premium
    # column and the extra columns share the remaining width equally.
    margin = 0.2                    # left/right margin (inches)
    gap = 0.6                       # space between the two halves (inches)
    lang_pad = 0.25                 # space between language names and first box
    col_pad = 0.15                  # horizontal space between neighbouring boxes
    min_col_w = 0.7                 # below this, cell values no longer fit

    lang_w = max(text_width_in(s, FS_LANG, "medium") for s in list(labels.values()) + ["Language"]) + lang_pad
    n_extra = len(display_cols)
    n_cols = 1 + n_extra            # premium + extras

    block_w = (fig_width - 2 * margin - gap) / 2
    col_w = (block_w - lang_w) / n_cols
    if col_w < min_col_w:           # too many columns for this width: grow the figure
        col_w = min_col_w
        block_w = lang_w + n_cols * col_w
        fig_width = 2 * margin + 2 * block_w + gap
        print(f"note: widened figure to {fig_width:.1f} in to fit {n_cols} columns")
    sq = col_w - col_pad            # box width (inches)

    # Column headers wrap to fit their column
    chars_per_line = max(5, int(col_w / (0.62 * FS_COL_HEADER / 72)))
    headers = {c: wrap_col_header(DISPLAY_NAMES.get(c, c), chars_per_line) for c in display_cols}
    header_lines = max([h.count("\n") + 1 for h in headers.values()] + [1])

    # Vertical layout
    box_h_in = 0.2                 # box height (inches)
    row_gap_in = 0.08               # empty space between rows of boxes (inches)
    row_h = box_h_in + row_gap_in   # inches per row
    box_h = box_h_in / row_h        # box height as a fraction of the row (y is in row units)
    # y is in row units, so header/legend offsets are scaled by u to keep them
    # a fixed physical size no matter how tight the rows are.
    u = 0.55 / row_h

    table_left = margin
    table_right = fig_width - margin - col_pad / 2   # right edge of the last box

    # Legend row: one gradient per column (premium + extras), stretched so
    # together they span the full table width from the left edge to the right.
    n_legends = n_cols
    legend_gap = 0.6
    grad_w = (table_right - table_left - (n_legends - 1) * legend_gap) / n_legends
    legend_spacing = grad_w + legend_gap
    fig_w = fig_width

    header_line_h = 0.35 * u
    extra_top = header_line_h * (header_lines - 1)
    top_units = 2.0 * u + extra_top      # title + column headers
    bottom_units = 2.7 * u               # legend row
    fig_h = (half + top_units + bottom_units) * row_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(-bottom_units, half + top_units)
    ax.axis("off")

    def draw_square(x, y, color, text):
        if color is None:
            rect = patches.FancyBboxPatch(
                (x, y), sq, box_h,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=0.6, linestyle="--", edgecolor="#cccccc", facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(x + sq / 2, y + box_h / 2, "N/A", ha="center", va="center",
                    fontsize=FS_CELL, color="#aaaaaa", fontweight="bold")
        else:
            rect = patches.FancyBboxPatch(
                (x, y), sq, box_h,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=0.6, edgecolor="#333333", facecolor=color,
            )
            ax.add_patch(rect)
            ax.text(x + sq / 2, y + box_h / 2, text, ha="center", va="center",
                    fontsize=FS_CELL, color=text_color_for(color), fontweight="bold")

    # --- Title --------------------------------------------------------------
    ytop = half + 0.6 * u
    ax.text(fig_w / 2, ytop + extra_top + 0.75 * u, title, fontsize=FS_TITLE,
            fontweight="bold", ha="center", va="bottom", color="#333333")

    # --- Two blocks ---------------------------------------------------------
    for b, block_langs in enumerate(halves):
        if not block_langs:
            continue
        x0 = margin + b * (block_w + gap)
        # left edge of each box; every data column is col_w wide, box centered in it
        x_prem = x0 + lang_w + col_pad / 2
        x_extra = {c: x_prem + (i + 1) * col_w for i, c in enumerate(display_cols)}

        ax.text(x0, ytop, "Language", fontsize=FS_LANG_HEADER, fontweight="bold", va="bottom")
        ax.text(x_prem + sq / 2, ytop, "Premium", fontsize=FS_COL_HEADER, fontweight="bold", ha="center", va="bottom")
        for col in display_cols:
            ax.text(x_extra[col] + sq / 2, ytop, headers[col], fontsize=FS_COL_HEADER, fontweight="bold",
                    ha="center", va="bottom", multialignment="center", linespacing=1.1)

        for r, lang in enumerate(block_langs):
            y = half - r - 0.3 - box_h / 2   # box vertically centered in its row
            premium, extras = table[lang]
            ax.text(x0, y + box_h / 2, labels[lang], fontsize=FS_LANG, va="center", fontweight="medium")
            draw_square(x_prem, y, premium_color(premium, p_min, p_max), f"{premium:.2f}")
            for col in display_cols:
                val = extras[col]
                vmin, vmax = extra_minmax[col]
                color = grey_color(val, vmin, vmax) if val is not None else None
                draw_square(x_extra[col], y, color, format_cell_value(col, val))

    # Divider between the two halves
    if halves[1]:
        x_div = margin + block_w + gap / 2
        ax.plot([x_div, x_div], [0.2, ytop + 0.3 * u], color="#dddddd", lw=1)

    # --- Legends ------------------------------------------------------------
    n_steps = 200
    leg_y = -0.8 * u

    def draw_gradient(x0, color_at_t):
        for k in range(n_steps):
            t0 = k / n_steps
            ax.add_patch(patches.Rectangle(
                (x0 + t0 * grad_w, leg_y), grad_w / n_steps + 0.001, 0.25 * u,
                color=color_at_t(t0), linewidth=0,
            ))

    # Legend row spans the full table width (see geometry above)
    leg_x0 = table_left

    # Premium legend (this chart's own scale)
    draw_gradient(leg_x0, green_red_gradient)
    ax.text(leg_x0 + grad_w / 2, leg_y + 0.35 * u, "Premium",
            fontsize=FS_LEGEND_LABEL, ha="center", va="bottom")
    ax.text(leg_x0, leg_y - 0.3 * u, f"{p_min:.2f}", fontsize=FS_LEGEND_TICK, ha="left", fontweight="semibold")
    ax.text(leg_x0 + grad_w, leg_y - 0.3 * u, f"{p_max:.2f}", fontsize=FS_LEGEND_TICK, ha="right", fontweight="semibold")
    ax.text(leg_x0 + grad_w / 2, leg_y - 0.3 * u, f"(\u0394 {p_max - p_min:.2f})", fontsize=FS_LEGEND_TICK - 0.5,
            ha="center", color="#555555", fontstyle="italic")

    # Extra-column legends
    for i, col in enumerate(display_cols):
        x0 = leg_x0 + (i + 1) * legend_spacing
        vmin, vmax = extra_minmax[col]
        draw_gradient(x0, lambda t, vmin=vmin, vmax=vmax: grey_color(vmin + t * (vmax - vmin), vmin, vmax))
        ax.text(x0 + grad_w / 2, leg_y + 0.35 * u, DISPLAY_NAMES.get(col, col), fontsize=FS_LEGEND_LABEL,
                ha="center", va="bottom")
        ax.text(x0, leg_y - 0.3 * u, format_legend_value(col, vmin), fontsize=FS_LEGEND_TICK, ha="left")
        ax.text(x0 + grad_w, leg_y - 0.3 * u, format_legend_value(col, vmax), fontsize=FS_LEGEND_TICK, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Generate separate global and monotonicity premium color charts."
    )
    parser.add_argument("inputs", nargs="+",
                        help="Preset mode ('balanced', 'imbalanced', 'balanced-custom'), a folder, "
                             "or 2 file paths: <global_file> <mono_file>")
    parser.add_argument("--lang-csv", default="training_setup/langs/langs_chosen.csv",
                        help="Path to training data CSV")
    parser.add_argument("--cols", default=DEFAULT_COLS,
                        help="Default extra columns for BOTH charts (e.g. mean,var,spread,log_bytes,byte_ratio). "
                             "Any --lang-csv column name also works.")
    parser.add_argument("--global-cols", default=None,
                        help="Extra columns for the global chart (overrides --cols)")
    parser.add_argument("--mono-cols", default=None,
                        help="Extra columns for the monotonicity chart (overrides --cols)")
    parser.add_argument("--width", type=float, default=DEFAULT_FIG_WIDTH,
                        help=f"Total chart width in inches (default {DEFAULT_FIG_WIDTH}). "
                             "Fonts stay fixed; columns share the space left after the language column.")
    parser.add_argument("--spread-json", help="Path to byte position stats JSON file for standard spread")
    parser.add_argument("--spread-customenc-json", help="Path to byte position stats JSON file for custom encoding spread")
    args = parser.parse_args()

    global_file, mono_file = resolve_inputs(args.inputs)

    prefix = get_prefix(global_file)
    assert prefix == get_prefix(mono_file), "Ranking filenames must share the same prefix."

    # Balanced vs. imbalanced decides which bytes column train_bytes / log_bytes read
    def is_balanced_path(path):
        folder_and_file = os.path.join(os.path.basename(os.path.dirname(path)),
                                       os.path.basename(path)).lower()
        tokens = re.split(r"[^a-z0-9]+", folder_and_file)
        return "balanced" in tokens and "imbalanced" not in tokens

    is_balanced = is_balanced_path(global_file) or is_balanced_path(mono_file)

    spread_sources = {
        "spread_json": load_spread_json(args.spread_json),
        "spread_customenc_json": load_spread_json(args.spread_customenc_json),
    }
    csv_fields, langs_csv = load_langs_csv(args.lang_csv)
    training_data = parse_training_data(csv_fields, langs_csv, is_balanced=is_balanced)

    out_dir = os.path.dirname(os.path.abspath(global_file))
    charts = [
        ("global", global_file, args.global_cols or args.cols),
        ("monotonicity", mono_file, args.mono_cols or args.cols),
    ]

    for name, path, cols_arg in charts:
        display_cols, cols_by_source = resolve_col_list(cols_arg, peek_header(path), csv_fields)
        table = parse_table(path, cols_by_source, spread_sources, training_data, langs_csv)
        m = re.search(r"_t_([0-9.]+)_", os.path.basename(path))
        title = f"{prefix} \u2014 {name.capitalize()} premium" + (f" (t = {m.group(1)})" if m else "")
        out_path = os.path.join(out_dir, f"{prefix}_{name}_premium_chart.png")
        draw_chart(table, title, display_cols, out_path, fig_width=args.width)


if __name__ == "__main__":
    main()