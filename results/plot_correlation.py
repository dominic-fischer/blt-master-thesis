#!/usr/bin/env python3
"""
plot_correlation.py -- Correlation (plain or partial) between any two
columns of a premiums_sorted.txt file produced by results_to_txt_premiums.py.

COLUMN SPECS (--cols, --control-for)
    Each column can be given as:
      - a short alias: premium, pps, bpp, mean, var/variance, skew,
        kurtosis/kurt, autocorr/autocorrelation, volatility/vol,
        training_data/training_data_balanced (RANK by training-data
        size, 1=most -- not raw bytes, see COLUMN_ALIASES), density,
        codepoints, cp_per_baseline, spread, spread_customenc,
        word_length, n_words, n_char, char_ratio, word_ratio, dense
      - the exact header text from the file (case-insensitive): Premium,
        PPS, BPP, EntropyMean, EntropyVar, EntropySkew, EntropyKurtosis,
        EntropyAutocorr1, EntropyVolatility -- or any merged-in column
        (e.g. Dense_Script, Premium_global from --extra-premium)
      - a RATIO of two such names, e.g. mean/variance -- computed
        elementwise from the two underlying columns

    --cols takes exactly two comma-separated specs: X,Y.

LANGUAGE FILTERS (--script-type, --bytes-per-char)
    Restrict the plot / correlation to a subset of the languages, using
    the 'Script_Type' and 'Approx_Bytes_Per_Char' columns of --langs-csv
    (training_setup/langs/langs_chosen.csv). Both take one or more
    comma-separated values; if both are given, a language must match
    BOTH (AND across the two flags, OR within one flag).

      --script-type     Alphabetic, Abjad, Abugida, Syllabary,
                        Logosyllabary, Logographic   (case-insensitive)
      --bytes-per-char  1, 1-2, 2, 2-3, 3            (exact CSV values;
                        "1-2" = Vietnamese, "2-3" = Georgian)

    Ranks (training_data / training_data_balanced) are always computed
    over ALL languages BEFORE filtering, so a language keeps the same
    rank it has in the unfiltered plots. Log10 columns are per-language and
    therefore unaffected by filtering. The active filter is appended to
    the output filename and the plot title.

COLOUR CODING (--color-by)
    Colours each point by a categorical property of its language, read
    from --langs-csv. Default is 'bytes-per-char' (falls back to 'none'
    with a warning if --langs-csv can't be loaded).

      --color-by bytes-per-char  Approx_Bytes_Per_Char (1, 1-2, 2, 2-3, 3) (default)
      --color-by script-type     Script_Type (Alphabetic, Abjad, ...)
      --color-by none            no colour coding

    Colours are FIXED per category (see CATEGORY_PALETTES), so e.g.
    Abjad is the same colour in every figure you make, filtered or not.
    Bytes-per-char uses blue (1 byte), green (2 bytes) and yellow-beige
    (3 bytes); range values like "1-2" (Vietnamese) or "2-3"
    (Georgian) are drawn as split markers. The regression line and r/p
    values are unchanged -- colour is purely visual. A '_color-<choice>'
    tag is appended to the auto-derived output filename.

GROUPS (--group-by) AND RESIDUAL BREAKDOWN (--breakdown)
    --group-by decides how languages are split into two groups for the
    residual breakdown and for --group-fit:
      bytes  (default) single-byte = Approx_Bytes_Per_Char "1" or "1-2",
                       multi-byte  = "2", "2-3", "3"
      dense            non-dense / dense, from the Dense_Script column of
                       --langs-csv (1 = dense: abjads, syllabaries,
                       logosyllabaries, logographic scripts)
      none             no groups (no breakdown, no --group-fit)

    For plain (non-partial) plots, the residuals of the single joint
    regression line are broken down by group:
      - each group's share of the unexplained variance (sum of squared
        residuals), split into
          offset  = n_g * mean_residual_g^2   (group systematically
                    above/below the line)
          scatter = sum (residual - mean_residual_g)^2
      - per-group mean residual, RMSE and largest deviation
    Printed by default; --breakdown legend also summarises it inside the
    plot legend, --breakdown off disables it. Skipped automatically if
    either group has fewer than 3 languages.

    Out-of-sample check (--oos): a line fitted to the SECOND group only
    (multi-byte / dense), and how well it predicts the first. Only
    useful when there is NO group offset. By default ('auto') it is
    printed only when the group offset is below OOS_OFFSET_THRESHOLD of
    the unexplained variance; 'always' / 'never' override.

COMMON SLOPE + GROUP OFFSET (--group-fit)
    Fits y = b*x + a_g: ONE slope shared by both groups, a separate
    intercept per group (i.e. the joint line plus a group offset). Draws
    one parallel line per group, and reports:
      - the common slope, each group's intercept and the offset between
        them,
      - R^2 of this model next to R^2 of the single joint line,
      - the partial correlation of x and y controlling for the group
        (= correlation within groups; one extra degree of freedom),
      - the slopes each group would get if fitted SEPARATELY (to judge
        whether a common slope is reasonable),
      - how the remaining unexplained variance splits between groups.
    Plain plots only (ignored with --control-for). Adds '_groupfit-<by>'
    to the output filename.

OUTLIERS (--ignore-outliers / --ignore_outliers)
    Comma-separated languages that stay IN the plot but are left OUT of
    every statistic: the regression line, r / r^2 / p, the residual
    breakdown, --group-fit, and (with --control-for) the control
    regressions used to form the residuals. Each language can be given
    as its code (amh_Ethi), the code prefix (amh), its full name
    (Mandarin Chinese) or any single word of the name (chinese) --
    case-insensitive.

    Ignored points keep their --color-by colour on the left half and
    are filled black on the right half; their labels are grey italic,
    and the legend gets an 'Excluded from fit' entry. Ranks are still
    computed over all languages. Adds '_ignore-<codes>' to the output
    filename and lists the excluded languages in the subtitle.

USAGE
    python3 plot_correlation.py balanced --threshold global --cols mean,premium
    python3 plot_correlation.py balanced-custom --threshold global --cols word_ratio,premium \\
        --color-by script-type --group-by dense --group-fit --breakdown legend
    python3 plot_correlation.py balanced --threshold global --cols mean,premium --control-for variance
    python3 plot_correlation.py balanced --char-level --threshold global --cols mean,premium \\
        --color-by script-type --ignore-outliers amharic,japanese,chinese,korean
    python3 plot_correlation.py balanced-custom --threshold mono \\
        --extra-premium global=balanced-custom:global \\
        --cols spread_customenc,premium --control-for premium_global

PARSING THE INPUT FILE
    That file format is FIXED-WIDTH. This script locates each header's
    start column IN THE HEADER LINE ITSELF and slices every data row at
    those exact character offsets -- correct regardless of where blank
    fields fall.

Requires: matplotlib, numpy
    pip install matplotlib numpy
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CODE_TO_LANG_NAME = {
    "eng_Latn": "English", "cmn_Hans": "Mandarin Chinese", "deu_Latn": "German",
    "jpn_Jpan": "Japanese", "spa_Latn": "Spanish", "fra_Latn": "French",
    "ita_Latn": "Italian", "vie_Latn": "Vietnamese", "arb_Arab": "Arabic",
    "tha_Thai": "Thai", "kor_Hang": "Korean", "ron_Latn": "Romanian",
    "fin_Latn": "Finnish", "heb_Hebr": "Hebrew", "tam_Taml": "Tamil",
    "hrv_Latn": "Croatian", "srp_Cyrl": "Serbian", "kat_Geor": "Georgian",
    "amh_Ethi": "Amharic", "nya_Latn": "Chichewa",
}

# Header names in the exact left-to-right order results_to_txt_premiums.py
# writes them in (only whichever subset are actually present in a given
# file are matched -- see parse_premium_txt).
KNOWN_HEADERS = ["Language", "Premium", "PPS", "BPP", "EntropyMean",
                 "EntropyVar", "EntropySkew", "EntropyKurtosis",
                 "EntropyAutocorr1", "EntropyVolatility"]

COLUMN_ALIASES = {
    "language": "Language", "lang": "Language",
    "premium": "Premium",
    "pps": "PPS",
    "bpp": "BPP",
    "mean": "EntropyMean", "entropymean": "EntropyMean",
    "var": "EntropyVar", "variance": "EntropyVar", "entropyvar": "EntropyVar",
    "skew": "EntropySkew", "skewness": "EntropySkew", "entropyskew": "EntropySkew",
    "kurtosis": "EntropyKurtosis", "kurt": "EntropyKurtosis", "entropykurtosis": "EntropyKurtosis",
    "autocorr": "EntropyAutocorr1", "autocorrelation": "EntropyAutocorr1",
    "autocorr1": "EntropyAutocorr1", "entropyautocorr1": "EntropyAutocorr1",
    "volatility": "EntropyVolatility", "vol": "EntropyVolatility", "entropyvolatility": "EntropyVolatility",

    # --- from --langs-csv (default: training_setup/langs/langs_chosen.csv) ---
    # "training_data"/"training_data_imbalanced"/"training_data_balanced"
    # resolve to RANK columns (computed by add_rank_columns, not raw byte
    # counts). Rank 1 = most training data. Use the literal CSV column
    # names if you want the raw byte values instead.
    "training_data": "imbalanced_allocation_bytes_rank",
    "training_data_rank": "imbalanced_allocation_bytes_rank",
    "training_data_imbalanced": "imbalanced_allocation_bytes_rank",
    "imbalanced_rank": "imbalanced_allocation_bytes_rank",
    "training_data_log": "imbalanced_allocation_bytes_log10",
    "training_data_log10": "imbalanced_allocation_bytes_log10",
    "imbalanced_log": "imbalanced_allocation_bytes_log10",
    "imbalanced_bytes": "imbalanced_allocation_bytes",
    "imbalanced_allocation_bytes": "imbalanced_allocation_bytes",
    "training_data_balanced": "balanced_allocation_bytes_rank",
    "balanced_rank": "balanced_allocation_bytes_rank",
    "training_data_balanced_log": "balanced_allocation_bytes_log10",
    "balanced_log": "balanced_allocation_bytes_log10",
    "balanced_bytes": "balanced_allocation_bytes",
    "balanced_allocation_bytes": "balanced_allocation_bytes",
    "documents": "documents", "n_documents": "documents",
    "utf8_bytes": "utf8_bytes",
    "ratio_vs_english": "ratio_vs_english",
    "ratio_vs_english_imbalanced": "ratio_vs_english_imbalanced",
    "bytes_per_char": "Approx_Bytes_Per_Char", "approx_bytes_per_char": "Approx_Bytes_Per_Char",
    "dense": "Dense_Script", "dense_script": "Dense_Script",

    # --- from --density-json (default: char_density.json) ---
    "density": "density_index", "density_index": "density_index",
    "codepoints": "n_codepoints", "n_codepoints": "n_codepoints",
    "cp_per_baseline": "codepoints_per_baseline_codepoint",
    "codepoints_per_baseline_codepoint": "codepoints_per_baseline_codepoint",
    "customenc_bytes": "n_bytes_total",

    # --- from --spread-json / --spread-customenc-json ---
    "spread": "spread", "entropy_spread": "spread",
    "spread_customenc": "spread_customenc", "custom_spread": "spread_customenc",
    "spread_custom_encoding": "spread_customenc",
    "spread_sum": "main_length_entropy_sum", "entropy_sum": "main_length_entropy_sum",
    "spread_customenc_sum": "main_length_entropy_sum_customenc",

    # --- from --wordlen-json (default: word_length_stats.json) ---
    "word_length": "word_length", "avg_word_length": "word_length",
    "n_words": "n_words", "words": "n_words",
    "n_char": "n_char", "total_chars": "n_char",
    "char_ratio": "char_ratio",
    "word_ratio": "word_ratio",
}

# Which literal column names get pulled in from each optional external
# source, if that source is loaded and has a matching language row.
LANGS_CSV_COLUMNS = [
    "ratio_vs_english", "documents", "utf8_bytes", "balanced_allocation_bytes",
    "ratio_vs_english_imbalanced", "imbalanced_allocation_bytes", "Approx_Bytes_Per_Char",
    "Dense_Script",
]
DENSITY_JSON_COLUMNS = [
    "n_bytes_total", "n_codepoints", "codepoints_per_baseline_codepoint", "density_index",
]
SPREAD_JSON_FIELDS = {
    "spread": "spread",
    "main_length_entropy_sum": "main_length_entropy_sum",
}
SPREAD_CUSTOMENC_JSON_FIELDS = {
    "spread": "spread_customenc",
    "main_length_entropy_sum": "main_length_entropy_sum_customenc",
}
WORDLEN_COLUMNS = ["word_length", "n_words", "n_char", "char_ratio", "word_ratio"]

# --- COLOUR CODING (--color-by) ---
COLOR_BY_OPTIONS = {
    "none": None,
    "script-type": ("Script_Type", "Script type"),
    "bytes-per-char": ("Approx_Bytes_Per_Char", "Bytes per char"),
}
CATEGORY_PALETTES = {
    "Script_Type": {
        "Alphabetic":    "#4C72B0",  # blue
        "Abjad":         "#DD8452",  # orange
        "Abugida":       "#55A868",  # green
        "Syllabary":     "#C44E52",  # red
        "Logosyllabary": "#8172B3",  # purple
        "Logographic":   "#937860",  # brown
    },
    "Approx_Bytes_Per_Char": {
        "1": "#4C72B0",
        "2": "#55A868",
        "3": "#CCB974",
    },
}
UNKNOWN_CATEGORY_COLOR = "#BBBBBB"
DEFAULT_POINT_COLOR = "#4C72B0"
# --ignore-outliers: right half of an excluded point is filled with this,
# and its label is drawn in IGNORED_LABEL_COLOR (italic).
IGNORED_FILL_COLOR = "#111111"
IGNORED_LABEL_COLOR = "#777777"

# --- GROUPS (--group-by) ---
# Bytes-per-char values counted as single-byte (dominant character
# length 1 -- matches spread being undefined for these languages).
SINGLE_BYTE_VALUES = {"1", "1-2"}
# Script_Type values (case-insensitive) counted as DENSE: one character
# carries more than one alphabetic letter's worth of information.
DENSE_SCRIPT_TYPES = {"abjad", "syllabary", "logosyllabary", "logographic"}
# group_by -> (label of first group, label of second group, footnote)
GROUP_DEFS = {
    "bytes": ("single-byte", "multi-byte",
              "single-byte = Approx_Bytes_Per_Char 1 or 1-2; multi-byte = 2, 2-3, 3"),
    "dense": ("non-dense", "dense",
            "dense = Script_Type Abjad, Syllabary, Logosyllabary or Logographic"),
}
# Colours of the per-group lines drawn by --group-fit (first, second group).
GROUP_LINE_COLORS = ("#2F4B7C", "#B5473A")

# --oos auto: run the out-of-sample check only if the group offset is
# below this share of the unexplained variance.
OOS_OFFSET_THRESHOLD = 0.10

# Readable names for plot titles and axis labels.
DISPLAY_NAMES = {
    "Premium": "Premium",
    "EntropyMean": "Entropy mean",
    "EntropyVar": "Entropy variance",
    "EntropySkew": "Entropy skewness",
    "EntropyKurtosis": "Entropy kurtosis",
    "EntropyAutocorr1": "Entropy autocorrelation (lag 1)",
    "EntropyVolatility": "Entropy volatility",
    "ratio_vs_english": "Byte ratio (rel. to English)",
    "ratio_vs_english_imbalanced": "Byte ratio (rel. to English, imbalanced)",
    "imbalanced_allocation_bytes": "Training data (bytes)",
    "imbalanced_allocation_bytes_log10": "Training data (log$_{10}$ bytes)",
    "imbalanced_allocation_bytes_rank": "Training data (rank)",
    "balanced_allocation_bytes": "Training data (bytes)",
    "balanced_allocation_bytes_log10": "Training data (log$_{10}$ bytes)",
    "balanced_allocation_bytes_rank": "Training data (rank)",
    "documents": "Documents",
    "utf8_bytes": "UTF-8 bytes",
    "spread": "Spread",
    "spread_customenc": "Spread (custom encoding)",
    "main_length_entropy_sum": "Identity entropy per character (bits)",
    "main_length_entropy_sum_customenc": "Identity entropy per character (bits, custom enc.)",
    "density_index": "Density index",
    "n_codepoints": "Codepoints",
    "codepoints_per_baseline_codepoint": "Codepoints per English codepoint",
    "n_bytes_total": "Bytes (custom encoding)",
    "Dense_Script": "Dense script (0/1)",
    "word_length": "Average word length (characters)",
    "n_words": "Words",
    "n_char": "Characters",
    "char_ratio": "Character ratio (rel. to English)",
    "word_ratio": "Word ratio (rel. to English)",
}

# Font sizes (points).
FS_TITLE = 16
FS_SUBTITLE = 12
FS_AXIS_LABEL = 14
FS_TICKS = 12
FS_POINT_LABEL = 11
FS_LEGEND = 11
FS_LEGEND_TITLE = 11.5
POINT_SIZE = 140  # scatter marker area

# Legend position inside the axes (matplotlib loc string); --legend-loc.
LEGEND_LOC = "lower left"


def pretty_label(label):
    """Readable version of an internal column label for titles/axes.
    Handles plain names (via DISPLAY_NAMES), log10(...) wrappers, A/B
    ratios, and the ' residual' suffix used by the partial plots."""
    if label.endswith(" residual"):
        return f"{pretty_label(label[:-len(' residual')])} (residual)"
    m = re.match(r"^log10\((.+)\)$", label)
    if m:
        return f"log$_{{10}}$({pretty_label(m.group(1))})"
    if "/" in label:
        num, den = label.split("/", 1)
        return f"{pretty_label(num)} / {pretty_label(den)}"
    return DISPLAY_NAMES.get(label, label)


def describe_filter(script_types, bytes_per_char):
    """Readable description of the active language filter, '' if none."""
    parts = []
    if bytes_per_char:
        b = set(bytes_per_char)
        if b == SINGLE_BYTE_VALUES or b == {"1"}:
            parts.append("single-byte languages" if b == {"1"} else "single-byte languages (incl. 1-2)")
        elif b == {"2", "2-3", "3"}:
            parts.append("multi-byte languages")
        else:
            parts.append(f"{', '.join(sorted(b))} bytes/char")
    if script_types:
        parts.append(", ".join(s.capitalize() for s in script_types) + " scripts")
    return ", ".join(parts)


def describe_run(stem, char_level=False):
    """Readable run description from a premiums file stem."""
    m = re.match(r"^(?P<run>.+?)_raw_(?P<case>entropy|monotonicity)_t_(?P<t>[\d.]+)", stem)
    if not m:
        return stem
    strategy = "global threshold" if m.group("case") == "entropy" else "monotonicity threshold"
    desc = f"{m.group('run')}, {strategy} (t = {m.group('t')})"
    if char_level:
        desc += ", char-level"
    return desc


RUN_NAME_LOOKUP = {
    "balanced": "Balanced",
    "imbalanced": "Imbalanced",
    "balanced-custom": "Balanced-Custom",
    "balanced_custom": "Balanced-Custom",
    "balancedcustom": "Balanced-Custom",
}
RUN_INFO = {
    "Balanced": {
        "step": "0000007200",
        "byte": {"global": "1.9458", "mono": "0.6664"},
        "char": {"global": "1.9448", "mono": "0.6646"},
    },
    "Imbalanced": {
        "step": "0000002600",
        "byte": {"global": "1.7510", "mono": "0.5531"},
        "char": {"global": "1.7488", "mono": "0.5552"},
    },
    "Balanced-Custom": {
        "step": "0000006400",
        "byte": {"global": "2.0176", "mono": "2.0371"},
        "char": {"global": "2.0630", "mono": "0.6877"},
    },
}


def resolve_premium_path(spec, char_level, threshold, base_dir="results/txt_premiums"):
    """Full premiums_sorted.txt path for a shorthand run name, or None if
    spec isn't a recognized shorthand (caller then uses it as a path)."""
    key = RUN_NAME_LOOKUP.get(spec.strip().lower())
    if key is None:
        return None
    info = RUN_INFO[key]
    granularity = "char" if char_level else "byte"
    t_value = info[granularity][threshold]
    case_name = "raw_entropy" if threshold == "global" else "raw_monotonicity"
    parts = [base_dir]
    if char_level:
        parts.append("char_level")
    parts.append("t_anchor")
    parts.append(key)
    parts.append(f"step_{info['step']}")
    parts.append(f"{key}_{case_name}_t_{t_value}_premiums_sorted.txt")
    return os.path.join(*parts)


def resolve_column_name(name, available_columns):
    """Maps a user-given column name (alias or literal header text,
    case-insensitive either way) to the actual key present in the merged
    row data. Raises a clear error listing what IS available otherwise."""
    key = name.strip().lower()
    resolved = COLUMN_ALIASES.get(key)
    if resolved is None:
        for h in available_columns:
            if h.lower() == key:
                resolved = h
                break
    if resolved is None or resolved not in available_columns:
        raise ValueError(
            f"Column '{name}' not found or not present in the available data. "
            f"Available columns: {sorted(available_columns)}. Known aliases: "
            f"{sorted(set(COLUMN_ALIASES.keys()))}. If you expected a column from "
            f"--langs-csv, --density-json, --spread-json, --spread-customenc-json, "
            f"--wordlen-json or --extra-premium, check those files were found and "
            f"contain a matching language code."
        )
    return resolved


def load_langs_csv(path):
    """Returns {language_code: {csv_column: raw_string_value}}."""
    result = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("language_code") or "").strip()
            if code:
                result[code] = row
    return result


def parse_filter_values(raw):
    """'1,1-2' -> ['1', '1-2']; None if raw is None/empty."""
    if raw is None:
        return None
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    return vals or None


def select_languages(langs_csv_data, script_types=None, bytes_per_char=None):
    """Language codes matching the --script-type / --bytes-per-char
    filters (AND across flags, OR within one). Raises ValueError for
    values that don't occur in the CSV."""
    known_types = sorted({(r.get("Script_Type") or "").strip()
                          for r in langs_csv_data.values()} - {""})
    known_bytes = sorted({(r.get("Approx_Bytes_Per_Char") or "").strip()
                          for r in langs_csv_data.values()} - {""})

    wanted_types = {v.lower() for v in script_types} if script_types else None
    wanted_bytes = set(bytes_per_char) if bytes_per_char else None

    if wanted_types:
        unknown = wanted_types - {t.lower() for t in known_types}
        if unknown:
            raise ValueError(f"Unknown --script-type value(s) {sorted(unknown)}. "
                             f"Available: {known_types}")
    if wanted_bytes:
        unknown = wanted_bytes - set(known_bytes)
        if unknown:
            raise ValueError(f"Unknown --bytes-per-char value(s) {sorted(unknown)}. "
                             f"Available: {known_bytes}")

    allowed = set()
    for code, row in langs_csv_data.items():
        st = (row.get("Script_Type") or "").strip().lower()
        bpc = (row.get("Approx_Bytes_Per_Char") or "").strip()
        if wanted_types and st not in wanted_types:
            continue
        if wanted_bytes and bpc not in wanted_bytes:
            continue
        allowed.add(code)
    return allowed


def resolve_language_tokens(tokens, codes):
    """Maps --ignore-outliers entries to language codes from `codes`.
    Accepts a code (amh_Ethi), its prefix (amh), the full name (Mandarin
    Chinese) or one word of the name (chinese), case-insensitive.
    Raises ValueError for unknown or ambiguous entries."""
    resolved = []
    for tok in tokens:
        t = tok.strip().lower()
        exact, partial = [], []
        for c in codes:
            name = CODE_TO_LANG_NAME.get(c, c).lower()
            if t in (c.lower(), c.split("_")[0].lower(), name):
                exact.append(c)
            elif t in name.split():
                partial.append(c)
        hits = exact or partial
        if len(hits) != 1:
            known = ", ".join(sorted(f"{CODE_TO_LANG_NAME.get(c, c)} ({c})" for c in codes))
            what = "matches several languages" if hits else "matches no language"
            raise ValueError(f"--ignore-outliers entry '{tok}' {what}. Available: {known}")
        if hits[0] not in resolved:
            resolved.append(hits[0])
    return resolved


def outlier_mask(langs, ignored_codes):
    """Bool per plotted language: True = excluded from the fit. Prints
    which languages are excluded, and notes requested ones not plotted."""
    mask = [l in ignored_codes for l in langs]
    if ignored_codes:
        name = lambda c: CODE_TO_LANG_NAME.get(c, c)
        shown = [c for c in ignored_codes if c in langs]
        absent = [c for c in ignored_codes if c not in langs]
        print(f"Excluded from fit (still plotted): {', '.join(name(c) for c in shown) or '(none)'}")
        if absent:
            print(f"Note: --ignore-outliers {', '.join(name(c) for c in absent)} not in the plot "
                  f"(filtered out or missing values) -- nothing to exclude.", file=sys.stderr)
    return mask


def make_filter_tag(script_types, bytes_per_char):
    """Filename-safe description of the active filters, '' if none."""
    tag = ""
    if script_types:
        tag += "_script-" + "+".join(sorted(v.lower() for v in script_types))
    if bytes_per_char:
        tag += "_bytes-" + "+".join(sorted(bytes_per_char))
    return tag


def build_coloring(langs, langs_csv_data, color_by):
    """None for color_by 'none'; otherwise per-point categories/colours,
    legend title and legend order (see module docstring)."""
    spec = COLOR_BY_OPTIONS.get(color_by)
    if spec is None:
        return None
    csv_col, title = spec
    palette = CATEGORY_PALETTES[csv_col]
    canon_by_lower = {k.lower(): k for k in palette}

    categories, colors = [], []
    split_first = {}
    for lang in langs:
        raw = (langs_csv_data.get(lang, {}).get(csv_col) or "").strip()
        canon = canon_by_lower.get(raw.lower())
        m_range = re.match(r"^\s*(.+?)\s*-\s*(.+?)\s*$", raw)
        if canon is not None:
            categories.append(canon)
            colors.append(palette[canon])
        elif (m_range and m_range.group(1).lower() in canon_by_lower
              and m_range.group(2).lower() in canon_by_lower):
            lo = canon_by_lower[m_range.group(1).lower()]
            hi = canon_by_lower[m_range.group(2).lower()]
            cat = f"{lo}-{hi}"
            categories.append(cat)
            colors.append((palette[lo], palette[hi]))
            split_first[cat] = lo
        else:
            categories.append(raw or "unknown")
            colors.append(UNKNOWN_CATEGORY_COLOR)

    present = set(categories)
    order = []
    for k in palette:
        if k in present:
            order.append(k)
        order += sorted(c for c in present if split_first.get(c) == k)
    order += sorted(present - set(order))
    return {"categories": categories, "colors": colors, "title": title, "order": order}


def is_split_color(color):
    """True for a (left, right) colour pair from a range category."""
    return isinstance(color, tuple)


def category_legend_handles(coloring):
    """One legend entry per category present, with its point count."""
    counts = Counter(coloring["categories"])
    color_of = dict(zip(coloring["categories"], coloring["colors"]))
    handles = []
    for cat in coloring["order"]:
        col = color_of[cat]
        style = dict(markerfacecolor=col[0], markerfacecoloralt=col[1], fillstyle="left") \
            if is_split_color(col) else dict(markerfacecolor=col)
        handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=10,
                              markeredgecolor="#333333", markeredgewidth=0.8,
                              label=f"{cat} (n={counts[cat]})", **style))
    return handles


def ignored_legend_handle(n):
    """Legend entry for points excluded via --ignore-outliers."""
    return Line2D([0], [0], marker="o", linestyle="", markersize=10, fillstyle="left",
                  markerfacecolor=UNKNOWN_CATEGORY_COLOR, markerfacecoloralt=IGNORED_FILL_COLOR,
                  markeredgecolor="#333333", markeredgewidth=0.8,
                  label=f"Excluded from fit (n={n})")


def load_density_json(path):
    """{language_code: {stat_name: value}} as produced by char_density.py."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_spread_json(path, fill_nulls=None):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if fill_nulls is not None:
        for lang_code, stats in data.items():
            if isinstance(stats, dict):
                for stat_key, val in stats.items():
                    if val is None:
                        stats[stat_key] = fill_nulls

    return data


def merge_external_columns(rows, langs_csv_data, density_data,
                            spread_data=None, spread_customenc_data=None):
    """Adds the external columns available for each row's language (in
    place). Returns the set of column names actually added."""
    added = set()
    spread_data = spread_data or {}
    spread_customenc_data = spread_customenc_data or {}
    for row in rows:
        lang = row.get("Language", "")
        if lang in langs_csv_data:
            src = langs_csv_data[lang]
            for col in LANGS_CSV_COLUMNS:
                if col in src and src[col] is not None and src[col] != "":
                    row[col] = src[col]
                    added.add(col)
        if lang in density_data:
            src = density_data[lang]
            for col in DENSITY_JSON_COLUMNS:
                if col in src and src[col] is not None:
                    row[col] = str(src[col])
                    added.add(col)
        if lang in spread_data:
            src = spread_data[lang]
            for src_field, out_col in SPREAD_JSON_FIELDS.items():
                if src_field in src and src[src_field] is not None:
                    row[out_col] = str(src[src_field])
                    added.add(out_col)
        if lang in spread_customenc_data:
            src = spread_customenc_data[lang]
            for src_field, out_col in SPREAD_CUSTOMENC_JSON_FIELDS.items():
                if src_field in src and src[src_field] is not None:
                    row[out_col] = str(src[src_field])
                    added.add(out_col)
    return added


# --- from --wordlen-json (word_length_stats.py) ---
# Languages with whitespace-delimited words (has_whitespace_words: true,
# incl. Vietnamese and Korean) use the "whitespace" values; languages
# without (Chinese, Japanese, Thai) use the MEAN over all their
# segmenters, whitespace excluded.

def _pick_word_value(method_values, has_whitespace_words):
    vals = method_values or {}
    if has_whitespace_words:
        return vals.get("whitespace")
    seg = [v for m, v in vals.items() if m != "whitespace" and v is not None]
    return sum(seg) / len(seg) if seg else None


def load_wordlen_json(path):
    """{lang: {word_length, n_words, n_char, char_ratio, word_ratio, _source}}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for lang, res in data.items():
        if not isinstance(res, dict):
            continue
        ws = res.get("has_whitespace_words", True)
        rel = res.get("rel_to_eng") or {}
        segs = [m for m in (res.get("n_words") or {}) if m != "whitespace"]
        out[lang] = {
            "word_length": _pick_word_value(res.get("avg_word_length"), ws),
            "n_words": _pick_word_value(res.get("n_words"), ws),
            "n_char": res.get("n_char"),
            "char_ratio": rel.get("n_char"),
            "word_ratio": _pick_word_value(rel.get("n_words"), ws),
            "_source": "whitespace" if ws else f"mean of {len(segs)} segmenter(s) ({', '.join(segs)})",
        }
    return out


def merge_wordlen_columns(rows, wordlen_data):
    """Adds WORDLEN_COLUMNS to each row; returns the columns added."""
    added = set()
    for row in rows:
        src = wordlen_data.get(row.get("Language", ""))
        if not src:
            continue
        for col in WORDLEN_COLUMNS:
            if src.get(col) is not None:
                row[col] = str(src[col])
                added.add(col)
    return added


RANK_SOURCE_COLUMNS = ["imbalanced_allocation_bytes", "balanced_allocation_bytes"]


def add_rank_columns(rows, source_columns=RANK_SOURCE_COLUMNS):
    """Adds '<column>_rank' (1 = largest value) for each source column."""
    added = set()
    for col in source_columns:
        pairs = []
        for row in rows:
            raw = row.get(col, "")
            if raw == "":
                continue
            try:
                pairs.append((row.get("Language", ""), float(raw)))
            except ValueError:
                continue
        if not pairs:
            continue
        pairs.sort(key=lambda p: -p[1])
        rank_by_lang = {lang: i + 1 for i, (lang, _) in enumerate(pairs)}
        rank_col = f"{col}_rank"
        for row in rows:
            lang = row.get("Language", "")
            if lang in rank_by_lang:
                row[rank_col] = str(rank_by_lang[lang])
                added.add(rank_col)
    return added


def add_log_columns(rows, source_columns=RANK_SOURCE_COLUMNS):
    """Adds '<column>_log10' for each source column (positive values)."""
    added = set()
    for col in source_columns:
        log_col = f"{col}_log10"
        n_skipped = 0
        for row in rows:
            raw = row.get(col, "")
            if raw == "":
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            if v <= 0:
                n_skipped += 1
                continue
            row[log_col] = str(math.log10(v))
            added.add(log_col)
        if n_skipped:
            print(f"Note: skipped {n_skipped} non-positive value(s) for {log_col} (log undefined)", file=sys.stderr)
    return added


def parse_premium_txt(path):
    """Returns (present_headers, rows) -- see module docstring for why
    this uses header-position slicing rather than whitespace-splitting."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    header_line = None
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip():
            header_line = line.rstrip("\n")
            header_idx = i
            break
    if header_line is None:
        raise ValueError(f"No header line found in {path}")

    present_headers = []
    positions = []
    search_from = 0
    for h in KNOWN_HEADERS:
        idx = header_line.find(h, search_from)
        if idx == -1:
            continue
        present_headers.append(h)
        positions.append(idx)
        search_from = idx + len(h)

    if not present_headers:
        raise ValueError(f"Could not find any known column headers in {path}")

    bounds = list(zip(positions, positions[1:] + [None]))

    rows = []
    for line in lines[header_idx + 1:]:
        if not line.strip():
            break
        if line.lstrip().startswith("#"):
            break
        row = {}
        for h, (start, end) in zip(present_headers, bounds):
            raw = line[start:end] if end is not None else line[start:]
            row[h] = raw.strip()
        rows.append(row)

    return present_headers, rows


def load_extra_premium_files(specs, char_level=False):
    """Loads additional premiums files given as NAME=FILE (repeatable
    --extra-premium). FILE is a literal path, or a shorthand
    'run:threshold' / 'run:threshold:char' (e.g. balanced-custom:mono).
    Every column except Language is merged in with the suffix _NAME
    (e.g. Premium -> Premium_mono). Returns {lang: {column: value}}."""
    merged = defaultdict(dict)
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"--extra-premium expects NAME=FILE, got '{spec}'")
        name, file_spec = (s.strip() for s in spec.split("=", 1))
        path = file_spec
        if not os.path.exists(file_spec) and ":" in file_spec:
            parts = file_spec.split(":")
            run, thr = parts[0], parts[1]
            char = char_level or (len(parts) > 2 and parts[2].lower() == "char")
            if thr not in ("mono", "global"):
                raise ValueError(f"Unknown threshold '{thr}' in '{file_spec}' (use mono/global)")
            path = resolve_premium_path(run, char, thr)
            if path is None:
                raise ValueError(f"Unknown run '{run}' in '{file_spec}'")
        if not os.path.exists(path):
            raise FileNotFoundError(f"--extra-premium {name}: file not found: {path}")
        print(f"Extra premiums '{name}': {path}")
        headers, extra_rows = parse_premium_txt(path)
        for r in extra_rows:
            lang = r.get("Language", "")
            for h in headers:
                if h == "Language" or r.get(h, "") == "":
                    continue
                merged[lang][f"{h}_{name}"] = r[h]
                DISPLAY_NAMES.setdefault(f"{h}_{name}", f"{DISPLAY_NAMES.get(h, h)} ({name})")
    return merged


def parse_col_spec(spec, available_columns):
    """One --cols/--control-for entry: plain column, ratio A/B, or
    log(...) around either. Returns (kind, payload)."""
    spec = spec.strip()
    m_log = re.match(r"^log\((.+)\)$", spec, re.IGNORECASE)
    if m_log:
        inner_kind, inner_payload = parse_col_spec(m_log.group(1), available_columns)
        return "log", (inner_kind, inner_payload)
    m_ratio = re.match(r"^([^/]+)/([^/]+)$", spec)
    if m_ratio:
        num = resolve_column_name(m_ratio.group(1), available_columns)
        den = resolve_column_name(m_ratio.group(2), available_columns)
        return "ratio", (num, den)
    return "plain", resolve_column_name(spec, available_columns)


def extract_values(kind, payload, rows):
    """Returns (values, label, langs), skipping rows with missing,
    non-numeric or (for log) non-positive values."""
    if kind == "log":
        inner_kind, inner_payload = payload
        inner_values, inner_label, inner_langs = extract_values(inner_kind, inner_payload, rows)
        values, langs = [], []
        n_skipped = 0
        for v, l in zip(inner_values, inner_langs):
            if v <= 0:
                n_skipped += 1
                continue
            values.append(math.log10(v))
            langs.append(l)
        if n_skipped:
            print(f"Note: skipped {n_skipped} non-positive value(s) taking log10 of {inner_label}", file=sys.stderr)
        return values, f"log10({inner_label})", langs

    values = []
    langs = []
    if kind == "plain":
        header = payload
        for r in rows:
            raw = r.get(header, "")
            if raw == "":
                continue
            try:
                values.append(float(raw))
                langs.append(r.get("Language", ""))
            except ValueError:
                continue
        label = header
    else:
        num_h, den_h = payload
        for r in rows:
            raw_num, raw_den = r.get(num_h, ""), r.get(den_h, "")
            if raw_num == "" or raw_den == "":
                continue
            try:
                num_v, den_v = float(raw_num), float(raw_den)
                if den_v == 0:
                    continue
                values.append(num_v / den_v)
                langs.append(r.get("Language", ""))
            except ValueError:
                continue
        label = f"{num_h}/{den_h}"
    return values, label, langs


def pearson_r_p(x, y, extra_df_used=0):
    """Pearson r and two-tailed p. extra_df_used = degrees of freedom
    already spent on control regressions (1 per control variable)."""
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


def p_str(p):
    if p != p:  # NaN
        return "n/a"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


# ---------------------------------------------------------------------------
# Groups (--group-by), residual breakdown and common-slope fit
# ---------------------------------------------------------------------------

def assign_groups(langs, langs_csv_data, group_by):
    """Per-language group label (or None if unknown) for --group-by
    'bytes' / 'dense'. Returns (labels, (first_label, second_label),
    footnote), or (None, None, None) for 'none'."""
    if group_by not in GROUP_DEFS:
        return None, None, None
    first, second, note = GROUP_DEFS[group_by]
    labels = []
    for l in langs:
        row = langs_csv_data.get(l, {})
        if group_by == "bytes":
            b = (row.get("Approx_Bytes_Per_Char") or "").strip()
            labels.append(None if b == "" else (first if b in SINGLE_BYTE_VALUES else second))
        else:
            st = (row.get("Script_Type") or "").strip().lower()
            labels.append(None if st == "" else (second if st in DENSE_SCRIPT_TYPES else first))
    return labels, (first, second), note


def _group_stats(resid, mask, langs, total_ss):
    e = resid[mask]
    m = float(e.mean())
    idx_max = int(np.argmax(np.abs(e)))
    return {
        "n": int(mask.sum()),
        "mean_resid": m,
        "rmse": float(np.sqrt((e ** 2).mean())),
        "max_abs": float(abs(e[idx_max])),
        "max_lang": [l for l, k in zip(langs, mask) if k][idx_max],
        "share": float((e ** 2).sum()) / total_ss,
        "offset_share": len(e) * m ** 2 / total_ss,
        "scatter_share": float(((e - m) ** 2).sum()) / total_ss,
    }


def residual_breakdown(x, y, langs, groups, order, min_per_group=3):
    """Breaks the residuals of the single joint line y ~ x down by the two
    groups in `order` (labels from assign_groups). Returns None if either
    group has fewer than min_per_group languages."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    masks = [np.array([g == lab for g in groups]) for lab in order]
    if any(m.sum() < min_per_group for m in masks):
        return None

    coefs = np.polyfit(x, y, 1)
    resid = y - np.polyval(coefs, x)
    total_ss = float((resid ** 2).sum())
    if total_ss <= 0:
        return None

    out = {
        "slope_all": float(coefs[0]), "intercept_all": float(coefs[1]),
        "order": list(order),
        "groups": [dict(label=lab, **_group_stats(resid, m, langs, total_ss))
                   for lab, m in zip(order, masks)],
    }
    out["offset_share"] = sum(g["offset_share"] for g in out["groups"])

    # Out-of-sample: fit on the second group alone, predict the first.
    coefs_b = np.polyfit(x[masks[1]], y[masks[1]], 1)
    e_oos = y[masks[0]] - np.polyval(coefs_b, x[masks[0]])
    out["oos"] = {
        "slope": float(coefs_b[0]), "intercept": float(coefs_b[1]),
        "mean_resid": float(e_oos.mean()),
        "rmse": float(np.sqrt((e_oos ** 2).mean())),
        "max_abs": float(np.abs(e_oos).max()),
    }
    return out


def print_breakdown(bd, oos="auto", note=""):
    """Residual breakdown of the joint line as a readable block."""
    name = lambda c: CODE_TO_LANG_NAME.get(c, c)
    print("\nResidual breakdown (single line fitted to all languages in the fit, "
          f"slope {bd['slope_all']:.3f}, intercept {bd['intercept_all']:.3f}):")
    print(f"  {'group':<12}{'n':>3}  {'mean res.':>9}  {'RMSE':>6}  {'max |res|':>24}  "
          f"{'share':>6}  {'offset':>6}  {'scatter':>7}")
    for g in bd["groups"]:
        max_str = f"{g['max_abs']:.3f} ({name(g['max_lang'])})"
        print(f"  {g['label']:<12}{g['n']:>3}  {g['mean_resid']:>+9.3f}  {g['rmse']:>6.3f}  "
              f"{max_str:>24}  {g['share']:>6.1%}  {g['offset_share']:>6.1%}  "
              f"{g['scatter_share']:>7.1%}")
    print(f"  total group offset: {bd['offset_share']:.1%} of the unexplained variance")
    run_oos = oos == "always" or (oos == "auto" and bd["offset_share"] < OOS_OFFSET_THRESHOLD)
    if run_oos:
        o = bd["oos"]
        a, b = bd["order"]
        print(f"  Out-of-sample: line fitted to {b} languages only "
              f"(slope {o['slope']:.3f}, intercept {o['intercept']:.3f})")
        print(f"    predicting {a} languages: mean res. {o['mean_resid']:+.3f}, "
              f"RMSE {o['rmse']:.3f}, max |res| {o['max_abs']:.3f}")
    elif oos == "auto":
        print(f"  (out-of-sample check skipped: group offset >= {OOS_OFFSET_THRESHOLD:.0%}; "
              f"use --oos always to run it anyway)")
    if note:
        print(f"  ({note})")


def breakdown_legend_lines(bd):
    """Compact summary of the joint-line breakdown for the legend: each
    group's share of the unexplained variance = offset + scatter."""
    lines = ["Unexplained variance (joint line):"]
    for g in bd["groups"]:
        lines.append(f"   {g['label']} {g['share']:.1%} = offset {g['offset_share']:.1%} "
                     f"+ scatter {g['scatter_share']:.1%}")
    return lines


def group_fit(x, y, langs, groups, order, min_per_group=3):
    """Common slope + per-group intercept: y = b*x + a_g. Returns None if
    either group has fewer than min_per_group languages (languages with
    no group are left out of this fit)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.array([g in order for g in groups])
    x, y = x[keep], y[keep]
    g_arr = [g for g in groups if g in order]
    l_arr = [l for l, k in zip(langs, keep) if k]
    masks = [np.array([g == lab for g in g_arr]) for lab in order]
    if any(m.sum() < min_per_group for m in masks):
        return None

    # Design: x plus one indicator per group (no separate intercept).
    X = np.column_stack([x] + [m.astype(float) for m in masks])
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    slope, intercepts = float(coefs[0]), [float(c) for c in coefs[1:]]
    pred = X @ coefs
    resid = y - pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    joint = np.polyfit(x, y, 1)
    ss_joint = float(((y - np.polyval(joint, x)) ** 2).sum())

    # Partial r of x and y controlling for the group = correlation of
    # the within-group deviations (one df spent on the group indicator).
    x_w = x.copy()
    y_w = y.copy()
    for m in masks:
        x_w[m] -= x[m].mean()
        y_w[m] -= y[m].mean()
    r_p, p_p = pearson_r_p(x_w, y_w, extra_df_used=len(order) - 1)

    separate = []
    for lab, m in zip(order, masks):
        s, i = np.polyfit(x[m], y[m], 1)
        separate.append({"label": lab, "slope": float(s), "intercept": float(i),
                         "x_min": float(x[m].min()), "x_max": float(x[m].max())})

    groups_out = []
    for lab, m, a in zip(order, masks, intercepts):
        e = resid[m]
        idx_max = int(np.argmax(np.abs(e)))
        groups_out.append({
            "label": lab, "n": int(m.sum()), "intercept": a,
            "rmse": float(np.sqrt((e ** 2).mean())),
            "share": float((e ** 2).sum()) / ss_res if ss_res > 0 else float("nan"),
            "max_abs": float(abs(e[idx_max])),
            "max_lang": [l for l, k in zip(l_arr, m) if k][idx_max],
            "x_min": float(x[m].min()), "x_max": float(x[m].max()),
        })
    return {
        "slope": slope, "order": list(order), "groups": groups_out,
        "offset": intercepts[1] - intercepts[0],
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "r2_joint": 1 - ss_joint / ss_tot if ss_tot > 0 else float("nan"),
        "partial_r": r_p, "partial_p": p_p,
        "separate": separate,
    }


def print_group_fit(gf, note=""):
    name = lambda c: CODE_TO_LANG_NAME.get(c, c)
    a, b = gf["order"]
    print(f"\nCommon slope + group offset (y = slope*x + intercept_group):")
    print(f"  common slope {gf['slope']:.3f}; offset {b} vs. {a}: {gf['offset']:+.3f}")
    for g in gf["groups"]:
        print(f"    {g['label']:<12} n={g['n']:<3} intercept {g['intercept']:.3f}  "
              f"RMSE {g['rmse']:.3f}  max |res| {g['max_abs']:.3f} ({name(g['max_lang'])})  "
              f"share of remaining unexplained variance {g['share']:.1%}")
    print(f"  R^2: {gf['r2']:.3f} (single joint line: {gf['r2_joint']:.3f})")
    print(f"  partial r(x, y | group) = {gf['partial_r']:.4f}  r^2 = {gf['partial_r'] ** 2:.4f}  "
          f"p = {p_str(gf['partial_p'])}")
    sep = ", ".join(f"{s['label']} {s['slope']:.3f}" for s in gf["separate"])
    print(f"  slopes if fitted separately: {sep}")
    if note:
        print(f"  ({note})")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def label_points(ax, x, y, langs, avoid=None, muted=None):
    """Language-name labels next to each point (adjustText if installed).
    muted: optional bool per point -- drawn grey italic (excluded points)."""
    names = [CODE_TO_LANG_NAME.get(l, l) for l in langs] if langs else []
    if not names:
        return []
    muted = list(muted) if muted is not None else [False] * len(names)
    style = lambda i: (dict(color=IGNORED_LABEL_COLOR, fontstyle="italic") if muted[i]
                       else dict(color="#222222"))
    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None

    if adjust_text is not None:
        texts = [ax.text(xv, yv, nm, fontsize=FS_POINT_LABEL, zorder=4, **style(i))
                 for i, (xv, yv, nm) in enumerate(zip(x, y, names))]
        arrows = dict(arrowstyle="-", color="#999999", lw=0.6)
        objs = [a for a in (avoid or []) if a is not None]
        try:
            adjust_text(texts, x=list(x), y=list(y), ax=ax, expand=(1.4, 1.8),
                        objects=objs or None, arrowprops=arrows)
        except TypeError:
            adjust_text(texts, x=list(x), y=list(y), ax=ax, arrowprops=arrows)
        return texts

    x_groups = defaultdict(list)
    for i, xv in enumerate(x):
        x_groups[round(float(xv), 2)].append(i)
    texts = []
    for idxs in x_groups.values():
        idxs.sort(key=lambda i: y[i], reverse=True)
        for rank, i in enumerate(idxs):
            texts.append(ax.annotate(
                names[i], (x[i], y[i]), textcoords="offset points",
                xytext=(8, 6 + rank * 14), fontsize=FS_POINT_LABEL, **style(i),
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5, shrinkA=0, shrinkB=4)))
    return texts


def scatter_with_fit(ax, x, y, langs, xlabel, ylabel, title, extra_df_used=0,
                     coloring=None, show_category_legend=True, subtitle=None,
                     breakdown=None, invert_x=False, invert_y=False, compact=False,
                     groupfit=None, ignore_mask=None):
    """Scatter plot with regression line and one combined legend.
    groupfit: optional dict from group_fit -- draws one parallel line per
    group (common slope) and lists them in the legend; the joint line is
    then drawn lighter for reference.
    ignore_mask: optional bool per point -- those points are drawn (half
    black) but left out of the fit and of r / p."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ignored = (np.zeros(len(x), dtype=bool) if ignore_mask is None
               else np.asarray(ignore_mask, dtype=bool))
    keep = ~ignored
    xk, yk = x[keep], y[keep]
    r, p = pearson_r_p(xk, yk, extra_df_used=extra_df_used)
    scale = 0.85 if compact else 1.0

    cols = coloring["colors"] if coloring is not None else [DEFAULT_POINT_COLOR] * len(x)
    solid = [i for i, col in enumerate(cols) if keep[i] and not is_split_color(col)]
    if solid:
        ax.scatter(x[solid], y[solid], c=[cols[i] for i in solid], s=POINT_SIZE,
                   edgecolors="#333333", linewidths=0.8, zorder=3)
    for i, col in enumerate(cols):
        if keep[i] and not is_split_color(col):
            continue
        if ignored[i]:
            # Excluded point: own colour on the left, black on the right.
            left, right = (col[0] if is_split_color(col) else col), IGNORED_FILL_COLOR
        else:
            left, right = col
        ax.plot(x[i], y[i], marker="o", linestyle="", markersize=math.sqrt(POINT_SIZE),
                fillstyle="left", markerfacecolor=left, markerfacecoloralt=right,
                markeredgecolor="#333333", markeredgewidth=0.8, zorder=3)

    handles = []
    if coloring is not None and show_category_legend:
        handles += category_legend_handles(coloring)
    if ignored.any() and show_category_legend:
        handles.append(ignored_legend_handle(int(ignored.sum())))
    fit_line = None
    if len(xk) >= 2:
        coefs = np.polyfit(xk, yk, 1)
        x_line = np.linspace(xk.min(), xk.max(), 100)
        joint_kw = dict(color="red", linewidth=1.8, alpha=1.0)
        if groupfit is not None:
            joint_kw = dict(color="#999999", linewidth=1.4, alpha=0.8)
        fit_handle, = ax.plot(x_line, np.polyval(coefs, x_line), linestyle="--", zorder=2,
                               label=f"{'Joint fit' if groupfit else 'Linear fit'}: "
                                     f"r = {0 if round(r, 2) == 0 else r:.2f}, "
                                     f"r$^2$ = {r * r:.2f}, p = {p_str(p)}",
                               **joint_kw)
        handles.append(fit_handle)
        fit_line = fit_handle
    if groupfit is not None:
        for g, col in zip(groupfit["groups"], GROUP_LINE_COLORS):
            xs = np.linspace(g["x_min"], g["x_max"], 50)
            h, = ax.plot(xs, groupfit["slope"] * xs + g["intercept"], linestyle="-",
                         color=col, linewidth=2.2, zorder=2,
                         label=f"{g['label']} (n={g['n']}): intercept {g['intercept']:.2f}")
            handles.append(h)
        rp = groupfit["partial_r"]
        handles.append(Line2D([], [], linestyle="none", marker="none",
                              label=f"Common slope {groupfit['slope']:.2f}, offset "
                                    f"{groupfit['offset']:+.2f}: R$^2$ = {groupfit['r2']:.2f}"))
        handles.append(Line2D([], [], linestyle="none", marker="none",
                              label=f"Partial r (controlling for group) = {rp:.2f}, "
                                    f"p = {p_str(groupfit['partial_p'])}"))
    if breakdown is not None:
        handles += [Line2D([], [], linestyle="none", marker="none", label=line)
                    for line in breakdown_legend_lines(breakdown)]

    ax.set_xlabel(xlabel, fontsize=FS_AXIS_LABEL * scale)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS_LABEL * scale)
    ax.tick_params(labelsize=FS_TICKS * scale)
    if subtitle:
        ax.set_title(title, fontsize=FS_TITLE * scale, fontweight="bold", pad=26)
        ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=FS_SUBTITLE * scale, color="#555555")
    else:
        ax.set_title(title, fontsize=FS_TITLE * scale, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)

    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()

    legend = None
    if handles:
        title_kw = {}
        if coloring is not None and show_category_legend:
            title_kw = dict(title=coloring["title"], title_fontsize=FS_LEGEND_TITLE * scale)
        legend_kw = dict(handles=handles, loc=LEGEND_LOC, fontsize=FS_LEGEND * scale,
                         framealpha=0.92, **title_kw)
        try:
            legend = ax.legend(alignment="left", **legend_kw)  # matplotlib >= 3.6
        except TypeError:
            legend = ax.legend(**legend_kw)
        legend.set_zorder(5)

    texts = label_points(ax, x, y, langs, avoid=[legend], muted=ignored)

    ax._overlap_check = dict(legend=legend, x=x, y=y, langs=list(langs or []),
                             texts=texts, fit_line=fit_line, title=title)

    return r, p


def warn_legend_overlap(fig, ax, where=""):
    """Warns on stderr if the legend covers points, labels or the fit line."""
    info = getattr(ax, "_overlap_check", None)
    if not info or info["legend"] is None:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = info["legend"].get_window_extent(renderer)
    name = lambda c: CODE_TO_LANG_NAME.get(c, c)

    radius = math.sqrt(POINT_SIZE) / 2 * fig.dpi / 72
    pts = ax.transData.transform(np.column_stack([info["x"], info["y"]]))
    hit_points = [name(l) for (px, py), l in zip(pts, info["langs"])
                  if box.x0 - radius <= px <= box.x1 + radius
                  and box.y0 - radius <= py <= box.y1 + radius]
    hit_labels = [t.get_text() for t in info["texts"]
                  if t.get_window_extent(renderer).overlaps(box)]
    hit_line = False
    if info["fit_line"] is not None:
        lx, ly = info["fit_line"].get_data()
        lp = ax.transData.transform(np.column_stack([lx, ly]))
        hit_line = any(box.contains(px, py) for px, py in lp)

    parts = []
    if hit_points:
        parts.append(f"points ({', '.join(hit_points)})")
    if hit_labels:
        parts.append(f"labels ({', '.join(hit_labels)})")
    if hit_line:
        parts.append("the regression line")
    if parts:
        where_str = f" [{where}]" if where else ""
        print(f"Warning{where_str}: legend ({LEGEND_LOC}) overlaps {'; '.join(parts)}. "
              f"Try --legend-loc to move it.", file=sys.stderr)


def involves_autocorr(raw_label):
    """True if a raw column label involves lag-1 autocorrelation."""
    return "EntropyAutocorr1" in raw_label


def plot_plain(x, y, x_label, y_label, langs, subtitle, out_path, coloring=None,
               breakdown=None, groupfit=None, ignore_mask=None):
    """x_label/y_label are the RAW column labels (prettified here)."""
    fig, ax = plt.subplots(figsize=(10.5, 8))
    r, p = scatter_with_fit(ax, x, y, langs, pretty_label(x_label), pretty_label(y_label),
                             f"{pretty_label(y_label)} vs. {pretty_label(x_label)}",
                             coloring=coloring, subtitle=subtitle, breakdown=breakdown,
                             invert_x=involves_autocorr(x_label),
                             invert_y=involves_autocorr(y_label), groupfit=groupfit,
                             ignore_mask=ignore_mask)
    plt.tight_layout()
    warn_legend_overlap(fig, ax, os.path.basename(out_path))
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return r, p


def plot_partial(x, y, ctrl, x_label, y_label, ctrl_label, langs, subtitle, out_path,
                 out_path_single=None, coloring=None, ignore_mask=None):
    """6-panel partial-correlation walkthrough plus a separate
    residual-vs-residual plot. Labels are the RAW column labels.
    ignore_mask: points excluded from all fits (incl. the control
    regressions); their residuals are taken from the fits to the rest."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.asarray(ctrl, dtype=float)
    px, py, pc = pretty_label(x_label), pretty_label(y_label), pretty_label(ctrl_label)
    inv_x, inv_y, inv_c = (involves_autocorr(l) for l in (x_label, y_label, ctrl_label))
    keep = (np.ones(len(x), dtype=bool) if ignore_mask is None
            else ~np.asarray(ignore_mask, dtype=bool))

    x_fit = np.polyfit(c[keep], x[keep], 1)
    x_resid = x - np.polyval(x_fit, c)

    y_fit = np.polyfit(c[keep], y[keep], 1)
    y_resid = y - np.polyval(y_fit, c)

    if out_path_single is None:
        base, ext = os.path.splitext(out_path)
        out_path_single = f"{base}_partial_only{ext}"

    fig_single, ax_single = plt.subplots(figsize=(10.5, 8))
    scatter_with_fit(
        ax_single, x_resid, y_resid, langs,
        pretty_label(f"{x_label} residual"), pretty_label(f"{y_label} residual"),
        f"{py} vs. {px}, controlling for {pc}",
        extra_df_used=1, coloring=coloring, subtitle=subtitle,
        invert_x=inv_x, invert_y=inv_y, ignore_mask=ignore_mask,
    )
    plt.tight_layout()
    warn_legend_overlap(fig_single, ax_single, os.path.basename(out_path_single))
    plt.savefig(out_path_single, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig_single)

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    kw = dict(coloring=coloring, show_category_legend=False, compact=True,
              ignore_mask=ignore_mask)

    scatter_with_fit(axes[0, 0], c, x, langs, pc, px,
                      f"(a) Fit {px} ~ {pc}", invert_x=inv_c, invert_y=inv_x, **kw)
    scatter_with_fit(axes[0, 1], c, x_resid, langs, pc, f"{px} (residual)",
                      f"(b) Leftover {px}\n(should look flat vs. {pc})",
                      invert_x=inv_c, invert_y=inv_x, **kw)

    r_raw, p_raw = scatter_with_fit(axes[0, 2], x, y, langs, px, py,
                                     f"(c) Reference: raw {py} vs. {px}",
                                     invert_x=inv_x, invert_y=inv_y, **kw)
    for spine in axes[0, 2].spines.values():
        spine.set_linestyle((0, (4, 3)))
        spine.set_color("#999999")
    axes[0, 2].set_facecolor("#f5f5f5")

    scatter_with_fit(axes[1, 0], c, y, langs, pc, py,
                      f"(d) Fit {py} ~ {pc}", invert_x=inv_c, invert_y=inv_y, **kw)
    scatter_with_fit(axes[1, 1], c, y_resid, langs, pc, f"{py} (residual)",
                      f"(e) Leftover {py}\n(should look flat vs. {pc})",
                      invert_x=inv_c, invert_y=inv_y, **kw)

    r_partial, p_partial = scatter_with_fit(
        axes[1, 2], x_resid, y_resid, langs, f"{px} (residual)", f"{py} (residual)",
        "(f) Leftover vs. leftover\n= partial correlation", extra_df_used=1,
        invert_x=inv_x, invert_y=inv_y, **kw)

    fig.suptitle(f"{py} vs. {px}, controlling for {pc}\n"
                 f"{subtitle}  |  raw r = {r_raw:.2f} (p = {p_str(p_raw)}), "
                 f"partial r = {r_partial:.2f} (p = {p_str(p_partial)})",
                 fontsize=FS_TITLE + 2, fontweight="bold", y=1.04)
    plt.tight_layout()
    for panel, a in zip("abcdef", axes.flat):
        warn_legend_overlap(fig, a, f"{os.path.basename(out_path)}, panel ({panel})")
    n_ignored = int(np.sum(~keep))
    if coloring is not None or n_ignored:
        handles = category_legend_handles(coloring) if coloring is not None else []
        if n_ignored:
            handles.append(ignored_legend_handle(n_ignored))
        fig.legend(handles=handles, title=coloring["title"] if coloring else None,
                   loc="upper center",
                   bbox_to_anchor=(0.5, 0.0), ncol=min(len(handles), 6),
                   fontsize=FS_LEGEND + 1, title_fontsize=FS_LEGEND_TITLE + 1, framealpha=0.9)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved walkthrough grid: {out_path}")
    print(f"Saved partial-only plot: {out_path_single}")

    return r_raw, p_raw, r_partial, p_partial


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("premium_file",
                         help="A *_premiums_sorted.txt file, OR a shorthand run name -- "
                              "balanced / imbalanced / balanced-custom (case-insensitive), "
                              "resolved via RUN_INFO together with --threshold / --char-level.")
    parser.add_argument("--threshold", choices=["mono", "global"], required=True,
                         help="'global' = raw_entropy, 'mono' = raw_monotonicity. Required "
                              "always (unused for a literal path).")
    parser.add_argument("--char-level", action="store_true",
                         help="Resolve a shorthand run name to the char_level/ version.")
    parser.add_argument("--cols", required=True,
                         help="Two comma-separated column specs, X,Y (plain, ratio A/B, or log(...)).")
    parser.add_argument("--control-for", default=None,
                         help="A third column spec to control for -- partial correlation with "
                              "the 6-panel walkthrough instead of a plain scatter.")
    parser.add_argument("--script-type", default=None,
                         help="Restrict to these Script_Type values (comma-separated, "
                              "case-insensitive). Requires --langs-csv.")
    parser.add_argument("--bytes-per-char", default=None,
                         help="Restrict to these Approx_Bytes_Per_Char values (comma-separated, "
                              "exact). Requires --langs-csv.")
    parser.add_argument("--color-by", default=None,
                         type=lambda s: s.strip().lower().replace("_", "-"),
                         choices=list(COLOR_BY_OPTIONS),
                         help="Colour-code points: 'bytes-per-char' (default), 'script-type', "
                              "or 'none'.")
    parser.add_argument("--group-by", default="bytes", choices=["bytes", "dense", "none"],
                         help="Grouping for the residual breakdown and --group-fit: 'bytes' "
                              "(single- vs. multi-byte, default), 'dense' (Dense_Script column "
                              "of --langs-csv), or 'none'.")
    parser.add_argument("--group-fit", action="store_true",
                         help="Plain plots: also fit a common slope with one intercept per group "
                              "(see --group-by), draw the parallel group lines, and report R^2, "
                              "the offset and the partial correlation controlling for the group.")
    parser.add_argument("--breakdown", default="print", choices=["print", "legend", "off"],
                         help="Residual breakdown of the joint line by group (plain plots): "
                              "'print' (default), 'legend' (also in the plot legend), or 'off'.")
    parser.add_argument("--legend-loc", default="lower-left",
                         type=lambda v: v.strip().lower().replace("_", "-").replace(" ", "-"),
                         choices=["lower-left", "lower-right", "upper-left", "upper-right",
                                  "lower-center", "upper-center", "center-left", "center-right"],
                         help="Legend position inside the plot (default: lower-left).")
    parser.add_argument("--ignore-outliers", "--ignore_outliers", dest="ignore_outliers",
                         default=None, metavar="LANGS",
                         help="Comma-separated languages to keep in the plot but leave out of the "
                              "fit, r/p, breakdown and --group-fit (code, code prefix, full name "
                              "or one word of it, e.g. amharic,japanese,chinese,korean). Drawn "
                              "half black.")
    parser.add_argument("--oos", default="auto", choices=["auto", "always", "never"],
                         help="Out-of-sample check in the printed breakdown: 'auto' (default), "
                              "'always' or 'never'.")
    parser.add_argument("--langs-csv", default="training_setup/langs/langs_chosen.csv",
                         help="CSV with a 'language_code' column plus typology/training-data "
                              "columns (incl. Script_Type, Approx_Bytes_Per_Char, Dense_Script). "
                              "Silently skipped if not found. Empty string disables it.")
    parser.add_argument("--density-json", default="char_density.json",
                         help="JSON from char_density.py. Silently skipped if not found.")
    parser.add_argument("--spread-json", default="byte_position_stats.json",
                         help="JSON from byte_position_stats.py ('spread' columns). "
                              "Silently skipped if not found.")
    parser.add_argument("--spread-customenc-json", default="byte_position_stats_customenc.json",
                         help="Same for the custom-encoding run ('spread_customenc' columns). "
                              "Silently skipped if not found.")
    parser.add_argument("--wordlen-json", default="word_length_stats.json",
                         help="JSON from word_length_stats.py, merged in as word_length, n_words, "
                              "n_char, char_ratio and word_ratio. Silently skipped if not found.")
    parser.add_argument("--extra-premium", action="append", default=[], metavar="NAME=FILE",
                         help="Merge in another premiums file, with every column suffixed _NAME "
                              "(e.g. Premium_mono). FILE is a literal path or a shorthand like "
                              "balanced-custom:mono (add :char for char level). Repeatable.")
    parser.add_argument("--fill-spread-nulls", action="store_true",
                         help="Replace null spread values (e.g. for 1-byte languages) with 1.0 "
                              "to include all languages in charts.")
    parser.add_argument("--out", default=None, help="Output PNG path (overrides --out-dir).")
    parser.add_argument("--out-dir", default="correlation_plots",
                         help="Directory for the auto-derived output filename "
                              "(default: correlation_plots/).")
    args = parser.parse_args()
    global LEGEND_LOC
    LEGEND_LOC = args.legend_loc.replace("-", " ")

    resolved_path = resolve_premium_path(args.premium_file, args.char_level, args.threshold)
    premium_path = resolved_path if resolved_path is not None else args.premium_file
    if resolved_path is not None:
        print(f"Resolved '{args.premium_file}' -> {premium_path}")

    present_headers, rows = parse_premium_txt(premium_path)
    available_columns = set(present_headers)

    ignored_codes = []
    if args.ignore_outliers:
        try:
            ignored_codes = resolve_language_tokens(
                parse_filter_values(args.ignore_outliers) or [],
                [r.get("Language", "") for r in rows])
        except ValueError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

    langs_csv_data = {}
    if args.langs_csv and os.path.exists(args.langs_csv):
        langs_csv_data = load_langs_csv(args.langs_csv)
    density_data = {}
    if args.density_json and os.path.exists(args.density_json):
        density_data = load_density_json(args.density_json)
    fill_value = 1.0 if args.fill_spread_nulls else None

    spread_data = {}
    if args.spread_json and os.path.exists(args.spread_json):
        spread_data = load_spread_json(args.spread_json, fill_nulls=fill_value)
    spread_customenc_data = {}
    if args.spread_customenc_json and os.path.exists(args.spread_customenc_json):
        spread_customenc_data = load_spread_json(args.spread_customenc_json, fill_nulls=fill_value)

    available_columns |= merge_external_columns(
        rows, langs_csv_data, density_data, spread_data, spread_customenc_data)

    wordlen_data = {}
    if args.wordlen_json and os.path.exists(args.wordlen_json):
        wordlen_data = load_wordlen_json(args.wordlen_json)
        if "word" in (args.cols + (args.control_for or "")).lower():
            for lang, e in wordlen_data.items():
                if e["_source"] != "whitespace":
                    print(f"Word stats for {CODE_TO_LANG_NAME.get(lang, lang)}: {e['_source']}")
    available_columns |= merge_wordlen_columns(rows, wordlen_data)

    try:
        extra_data = load_extra_premium_files(args.extra_premium, args.char_level)
    except (ValueError, FileNotFoundError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    for row in rows:
        for col, val in extra_data.get(row.get("Language", ""), {}).items():
            row[col] = val
            available_columns.add(col)

    # Ranks are computed over ALL languages here, BEFORE the language
    # filter below is applied, so a language keeps the same
    # training_data rank in filtered and unfiltered plots.
    available_columns |= add_rank_columns(rows)
    available_columns |= add_log_columns(rows)

    if args.color_by is None:
        if langs_csv_data:
            args.color_by = "bytes-per-char"
        else:
            print(f"Note: no --langs-csv loaded (path: '{args.langs_csv}'), so points are "
                  f"not colour-coded.", file=sys.stderr)
            args.color_by = "none"
    elif args.color_by != "none" and not langs_csv_data:
        print(f"--color-by {args.color_by} needs Script_Type / Approx_Bytes_Per_Char from "
              f"--langs-csv, but no CSV was loaded (path: '{args.langs_csv}').", file=sys.stderr)
        sys.exit(1)

    if args.group_by == "dense" and not any(
            (r.get("Script_Type") or "").strip() for r in langs_csv_data.values()):
        print(f"--group-by dense needs the Script_Type column of --langs-csv "
              f"('{args.langs_csv}'), but none was found.", file=sys.stderr)
        sys.exit(1)

    # Derived 0/1 column, so --control-for dense still works as a control.
    for row in rows:
        st = (langs_csv_data.get(row.get("Language", ""), {}).get("Script_Type") or "").strip().lower()
        if st:
            row["Dense_Script"] = "1" if st in DENSE_SCRIPT_TYPES else "0"
            available_columns.add("Dense_Script")
    if args.group_fit and args.group_by == "none":
        print("--group-fit needs --group-by bytes or dense.", file=sys.stderr)
        sys.exit(1)

    # --- LANGUAGE FILTER (--script-type / --bytes-per-char) ---
    script_types = parse_filter_values(args.script_type)
    bytes_per_char = parse_filter_values(args.bytes_per_char)
    filter_tag = make_filter_tag(script_types, bytes_per_char)
    if script_types or bytes_per_char:
        if not langs_csv_data:
            print(f"--script-type / --bytes-per-char need Script_Type and Approx_Bytes_Per_Char "
                  f"from --langs-csv, but no CSV was loaded (path: '{args.langs_csv}').",
                  file=sys.stderr)
            sys.exit(1)
        try:
            allowed = select_languages(langs_csv_data, script_types, bytes_per_char)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        kept = [r.get("Language", "") for r in rows if r.get("Language", "") in allowed]
        dropped = [r.get("Language", "") for r in rows if r.get("Language", "") not in allowed]
        rows = [r for r in rows if r.get("Language", "") in allowed]
        name = lambda c: CODE_TO_LANG_NAME.get(c, c)
        crit = []
        if script_types:
            crit.append(f"Script_Type in {script_types}")
        if bytes_per_char:
            crit.append(f"Approx_Bytes_Per_Char in {bytes_per_char}")
        print(f"Language filter ({' AND '.join(crit)}): keeping {len(kept)} of "
              f"{len(kept) + len(dropped)} languages")
        print(f"  kept:    {', '.join(name(c) for c in kept) or '(none)'}")
        print(f"  dropped: {', '.join(name(c) for c in dropped) or '(none)'}")
        if not rows:
            print("No languages left after filtering.", file=sys.stderr)
            sys.exit(1)

    col_specs = [s.strip() for s in args.cols.split(",")]
    if len(col_specs) != 2:
        print(f"--cols must have exactly two comma-separated entries, got {len(col_specs)}: {col_specs}", file=sys.stderr)
        sys.exit(1)

    x_kind, x_payload = parse_col_spec(col_specs[0], available_columns)
    y_kind, y_payload = parse_col_spec(col_specs[1], available_columns)

    x_vals, x_label, x_langs = extract_values(x_kind, x_payload, rows)
    y_vals, y_label, y_langs = extract_values(y_kind, y_payload, rows)

    x_by_lang = dict(zip(x_langs, x_vals))
    y_by_lang = dict(zip(y_langs, y_vals))
    common_langs = [l for l in x_by_lang if l in y_by_lang]

    stem = os.path.splitext(os.path.basename(premium_path))[0] + filter_tag
    subtitle = describe_run(os.path.splitext(os.path.basename(premium_path))[0], args.char_level)
    filter_desc = describe_filter(script_types, bytes_per_char)
    if filter_desc:
        subtitle += f" \u2014 {filter_desc}"
    if ignored_codes:
        subtitle += (" \u2014 excluded from fit: "
                     + ", ".join(CODE_TO_LANG_NAME.get(c, c) for c in ignored_codes))
    ignore_tag = ("_ignore-" + "+".join(sorted(c.split("_")[0] for c in ignored_codes))
                  if ignored_codes else "")
    color_tag = "" if args.color_by == "none" else f"_color-{args.color_by}"
    color_tag += ignore_tag
    run_key = RUN_NAME_LOOKUP.get(args.premium_file.strip().lower())
    setting_dir = run_key.lower() if run_key else "other"

    if args.char_level:
        setting_dir = os.path.join("char_level", setting_dir)
    out_dir = os.path.join(args.out_dir, setting_dir, args.threshold)

    if args.control_for:
        if args.group_fit:
            print("Note: --group-fit applies to plain plots only; ignored with --control-for.",
                  file=sys.stderr)
        c_kind, c_payload = parse_col_spec(args.control_for, available_columns)
        c_vals, c_label, c_langs = extract_values(c_kind, c_payload, rows)
        c_by_lang = dict(zip(c_langs, c_vals))
        common_langs = [l for l in common_langs if l in c_by_lang]

        ignore_mask = outlier_mask(common_langs, ignored_codes)
        n_used = len(common_langs) - sum(ignore_mask)
        if n_used < 4:
            print(f"Not enough languages with all three fields present ({n_used}, after "
                  f"--ignore-outliers) to compute a partial correlation.", file=sys.stderr)
            sys.exit(1)

        x_final = [x_by_lang[l] for l in common_langs]
        y_final = [y_by_lang[l] for l in common_langs]
        c_final = [c_by_lang[l] for l in common_langs]
        coloring = build_coloring(common_langs, langs_csv_data, args.color_by)

        auto_name = (f"{stem}_partial_{x_label.replace('/','-')}_vs_{y_label.replace('/','-')}"
                     f"_ctrl_{c_label.replace('/','-')}{color_tag}.png")
        out_path = args.out or os.path.join(out_dir, auto_name)
        print(f"output path: {out_path}")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        r_raw, p_raw, r_partial, p_partial = plot_partial(
            x_final, y_final, c_final, x_label, y_label, c_label, common_langs, subtitle, out_path,
            coloring=coloring, ignore_mask=ignore_mask)
        print(f"saved {out_path}")
        print(f"raw r({x_label}, {y_label}) = {r_raw:.4f}  r^2 = {r_raw ** 2:.4f}  p = {p_str(p_raw)}  (n={n_used})")
        print(f"partial r({x_label}, {y_label} | {c_label}) = {r_partial:.4f}  r^2 = {r_partial ** 2:.4f}  p = {p_str(p_partial)}  (n={n_used})")
    else:
        ignore_mask = outlier_mask(common_langs, ignored_codes)
        n_used = len(common_langs) - sum(ignore_mask)
        if n_used < 2:
            print(f"Not enough languages with both fields present ({n_used}, after "
                  f"--ignore-outliers) to plot.", file=sys.stderr)
            sys.exit(1)
        x_final = [x_by_lang[l] for l in common_langs]
        y_final = [y_by_lang[l] for l in common_langs]
        coloring = build_coloring(common_langs, langs_csv_data, args.color_by)

        # Statistics below (breakdown, group fit) use the included languages only.
        inc = [i for i, ig in enumerate(ignore_mask) if not ig]
        x_inc = [x_final[i] for i in inc]
        y_inc = [y_final[i] for i in inc]
        langs_inc = [common_langs[i] for i in inc]

        groups, order, note = (None, None, None)
        if langs_csv_data and args.group_by != "none":
            groups, order, note = assign_groups(langs_inc, langs_csv_data, args.group_by)
            missing = [CODE_TO_LANG_NAME.get(l, l) for l, g in zip(langs_inc, groups) if g is None]
            if missing:
                print(f"Note: no {args.group_by} group for {', '.join(missing)} -- counted in the "
                      f"joint line, left out of the group statistics.", file=sys.stderr)

        breakdown = None
        if args.breakdown != "off" and groups is not None:
            breakdown = residual_breakdown(x_inc, y_inc, langs_inc, groups, order)
        gf = None
        if args.group_fit:
            if groups is None:
                print("--group-fit needs --langs-csv with the grouping column.", file=sys.stderr)
                sys.exit(1)
            gf = group_fit(x_inc, y_inc, langs_inc, groups, order)
            if gf is None:
                print("Note: --group-fit skipped -- a group has fewer than 3 languages.",
                      file=sys.stderr)

        group_tag = f"_groupfit-{args.group_by}" if gf is not None else ""
        auto_name = (f"{stem}_corr_{x_label.replace('/','-')}_vs_{y_label.replace('/','-')}"
                     f"{color_tag}{group_tag}.png")
        out_path = args.out or os.path.join(out_dir, auto_name)
        print(f"output path: {out_path}")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        r, p = plot_plain(x_final, y_final, x_label, y_label, common_langs, subtitle, out_path,
                          coloring=coloring,
                          breakdown=breakdown if args.breakdown == "legend" else None,
                          groupfit=gf, ignore_mask=ignore_mask)
        print(f"saved {out_path}")
        print(f"r({x_label}, {y_label}) = {r:.4f}  r^2 = {r * r:.4f}  p = {p_str(p)}  (n={n_used})")
        if breakdown is not None:
            print_breakdown(breakdown, oos=args.oos, note=note)
        if gf is not None:
            print_group_fit(gf, note=note)


if __name__ == "__main__":
    main()