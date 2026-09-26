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
        codepoints, cp_per_baseline, spread, spread_customenc
      - the exact header text from the file (case-insensitive): Premium,
        PPS, BPP, EntropyMean, EntropyVar, EntropySkew, EntropyKurtosis,
        EntropyAutocorr1, EntropyVolatility
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
    over ALL languages BEFORE filtering, so a language keeps the same rank
    it has in the unfiltered plots. Log10 columns are per-language and
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
    (Georgian) are drawn as split markers, left half in the colour of
    the lower end and right half in the colour of the upper end. The regression line and r/p values are unchanged --
    they are still computed over all plotted languages together; colour
    is purely visual. Can be combined freely with the language filters
    and with --control-for (in the 6-panel walkthrough the category
    legend is drawn once, below the grid). A '_color-<choice>' tag is
    appended to the auto-derived output filename.

RESIDUAL BREAKDOWN (--breakdown)
    For plain (non-partial) plots containing both single-byte and
    multi-byte languages, the residuals of the regression line are
    broken down by group -- single-byte = Approx_Bytes_Per_Char "1" or
    "1-2" (dominant length 1, i.e. spread undefined), multi-byte = "2",
    "2-3", "3":
      - each group's share of the unexplained variance (sum of squared
        residuals), split into
          offset  = n_g * mean_residual_g^2   (group systematically
                    above/below the line)
          scatter = sum (residual - mean_residual_g)^2
      - per-group mean residual, RMSE and largest deviation
    Printed by default; --breakdown legend also summarises it inside the
    plot legend, --breakdown off disables it. Skipped automatically if
    either group has fewer than 3 languages.

    Out-of-sample check (--oos): a line fitted to multi-byte languages
    ONLY, and how well it predicts the single-byte languages. Only
    needed when there is NO group offset -- then single-byte languages
    lying close to the joint line could be an artefact of them pulling
    the line towards themselves. With a clear offset that pull works
    against the result (it shrinks the offset), so the check adds
    nothing. By default ('auto') it is printed only when the group
    offset is below OOS_OFFSET_THRESHOLD of the unexplained variance;
    'always' / 'never' override. Never shown in the plot legend.

USAGE
    # plain correlation, literal file path (--threshold still required,
    # just unused when premium_file is a literal path)
    python3 plot_correlation.py premiums_sorted.txt --threshold global --cols mean,premium

    # shorthand run name instead of the full path -- resolves the step
    # folder and exact calibrated threshold automatically via RUN_INFO
    python3 plot_correlation.py balanced --threshold global --cols mean,premium
    python3 plot_correlation.py imbalanced --char-level --threshold mono --cols mean,premium
    python3 plot_correlation.py balanced-custom --threshold global --cols density,premium

    # ratio as one side
    python3 plot_correlation.py balanced --threshold global --cols mean/variance,premium

    # log10 of a skewed column instead of its rank -- see add_log_columns
    # and the training_data_log alias, or wrap ANY column/ratio directly:
    python3 plot_correlation.py imbalanced --threshold global --cols log(imbalanced_allocation_bytes),mean

    # entropy spread (see byte_position_stats.py) instead of autocorrelation --
    # merged in from byte_position_stats.json by default, see --spread-json
    python3 plot_correlation.py balanced --threshold global --cols spread,premium

    # partial correlation, controlling for a third column -- shows the
    # full 6-panel step-by-step walkthrough (same construction used
    # earlier for spread-vs-premium controlling for budget): X~control,
    # residual; Y~control, residual; then residual-vs-residual, with the
    # raw (uncontrolled) comparison shown separately for reference.
    python3 plot_correlation.py balanced --threshold global --cols mean,premium --control-for variance

    # restrict to alphabetic scripts that use 1 byte per character
    # (Latin-script languages except Vietnamese)
    python3 plot_correlation.py imbalanced --threshold global \
        --cols training_data_log,mean --script-type alphabetic --bytes-per-char 1

    # all multi-byte languages (2, 2-3 and 3 bytes per character)
    python3 plot_correlation.py imbalanced --threshold mono \
        --cols spread,premium --bytes-per-char 2,2-3,3

    # all abugidas and syllabaries, any byte length
    python3 plot_correlation.py imbalanced --threshold mono \
        --cols mean,premium --script-type abugida,syllabary

    # colour points by script type / by bytes per character
    python3 plot_correlation.py balanced --threshold global \
        --cols mean,premium --color-by script-type
    python3 plot_correlation.py imbalanced --threshold mono \
        --cols spread,premium --control-for training_data_log --color-by bytes-per-char

PARSING THE INPUT FILE
    That file format is FIXED-WIDTH (built by left-justifying every
    value to a per-column width, then right-stripping each row) -- NOT
    safely splittable by naive whitespace-splitting, because a blank
    entropy-stat field (a language whose curve had too few points, or a
    zero-variance signal) still occupies its column's full width as
    spaces, and if that blank happens to be the LAST field on a given
    row, that row's line ends up literally shorter than others (nothing
    left to rstrip). This script instead locates each header's start
    column IN THE HEADER LINE ITSELF (searching for each known header
    name, in order, so it works regardless of which columns are present
    in this particular file) and slices every data row at those exact
    character offsets -- correct regardless of where blanks fall.

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
    # counts) -- raw allocation bytes span ~5 orders of magnitude
    # (English ~9.7B down to the smallest language ~51K), which would
    # completely dominate/skew any correlation plot if used directly.
    # Rank 1 = most training data. Use the literal CSV column names
    # (imbalanced_allocation_bytes / balanced_allocation_bytes) if you
    # actually want the raw byte values instead.
    "training_data": "imbalanced_allocation_bytes_rank",
    "training_data_rank": "imbalanced_allocation_bytes_rank",
    "training_data_imbalanced": "imbalanced_allocation_bytes_rank",
    "imbalanced_rank": "imbalanced_allocation_bytes_rank",
    "training_data_log": "imbalanced_allocation_bytes_log10",
    "training_data_log10": "imbalanced_allocation_bytes_log10",
    "imbalanced_log": "imbalanced_allocation_bytes_log10",
    "imbalanced_bytes": "imbalanced_allocation_bytes",  # raw bytes, unchanged
    "imbalanced_allocation_bytes": "imbalanced_allocation_bytes",  # raw bytes, unchanged
    "training_data_balanced": "balanced_allocation_bytes_rank",
    "balanced_rank": "balanced_allocation_bytes_rank",
    "training_data_balanced_log": "balanced_allocation_bytes_log10",
    "balanced_log": "balanced_allocation_bytes_log10",
    "balanced_bytes": "balanced_allocation_bytes",  # raw bytes, unchanged
    "balanced_allocation_bytes": "balanced_allocation_bytes",  # raw bytes, unchanged
    "documents": "documents", "n_documents": "documents",
    "utf8_bytes": "utf8_bytes",
    "ratio_vs_english": "ratio_vs_english",
    "ratio_vs_english_imbalanced": "ratio_vs_english_imbalanced",
    "bytes_per_char": "Approx_Bytes_Per_Char", "approx_bytes_per_char": "Approx_Bytes_Per_Char",

    # --- from --density-json (default: char_density.json) ---
    "density": "density_index", "density_index": "density_index",
    "codepoints": "n_codepoints", "n_codepoints": "n_codepoints",
    "cp_per_baseline": "codepoints_per_baseline_codepoint",
    "codepoints_per_baseline_codepoint": "codepoints_per_baseline_codepoint",
    "customenc_bytes": "n_bytes_total",

    # --- from --spread-json / --spread-customenc-json (default:
    # byte_position_stats.json / byte_position_stats_customenc.json,
    # produced by byte_position_stats.py) --- normalized [0, 1] balance
    # of a language's per-character identity entropy across its
    # dominant byte-length's positions (0 = concentrated in one byte,
    # 1 = perfectly even). None/blank for languages whose dominant
    # length is 1 byte -- see spread_score() in byte_position_stats.py.
    "spread": "spread", "entropy_spread": "spread",
    "spread_customenc": "spread_customenc", "custom_spread": "spread_customenc",
    "spread_custom_encoding": "spread_customenc",
    # main_length_entropy_sum companions, merged in alongside spread --
    # total identity-entropy (bits) of the dominant byte-length, useful
    # to control for "how much information" while spread captures "how
    # evenly spread out".
    "spread_sum": "main_length_entropy_sum", "entropy_sum": "main_length_entropy_sum",
    "spread_customenc_sum": "main_length_entropy_sum_customenc",
}

# Which literal column names get pulled in from each optional external
# source, if that source is loaded and has a matching language row --
# see load_langs_csv / load_density_json / merge_external_columns.
LANGS_CSV_COLUMNS = [
    "ratio_vs_english", "documents", "utf8_bytes", "balanced_allocation_bytes",
    "ratio_vs_english_imbalanced", "imbalanced_allocation_bytes", "Approx_Bytes_Per_Char",
]
DENSITY_JSON_COLUMNS = [
    "n_bytes_total", "n_codepoints", "codepoints_per_baseline_codepoint", "density_index",
]
# Fields pulled per-language out of a byte_position_stats(.py)-style JSON
# (keyed by language code at the top level, e.g. {"eng_Latn": {...}, ...}).
# Mapped to the row-column names spread/spread_customenc resolve to above.
SPREAD_JSON_FIELDS = {
    "spread": "spread",
    "main_length_entropy_sum": "main_length_entropy_sum",
}
SPREAD_CUSTOMENC_JSON_FIELDS = {
    "spread": "spread_customenc",
    "main_length_entropy_sum": "main_length_entropy_sum_customenc",
}

# --- COLOUR CODING (--color-by) ---
# CLI choice -> (langs-csv column it reads, legend title).
COLOR_BY_OPTIONS = {
    "none": None,
    "script-type": ("Script_Type", "Script type"),
    "bytes-per-char": ("Approx_Bytes_Per_Char", "Bytes per char"),
}
# Fixed category -> colour maps, so a category keeps the same colour
# across every figure. Dict order = legend order. Categories not listed
# here (e.g. a new value added to the CSV later) fall back to
# UNKNOWN_CATEGORY_COLOR and are still shown in the legend by name.
CATEGORY_PALETTES = {
    "Script_Type": {
        "Alphabetic":    "#4C72B0",  # blue
        "Abjad":         "#DD8452",  # orange
        "Abugida":       "#55A868",  # green
        "Syllabary":     "#C44E52",  # red
        "Logosyllabary": "#8172B3",  # purple
        "Logographic":   "#937860",  # brown
    },
    # Blue (1 byte), green (2 bytes), yellow-beige (3 bytes) -- matched
    # in darkness/saturation (seaborn "deep" palette).
    # Range values like "1-2" / "2-3" are NOT listed: any "A-B" value
    # whose two ends are both in the palette is drawn as a split marker,
    # left half in A's colour, right half in B's -- see build_coloring.
    "Approx_Bytes_Per_Char": {
        "1": "#4C72B0",
        "2": "#55A868",
        "3": "#CCB974",
    },
}
UNKNOWN_CATEGORY_COLOR = "#BBBBBB"
DEFAULT_POINT_COLOR = "#4C72B0"

# Bytes-per-char values counted as single-byte for the residual
# breakdown (dominant character length 1 -- matches spread being
# undefined for these languages). Everything else is multi-byte.
SINGLE_BYTE_VALUES = {"1", "1-2"}

# --oos auto: run the out-of-sample check only if the group offset is
# below this share of the unexplained variance (see module docstring).
OOS_OFFSET_THRESHOLD = 0.10

# Readable names for plot titles and axis labels. Filenames keep the
# raw column names, so existing output paths don't change.
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
}

# Font sizes (points). Raised so language names and numbers stay
# readable when a figure is scaled down to column width in a paper.
FS_TITLE = 16
FS_SUBTITLE = 12
FS_AXIS_LABEL = 14
FS_TICKS = 12
FS_POINT_LABEL = 11
FS_LEGEND = 11
FS_LEGEND_TITLE = 11.5
POINT_SIZE = 140  # scatter marker area

# Legend position inside the axes (matplotlib loc string). Default
# bottom left; overridable with --legend-loc when a warning shows it
# covers data (see warn_legend_overlap).
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
    """Readable description of the active language filter, e.g.
    'multi-byte languages' -- '' if no filter is active."""
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
    """Readable run description from a premiums file stem, e.g.
    'Balanced_raw_monotonicity_t_0.6664_premiums_sorted' ->
    'Balanced, monotonicity threshold (t = 0.6664)'. Falls back to the
    stem itself if it doesn't follow the usual naming pattern."""
    m = re.match(r"^(?P<run>.+?)_raw_(?P<case>entropy|monotonicity)_t_(?P<t>[\d.]+)", stem)
    if not m:
        return stem
    strategy = "global threshold" if m.group("case") == "entropy" else "monotonicity threshold"
    desc = f"{m.group('run')}, {strategy} (t = {m.group('t')})"
    if char_level:
        desc += ", char-level"
    return desc

# Shorthand run names -> canonical folder/filename-prefix, matching
# results_to_txt_premiums.py's RUN_NAME_ALIASES output. Lets you write
# "balanced" instead of the full premiums_sorted.txt path -- see
# resolve_premium_path. Known checkpoint steps and calibrated thresholds
# for each run, both granularities -- update here if a run is
# recalibrated or a new one is added; everything else derives from this.
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
    """If spec is a recognized shorthand (balanced / imbalanced /
    balanced-custom, case-insensitive, hyphen or underscore), builds and
    returns the full premiums_sorted.txt path for it using RUN_INFO.
    Returns None if spec isn't a recognized shorthand -- caller should
    then treat spec as a literal path, unchanged (so full paths still
    work exactly as before)."""
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
    row data (premiums file columns, plus whatever was merged in from
    --langs-csv / --density-json / --spread-json / --spread-customenc-json).
    Raises a clear error listing what IS available if it can't be resolved."""
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
            f"{sorted(set(COLUMN_ALIASES.keys()))}. If you expected a "
            f"--langs-csv, --density-json, --spread-json, or "
            f"--spread-customenc-json column, check those files were found "
            f"and contain a matching language_code / top-level key."
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
    """Splits a comma-separated CLI filter string ("1,1-2") into a list
    of stripped, non-empty values. Returns None if raw is None/empty
    (i.e. the filter is not active)."""
    if raw is None:
        return None
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    return vals or None


def select_languages(langs_csv_data, script_types=None, bytes_per_char=None):
    """Returns the set of language codes in langs_csv_data whose
    'Script_Type' is in script_types (case-insensitive) AND whose
    'Approx_Bytes_Per_Char' is in bytes_per_char (exact string match,
    e.g. "1", "1-2", "2", "2-3", "3"). A filter that is None/empty is
    not applied. Raises ValueError, listing the values that DO exist, if
    a requested value doesn't occur anywhere in the CSV (catches typos
    like "abugida " vs "abjad" instead of silently returning nothing)."""
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


def make_filter_tag(script_types, bytes_per_char):
    """Short, filename-safe description of the active language filters,
    e.g. '_script-alphabetic_bytes-1' -- '' if no filter is active."""
    tag = ""
    if script_types:
        tag += "_script-" + "+".join(sorted(v.lower() for v in script_types))
    if bytes_per_char:
        tag += "_bytes-" + "+".join(sorted(bytes_per_char))
    return tag


def build_coloring(langs, langs_csv_data, color_by):
    """Returns None if color_by is 'none'; otherwise a dict describing
    the per-point colouring for the given (already aligned) list of
    language codes:
        {"categories": [cat per point], "colors": [hex per point],
         "title": legend title, "order": [categories in legend order]}
    Category values are matched case-insensitively against
    CATEGORY_PALETTES and reported with the palette's canonical
    spelling; values missing from the palette get UNKNOWN_CATEGORY_COLOR
    (languages with no CSV row / blank value are labelled 'unknown')."""
    spec = COLOR_BY_OPTIONS.get(color_by)
    if spec is None:
        return None
    csv_col, title = spec
    palette = CATEGORY_PALETTES[csv_col]
    canon_by_lower = {k.lower(): k for k in palette}

    categories, colors = [], []
    split_first = {}  # split category -> its left-hand palette key, for legend ordering
    for lang in langs:
        raw = (langs_csv_data.get(lang, {}).get(csv_col) or "").strip()
        canon = canon_by_lower.get(raw.lower())
        m_range = re.match(r"^\s*(.+?)\s*-\s*(.+?)\s*$", raw)
        if canon is not None:
            categories.append(canon)
            colors.append(palette[canon])
        elif (m_range and m_range.group(1).lower() in canon_by_lower
              and m_range.group(2).lower() in canon_by_lower):
            # Range value, e.g. "1-2": a (left, right) colour pair ->
            # drawn as a half-and-half marker.
            lo = canon_by_lower[m_range.group(1).lower()]
            hi = canon_by_lower[m_range.group(2).lower()]
            cat = f"{lo}-{hi}"
            categories.append(cat)
            colors.append((palette[lo], palette[hi]))
            split_first[cat] = lo
        else:
            categories.append(raw or "unknown")
            colors.append(UNKNOWN_CATEGORY_COLOR)

    # Legend order: palette order, each split category placed right
    # after its left-hand end (1, 1-2, 2, 2-3, 3), then anything unknown.
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
    """One round-marker legend entry per category present, labelled
    with its point count, in the coloring's legend order. Range
    categories get a half-and-half marker matching the plot."""
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


def load_density_json(path):
    """Returns {language_code: {stat_name: value}} as already produced
    by char_density.py."""
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
    """Mutates each row dict in place, adding whichever of
    LANGS_CSV_COLUMNS / DENSITY_JSON_COLUMNS / SPREAD_JSON_FIELDS /
    SPREAD_CUSTOMENC_JSON_FIELDS are available for that row's language.
    Returns the set of column names actually added (i.e. that were found
    for at least one language, with a non-null value), for
    available_columns."""
    added = set()
    spread_data = spread_data or {}
    spread_customenc_data = spread_customenc_data or {}
    for row in rows:
        lang = row.get("Language", "")
        if lang in langs_csv_data:
            src = langs_csv_data[lang]
            for col in LANGS_CSV_COLUMNS:
                if col in src and src[col] != "":
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


# Raw byte-count columns that get a companion "<column>_rank" version
# computed automatically -- see add_rank_columns.
RANK_SOURCE_COLUMNS = ["imbalanced_allocation_bytes", "balanced_allocation_bytes"]


def add_rank_columns(rows, source_columns=RANK_SOURCE_COLUMNS):
    """For each column name in source_columns, adds a companion
    "<column>_rank" field to every row that has a numeric value for it:
    rank 1 = the LARGEST value (e.g. most training data), matching this
    project's established "#1 = most bytes" convention. This is what
    training_data / training_data_balanced resolve to by default (see
    COLUMN_ALIASES) -- raw byte counts span several orders of magnitude
    across languages, which would dominate/skew a correlation plot far
    more than reflecting anything meaningful about the relationship
    being tested. Returns the set of rank column names actually added."""
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
        pairs.sort(key=lambda p: -p[1])  # descending: biggest value = rank 1
        rank_by_lang = {lang: i + 1 for i, (lang, _) in enumerate(pairs)}
        rank_col = f"{col}_rank"
        for row in rows:
            lang = row.get("Language", "")
            if lang in rank_by_lang:
                row[rank_col] = str(rank_by_lang[lang])
                added.add(rank_col)
    return added


def add_log_columns(rows, source_columns=RANK_SOURCE_COLUMNS):
    """For each column name in source_columns, adds a companion
    "<column>_log10" field (log base 10 of the raw value) to every row
    with a positive numeric value for it. Unlike the _rank companions,
    this PRESERVES relative magnitude -- appropriate when the underlying
    values plausibly follow a geometric/power-law spread (as a
    deliberately skewed training-data allocation typically does) and you
    want to test whether each order-of-magnitude change has a roughly
    constant effect, rather than only testing whether the relationship
    is monotonic (which is what the _rank version tests -- rank
    correlation is mathematically identical to Spearman's rho). Values
    <= 0 are skipped (log undefined) with a note printed once. Returns
    the set of log column names actually added."""
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
    """Returns (present_headers, rows) where rows is a list of
    {header: value_string} dicts, in file order (Language is a string,
    everything else is left as a string -- '' for a blank/missing
    field). See module docstring for why this uses header-position
    slicing rather than whitespace-splitting."""
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
            break  # blank line marks end of data rows
        if line.lstrip().startswith("#"):
            break
        row = {}
        for h, (start, end) in zip(present_headers, bounds):
            raw = line[start:end] if end is not None else line[start:]
            row[h] = raw.strip()
        rows.append(row)

    return present_headers, rows


def parse_col_spec(spec, available_columns):
    """Parses one --cols/--control-for entry. Supports:
      - a plain column name/alias
      - a ratio A/B
      - log(...) wrapping either of the above, e.g. log(imbalanced_allocation_bytes)
        or log(mean/variance) -- computes log10 of the underlying values
        (see add_log_columns for the precomputed training_data_log
        shortcut, which is usually what you want instead of wrapping the
        rank-based training_data alias in log() -- log of an already-
        ordinal rank isn't a meaningful quantity).
    Returns (kind, payload): ("plain", column_name), ("ratio",
    (numerator_name, denominator_name)), or ("log", (inner_kind, inner_payload)).
    """
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
    """Returns (values, display_label, langs) -- values as a list of
    floats, skipping any row where the needed field(s) are blank/missing,
    non-numeric, or (for log) non-positive. display_label is a human
    string for axis labels."""
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
    """Pearson r and its two-tailed p-value. extra_df_used=1 for a
    partial correlation computed from residuals (one degree of freedom
    already spent fitting the control regression) -- see the spread/
    premium/budget analysis earlier in this project for why this
    correction matters."""
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
    if p != p:  # NaN check
        return "n/a"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def residual_breakdown(x, y, langs, langs_csv_data, min_per_group=3):
    """Breaks the residuals of the least-squares line y ~ x down by
    single- vs. multi-byte languages (see SINGLE_BYTE_VALUES). Returns
    None if either group has fewer than min_per_group languages (e.g.
    a filtered plot); otherwise a dict with, per group, n, mean
    residual, RMSE, largest deviation, and share of the unexplained
    variance split into offset and scatter -- plus an out-of-sample
    check (line fitted to multi-byte languages only, applied to the
    single-byte ones). Languages with no Approx_Bytes_Per_Char value
    count towards the fit but not towards either group."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bpc = [(langs_csv_data.get(l, {}).get("Approx_Bytes_Per_Char") or "").strip() for l in langs]
    single = np.array([b in SINGLE_BYTE_VALUES for b in bpc])
    multi = np.array([b != "" and b not in SINGLE_BYTE_VALUES for b in bpc])
    if single.sum() < min_per_group or multi.sum() < min_per_group:
        return None

    coefs = np.polyfit(x, y, 1)
    resid = y - np.polyval(coefs, x)
    total_ss = float((resid ** 2).sum())
    if total_ss <= 0:
        return None

    def group_stats(mask):
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

    out = {
        "slope_all": float(coefs[0]), "intercept_all": float(coefs[1]),
        "single": group_stats(single), "multi": group_stats(multi),
    }
    out["offset_share"] = out["single"]["offset_share"] + out["multi"]["offset_share"]

    # Out-of-sample: fit on multi-byte languages alone, predict single-byte.
    coefs_mb = np.polyfit(x[multi], y[multi], 1)
    e_oos = y[single] - np.polyval(coefs_mb, x[single])
    out["oos"] = {
        "slope_multi": float(coefs_mb[0]), "intercept_multi": float(coefs_mb[1]),
        "mean_resid": float(e_oos.mean()),
        "rmse": float(np.sqrt((e_oos ** 2).mean())),
        "max_abs": float(np.abs(e_oos).max()),
    }
    return out


def print_breakdown(bd, oos="auto"):
    """Full residual breakdown as a readable block on stdout. oos:
    'auto' / 'always' / 'never' -- whether to include the out-of-sample
    check (see module docstring)."""
    name = lambda c: CODE_TO_LANG_NAME.get(c, c)
    print("\nResidual breakdown (line fitted to all plotted languages, "
          f"slope {bd['slope_all']:.3f}, intercept {bd['intercept_all']:.3f}):")
    print(f"  {'group':<12}{'n':>3}  {'mean res.':>9}  {'RMSE':>6}  {'max |res|':>18}  "
          f"{'share':>6}  {'offset':>6}  {'scatter':>7}")
    for key, label in (("single", "single-byte"), ("multi", "multi-byte")):
        g = bd[key]
        max_str = f"{g['max_abs']:.3f} ({name(g['max_lang'])})"
        print(f"  {label:<12}{g['n']:>3}  {g['mean_resid']:>+9.3f}  {g['rmse']:>6.3f}  {max_str:>18}  "
              f"{g['share']:>6.1%}  {g['offset_share']:>6.1%}  {g['scatter_share']:>7.1%}")
    print(f"  total group offset: {bd['offset_share']:.1%} of the unexplained variance")
    run_oos = oos == "always" or (oos == "auto" and bd["offset_share"] < OOS_OFFSET_THRESHOLD)
    if not run_oos:
        if oos == "auto":
            print(f"  (out-of-sample check skipped: group offset >= {OOS_OFFSET_THRESHOLD:.0%}; "
                  f"use --oos always to run it anyway)")
        print("  (single-byte = Approx_Bytes_Per_Char 1 or 1-2; multi-byte = 2, 2-3, 3)")
        return
    o = bd["oos"]
    print(f"  Out-of-sample: line fitted to multi-byte languages only "
          f"(slope {o['slope_multi']:.3f}, intercept {o['intercept_multi']:.3f})")
    print(f"    predicting single-byte languages: mean res. {o['mean_resid']:+.3f}, "
          f"RMSE {o['rmse']:.3f}, max |res| {o['max_abs']:.3f}")
    print("  (single-byte = Approx_Bytes_Per_Char 1 or 1-2; multi-byte = 2, 2-3, 3)")


def _signed(v, digits=2):
    """'+0.12' / '\u22120.12', and '0.00' instead of '-0.00'."""
    if round(v, digits) == 0:
        return f"{0:.{digits}f}"
    return f"{v:+.{digits}f}".replace("-", "\u2212")


def breakdown_legend_lines(bd):
    """Compact summary of the breakdown for the in-plot legend."""
    s, m = bd["single"], bd["multi"]
    return [
        "Unexplained variance:",
        f"   single-byte {s['share']:.0%} (RMSE {s['rmse']:.2f}, mean {_signed(s['mean_resid'])})",
        f"   multi-byte {m['share']:.0%} (RMSE {m['rmse']:.2f}, mean {_signed(m['mean_resid'])})",
        f"   group offset {bd['offset_share']:.0%}",
    ]


def label_points(ax, x, y, langs, avoid=None):
    """Language-name labels next to each point. Uses adjustText to
    avoid overlaps (with each other, the points, and the artists in
    'avoid', e.g. the legend) if it's installed (pip install
    adjustText); otherwise falls back to stacking labels of points that
    share an x value. Returns the list of label artists."""
    names = [CODE_TO_LANG_NAME.get(l, l) for l in langs] if langs else []
    if not names:
        return []
    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None

    if adjust_text is not None:
        texts = [ax.text(xv, yv, nm, fontsize=FS_POINT_LABEL, color="#222222", zorder=4)
                 for xv, yv, nm in zip(x, y, names)]
        arrows = dict(arrowstyle="-", color="#999999", lw=0.6)
        objs = [a for a in (avoid or []) if a is not None]
        try:
            # expand: keep labels clear of the (fairly large) markers;
            # objects: push labels out from under the legend
            adjust_text(texts, x=list(x), y=list(y), ax=ax, expand=(1.4, 1.8),
                        objects=objs or None, arrowprops=arrows)
        except TypeError:  # older adjustText versions: no 'expand'/'objects'
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
                xytext=(8, 6 + rank * 14), fontsize=FS_POINT_LABEL, color="#222222",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5, shrinkA=0, shrinkB=4)))
    return texts


def scatter_with_fit(ax, x, y, langs, xlabel, ylabel, title, extra_df_used=0,
                     coloring=None, show_category_legend=True, subtitle=None,
                     breakdown=None, invert_x=False, invert_y=False, compact=False):
    """coloring: None (single colour) or the dict from build_coloring,
    aligned point-for-point with x/y/langs. show_category_legend=False
    colours the points but leaves the category legend to the caller
    (used by the 6-panel grid, which draws one shared legend).
    breakdown: optional dict from residual_breakdown, summarised in the
    legend below the fit line. xlabel/ylabel/title are used as given
    (pass readable labels, see pretty_label). compact=True uses smaller
    fonts, for the multi-panel grid."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r, p = pearson_r_p(x, y, extra_df_used=extra_df_used)
    scale = 0.85 if compact else 1.0

    if coloring is None:
        ax.scatter(x, y, c=DEFAULT_POINT_COLOR, s=POINT_SIZE, edgecolors="#333333",
                   linewidths=0.8, zorder=3)
    else:
        cols = coloring["colors"]
        solid = [i for i, col in enumerate(cols) if not is_split_color(col)]
        if solid:
            ax.scatter(x[solid], y[solid], c=[cols[i] for i in solid], s=POINT_SIZE,
                       edgecolors="#333333", linewidths=0.8, zorder=3)
        # scatter() can't do two-colour markers, so range categories
        # (e.g. "1-2") are drawn one by one with plot(): left half in
        # the first colour, right half in the second. markersize=sqrt(s)
        # keeps them the same size as the scatter points.
        for i, col in enumerate(cols):
            if is_split_color(col):
                ax.plot(x[i], y[i], marker="o", linestyle="", markersize=math.sqrt(POINT_SIZE),
                        fillstyle="left", markerfacecolor=col[0], markerfacecoloralt=col[1],
                        markeredgecolor="#333333", markeredgewidth=0.8, zorder=3)

    # One combined legend: category markers (if any), the fit line, and
    # the residual breakdown (if any) -- a single box, so separate
    # legends can't overlap each other.
    handles = []
    if coloring is not None and show_category_legend:
        handles += category_legend_handles(coloring)
    fit_line = None
    if len(x) >= 2:
        coefs = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        fit_handle, = ax.plot(x_line, np.polyval(coefs, x_line), linestyle="--",
                               color="red", linewidth=1.8, zorder=2,
                               label=f"Linear fit: r = {0 if round(r, 2) == 0 else r:.2f}, "
                                     f"r$^2$ = {r * r:.2f}, p = {p_str(p)}")
        handles.append(fit_handle)
        fit_line = fit_handle
    if breakdown is not None:
        handles += [Line2D([], [], linestyle="none", marker="none", label=line)
                    for line in breakdown_legend_lines(breakdown)]
    # Titles, labels and axis orientation first, so the legend and the
    # point labels below are placed on the final layout.
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

    # Autocorrelation reads more naturally decreasing left-to-right
    # (more positive/persistent on the left, more negative/choppy on
    # the right) rather than matplotlib's default increasing order.
    # Decided by the caller from the RAW column labels (see
    # involves_autocorr), since the displayed labels are prettified.
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()

    # Legend always inside the axes, bottom left. It may cover data --
    # warn_legend_overlap() reports that after the final layout.
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
        legend.set_zorder(5)  # drawn above points and labels

    texts = label_points(ax, x, y, langs, avoid=[legend])

    # Kept for warn_legend_overlap(), which runs after tight_layout().
    ax._overlap_check = dict(legend=legend, x=x, y=y, langs=list(langs or []),
                             texts=texts, fit_line=fit_line, title=title)

    return r, p


def warn_legend_overlap(fig, ax, where=""):
    """Prints a warning to stderr if the legend of ax (placed by
    scatter_with_fit) covers any data point, point label, or part of
    the regression line. Call after the final layout (tight_layout)."""
    info = getattr(ax, "_overlap_check", None)
    if not info or info["legend"] is None:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = info["legend"].get_window_extent(renderer)
    name = lambda c: CODE_TO_LANG_NAME.get(c, c)

    # points: circle of radius ~ marker size overlapping the box
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
    """True if a raw column label (plain, ratio, log or residual)
    involves lag-1 autocorrelation -- see the axis inversion note in
    scatter_with_fit."""
    return "EntropyAutocorr1" in raw_label


def plot_plain(x, y, x_label, y_label, langs, subtitle, out_path, coloring=None,
               breakdown=None):
    """x_label/y_label are the RAW column labels (prettified here)."""
    fig, ax = plt.subplots(figsize=(10.5, 8))
    r, p = scatter_with_fit(ax, x, y, langs, pretty_label(x_label), pretty_label(y_label),
                             f"{pretty_label(y_label)} vs. {pretty_label(x_label)}",
                             coloring=coloring, subtitle=subtitle, breakdown=breakdown,
                             invert_x=involves_autocorr(x_label),
                             invert_y=involves_autocorr(y_label))
    plt.tight_layout()
    warn_legend_overlap(fig, ax, os.path.basename(out_path))
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return r, p


def plot_partial(x, y, ctrl, x_label, y_label, ctrl_label, langs, subtitle, out_path,
                 out_path_single=None, coloring=None):
    """x_label/y_label/ctrl_label are the RAW column labels."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.asarray(ctrl, dtype=float)
    px, py, pc = pretty_label(x_label), pretty_label(y_label), pretty_label(ctrl_label)
    inv_x, inv_y, inv_c = (involves_autocorr(l) for l in (x_label, y_label, ctrl_label))

    # 1. Compute residuals
    x_fit = np.polyfit(c, x, 1)
    x_resid = x - np.polyval(x_fit, c)

    y_fit = np.polyfit(c, y, 1)
    y_resid = y - np.polyval(y_fit, c)

    # --- SAVE SEPARATE LEFTOVER VS LEFTOVER PLOT ---
    if out_path_single is None:
        # Default name if not provided: replaces .png with _partial_only.png
        base, ext = os.path.splitext(out_path)
        out_path_single = f"{base}_partial_only{ext}"

    fig_single, ax_single = plt.subplots(figsize=(10.5, 8))
    scatter_with_fit(
        ax_single, x_resid, y_resid, langs,
        pretty_label(f"{x_label} residual"), pretty_label(f"{y_label} residual"),
        f"{py} vs. {px}, controlling for {pc}",
        extra_df_used=1, coloring=coloring, subtitle=subtitle,
        invert_x=inv_x, invert_y=inv_y,
    )
    plt.tight_layout()
    warn_legend_overlap(fig_single, ax_single, os.path.basename(out_path_single))
    plt.savefig(out_path_single, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig_single)

    # --- BUILD 6-PANEL WALKTHROUGH GRID ---
    # Points are coloured in every panel, but the category legend is
    # drawn once for the whole figure (below the grid) instead of six times.
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    kw = dict(coloring=coloring, show_category_legend=False, compact=True)

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
    if coloring is not None:
        handles = category_legend_handles(coloring)
        fig.legend(handles=handles, title=coloring["title"], loc="upper center",
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
                              "balanced / imbalanced / balanced-custom (case-insensitive). "
                              "With a shorthand, --threshold is required and --char-level is "
                              "used to pick which granularity; the full path (step folder, "
                              "exact calibrated threshold value, etc.) is resolved automatically "
                              "from RUN_INFO. A literal path is used exactly as given, unchanged.")
    parser.add_argument("--threshold", choices=["mono", "global"], required=True,
                         help="Which case to use when premium_file is a shorthand run name -- "
                              "'global' = raw_entropy, 'mono' = raw_monotonicity. Required "
                              "always (even for a literal path, where it's simply unused) to "
                              "keep the interface consistent.")
    parser.add_argument("--char-level", action="store_true",
                         help="When premium_file is a shorthand run name, resolve to the "
                              "char_level/ version instead of the byte-level (t_anchor) one. "
                              "Ignored if premium_file is a literal path.")
    parser.add_argument("--cols", required=True,
                         help="Two comma-separated column specs, X,Y. Each can be a plain "
                              "column (name or alias) or a ratio A/B. E.g. mean,premium or "
                              "mean/variance,premium. Columns can come from the premiums file "
                              "itself, --langs-csv, --density-json, --spread-json, or "
                              "--spread-customenc-json -- see module docstring.")
    parser.add_argument("--control-for", default=None,
                         help="A third column spec (same rules as --cols) to control for -- "
                              "if given, computes a partial correlation and shows the full "
                              "6-panel step-by-step walkthrough instead of a plain scatter.")
    parser.add_argument("--script-type", default=None,
                         help="Restrict to languages whose 'Script_Type' in --langs-csv matches "
                              "one of these comma-separated values (case-insensitive): "
                              "Alphabetic, Abjad, Abugida, Syllabary, Logosyllabary, Logographic. "
                              "E.g. --script-type alphabetic  or  --script-type abugida,syllabary. "
                              "Combined with --bytes-per-char by AND. Requires --langs-csv.")
    parser.add_argument("--bytes-per-char", default=None,
                         help="Restrict to languages whose 'Approx_Bytes_Per_Char' in --langs-csv "
                              "matches one of these comma-separated values (exact): "
                              "1, 1-2, 2, 2-3, 3. E.g. --bytes-per-char 1  or  "
                              "--bytes-per-char 2,2-3,3. Combined with --script-type by AND. "
                              "Requires --langs-csv.")
    parser.add_argument("--color-by", default=None,
                         type=lambda s: s.strip().lower().replace("_", "-"),
                         choices=list(COLOR_BY_OPTIONS),
                         help="Colour-code points by a language property from --langs-csv: "
                              "'bytes-per-char' (default, Approx_Bytes_Per_Char), 'script-type' "
                              "(Script_Type), or 'none' (single colour). Colours are fixed per "
                              "category across all plots (see CATEGORY_PALETTES). If left at the "
                              "default and --langs-csv can't be loaded, falls back to 'none'.")
    parser.add_argument("--breakdown", default="print", choices=["print", "legend", "off"],
                         help="Residual breakdown by single- vs. multi-byte languages for plain "
                              "plots (see module docstring): 'print' (default) prints it, "
                              "'legend' also summarises it in the plot legend, 'off' disables it. "
                              "Skipped automatically if either group has < 3 languages.")
    parser.add_argument("--legend-loc", default="lower-left",
                         type=lambda v: v.strip().lower().replace("_", "-").replace(" ", "-"),
                         choices=["lower-left", "lower-right", "upper-left", "upper-right",
                                  "lower-center", "upper-center", "center-left", "center-right"],
                         help="Legend position inside the plot (default: lower-left), e.g. "
                              "--legend-loc lower-right (a quoted 'lower right' works too). A "
                              "warning is printed if the legend covers points, labels or the fit line.")
    parser.add_argument("--oos", default="auto", choices=["auto", "always", "never"],
                         help="Out-of-sample check in the printed breakdown (line fitted to "
                              "multi-byte languages only, applied to single-byte ones): 'auto' "
                              "(default) runs it only when the group offset is small, see "
                              "OOS_OFFSET_THRESHOLD; 'always' / 'never' override. Never shown "
                              "in the plot legend.")
    parser.add_argument("--langs-csv", default="training_setup/langs/langs_chosen.csv",
                         help="CSV with a 'language_code' column plus training-data/typology "
                              "columns (ratio_vs_english, documents, utf8_bytes, "
                              "balanced_allocation_bytes, imbalanced_allocation_bytes, "
                              "ratio_vs_english_imbalanced, Approx_Bytes_Per_Char), merged in by "
                              "language code. Silently skipped if not found -- only an error if "
                              "you then reference a column that would have come from it (or use "
                              "--script-type / --bytes-per-char / --color-by, which need "
                              "Script_Type / Approx_Bytes_Per_Char from it). Pass an empty "
                              "string to disable.")
    parser.add_argument("--density-json", default="char_density.json",
                         help="JSON from char_density.py ({lang_code: {n_bytes_total, "
                              "n_codepoints, codepoints_per_baseline_codepoint, density_index}}), "
                              "merged in by language code. Same not-found behavior as "
                              "--langs-csv. Pass an empty string to disable.")
    parser.add_argument("--spread-json", default="byte_position_stats.json",
                         help="JSON from byte_position_stats.py ({lang_code: {spread, "
                              "main_length_entropy_sum, ...}}), merged in by language code as "
                              "the 'spread' / 'spread_sum' columns (see COLUMN_ALIASES). Same "
                              "not-found behavior as --langs-csv: silently skipped if missing. "
                              "Pass an empty string to disable.")
    parser.add_argument("--spread-customenc-json", default="byte_position_stats_customenc.json",
                         help="Same as --spread-json, but for a custom (non-UTF-8) encoding run "
                              "of byte_position_stats.py (--fixed-length), merged in as the "
                              "'spread_customenc' / 'spread_customenc_sum' columns. Silently "
                              "skipped if not found. Pass an empty string to disable.")
    parser.add_argument(
                            "--fill-spread-nulls",
                            action="store_true",
                            help="Replace null spread values (e.g. for 1-byte languages) with 1.0 to include all languages in charts.",
                        )
    parser.add_argument("--out", default=None, help="Output PNG path (overrides --out-dir entirely -- used exactly as given)")
    parser.add_argument("--out-dir", default="correlation_plots",
                         help="Directory the auto-derived output filename is saved into (default: correlation_plots/). "
                              "Created automatically if it doesn't exist. Ignored if --out is given.")
    args = parser.parse_args()
    global LEGEND_LOC
    LEGEND_LOC = args.legend_loc.replace("-", " ")  # matplotlib wants "lower right"

    resolved_path = resolve_premium_path(args.premium_file, args.char_level, args.threshold)
    premium_path = resolved_path if resolved_path is not None else args.premium_file
    if resolved_path is not None:
        print(f"Resolved '{args.premium_file}' -> {premium_path}")

    present_headers, rows = parse_premium_txt(premium_path)
    available_columns = set(present_headers)

    langs_csv_data = {}
    if args.langs_csv and os.path.exists(args.langs_csv):
        langs_csv_data = load_langs_csv(args.langs_csv)
    density_data = {}
    if args.density_json and os.path.exists(args.density_json):
        density_data = load_density_json(args.density_json)
    # Define fallback value depending on flag state
    fill_value = 1.0 if args.fill_spread_nulls else None

    spread_data = {}
    if args.spread_json and os.path.exists(args.spread_json):
        spread_data = load_spread_json(args.spread_json, fill_nulls=fill_value)
    spread_customenc_data = {}
    if args.spread_customenc_json and os.path.exists(args.spread_customenc_json):
        spread_customenc_data = load_spread_json(args.spread_customenc_json, fill_nulls=fill_value)

    available_columns |= merge_external_columns(
        rows, langs_csv_data, density_data, spread_data, spread_customenc_data)
    # Ranks are computed over ALL languages here, BEFORE the language
    # filter below is applied, so a language keeps the same
    # training_data rank in filtered and unfiltered plots.
    available_columns |= add_rank_columns(rows)
    available_columns |= add_log_columns(rows)

    if args.color_by is None:  # default: colour by bytes per char if we can
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

    # align by language (some rows may have been dropped independently
    # from each side due to blank/missing fields)
    x_by_lang = dict(zip(x_langs, x_vals))
    y_by_lang = dict(zip(y_langs, y_vals))
    common_langs = [l for l in x_by_lang if l in y_by_lang]

    # filter_tag ('' if no filter is active) goes into the plot titles
    # (via stem) and into the auto-derived output filename. color_tag
    # only goes into the filename (the legend already shows it on the plot).
    stem = os.path.splitext(os.path.basename(premium_path))[0] + filter_tag
    # Readable second title line, e.g. "Balanced, monotonicity threshold
    # (t = 0.6664) -- multi-byte languages". The raw stem is still used
    # for filenames.
    subtitle = describe_run(os.path.splitext(os.path.basename(premium_path))[0], args.char_level)
    filter_desc = describe_filter(script_types, bytes_per_char)
    if filter_desc:
        subtitle += f" \u2014 {filter_desc}"
    color_tag = "" if args.color_by == "none" else f"_color-{args.color_by}"
    run_key = RUN_NAME_LOOKUP.get(args.premium_file.strip().lower())  # "Balanced", "Imbalanced", "Balanced-Custom", or None
    setting_dir = run_key.lower() if run_key else "other"             # literal file paths land in .../other/...
    out_dir = os.path.join(args.out_dir, setting_dir, args.threshold)

    if args.control_for:
        c_kind, c_payload = parse_col_spec(args.control_for, available_columns)
        c_vals, c_label, c_langs = extract_values(c_kind, c_payload, rows)
        c_by_lang = dict(zip(c_langs, c_vals))
        common_langs = [l for l in common_langs if l in c_by_lang]

        if len(common_langs) < 4:
            print(f"Not enough languages with all three fields present ({len(common_langs)}) "
                  f"to compute a partial correlation.", file=sys.stderr)
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
            coloring=coloring)
        print(f"saved {out_path}")
        print(f"raw r({x_label}, {y_label}) = {r_raw:.4f}  r^2 = {r_raw ** 2:.4f}  p = {p_str(p_raw)}  (n={len(common_langs)})")
        print(f"partial r({x_label}, {y_label} | {c_label}) = {r_partial:.4f}  r^2 = {r_partial ** 2:.4f}  p = {p_str(p_partial)}  (n={len(common_langs)})")
    else:
        if len(common_langs) < 2:
            print(f"Not enough languages with both fields present ({len(common_langs)}) to plot.", file=sys.stderr)
            sys.exit(1)
        x_final = [x_by_lang[l] for l in common_langs]
        y_final = [y_by_lang[l] for l in common_langs]
        coloring = build_coloring(common_langs, langs_csv_data, args.color_by)

        auto_name = f"{stem}_corr_{x_label.replace('/','-')}_vs_{y_label.replace('/','-')}{color_tag}.png"
        out_path = args.out or os.path.join(out_dir, auto_name)
        print(f"output path: {out_path}")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        breakdown = None
        if args.breakdown != "off" and langs_csv_data:
            breakdown = residual_breakdown(x_final, y_final, common_langs, langs_csv_data)
        r, p = plot_plain(x_final, y_final, x_label, y_label, common_langs, subtitle, out_path,
                          coloring=coloring,
                          breakdown=breakdown if args.breakdown == "legend" else None)
        print(f"saved {out_path}")
        print(f"r({x_label}, {y_label}) = {r:.4f}  r^2 = {r * r:.4f}  p = {p_str(p)}  (n={len(common_langs)})")
        if breakdown is not None:
            print_breakdown(breakdown, oos=args.oos)


if __name__ == "__main__":
    main()