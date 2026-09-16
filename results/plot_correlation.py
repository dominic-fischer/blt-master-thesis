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
        codepoints, cp_per_baseline
      - the exact header text from the file (case-insensitive): Premium,
        PPS, BPP, EntropyMean, EntropyVar, EntropySkew, EntropyKurtosis,
        EntropyAutocorr1, EntropyVolatility
      - a RATIO of two such names, e.g. mean/variance -- computed
        elementwise from the two underlying columns

    --cols takes exactly two comma-separated specs: X,Y.

USAGE
    # plain correlation
    python3 plot_correlation.py premiums_sorted.txt --cols mean,premium

    # ratio as one side
    python3 plot_correlation.py premiums_sorted.txt --cols mean/variance,premium

    # partial correlation, controlling for a third column -- shows the
    # full 6-panel step-by-step walkthrough (same construction used
    # earlier for spread-vs-premium controlling for budget): X~control,
    # residual; Y~control, residual; then residual-vs-residual, with the
    # raw (uncontrolled) comparison shown separately for reference.
    python3 plot_correlation.py premiums_sorted.txt --cols mean,premium --control-for variance

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
    "imbalanced_bytes": "imbalanced_allocation_bytes",  # raw bytes, unchanged
    "imbalanced_allocation_bytes": "imbalanced_allocation_bytes",  # raw bytes, unchanged
    "training_data_balanced": "balanced_allocation_bytes_rank",
    "balanced_rank": "balanced_allocation_bytes_rank",
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


def resolve_column_name(name, available_columns):
    """Maps a user-given column name (alias or literal header text,
    case-insensitive either way) to the actual key present in the merged
    row data (premiums file columns, plus whatever was merged in from
    --langs-csv / --density-json). Raises a clear error listing what IS
    available if it can't be resolved."""
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
            f"--langs-csv or --density-json column, check those files were "
            f"found and contain a matching language_code / top-level key."
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


def load_density_json(path):
    """Returns {language_code: {stat_name: value}} as already produced
    by char_density.py."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_external_columns(rows, langs_csv_data, density_data):
    """Mutates each row dict in place, adding whichever of
    LANGS_CSV_COLUMNS / DENSITY_JSON_COLUMNS are available for that row's
    language. Returns the set of column names actually added (i.e. that
    were found for at least one language), for available_columns."""
    added = set()
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
    """Parses one --cols/--control-for entry: either a plain column
    name/alias, or a "A/B" ratio expression. Returns (kind, payload):
    ("plain", column_name) or ("ratio", (numerator_name, denominator_name)).
    """
    spec = spec.strip()
    m = re.match(r"^([^/]+)/([^/]+)$", spec)
    if m:
        num = resolve_column_name(m.group(1), available_columns)
        den = resolve_column_name(m.group(2), available_columns)
        return "ratio", (num, den)
    return "plain", resolve_column_name(spec, available_columns)


def extract_values(kind, payload, rows):
    """Returns (values, display_label, langs) -- values as a list of
    floats, skipping any row where the needed field(s) are blank/missing
    or non-numeric. display_label is a human string for axis labels."""
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


def scatter_with_fit(ax, x, y, langs, xlabel, ylabel, title, extra_df_used=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r, p = pearson_r_p(x, y, extra_df_used=extra_df_used)

    ax.scatter(x, y, c="#4C72B0", s=90, edgecolors="#333333", linewidths=0.8, zorder=3)
    if len(x) >= 2:
        coefs = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        fit_handle, = ax.plot(x_line, np.polyval(coefs, x_line), linestyle="--",
                               color="red", linewidth=1.6, zorder=2,
                               label=f"Linear fit (r={r:.2f}, p={p_str(p)})")
        ax.legend(handles=[fit_handle], loc="lower left", fontsize=8.5, framealpha=0.9)

    from collections import defaultdict
    x_groups = defaultdict(list)
    for i, xv in enumerate(x):
        x_groups[round(float(xv), 2)].append(i)
    for key, idxs in x_groups.items():
        idxs.sort(key=lambda i: y[i], reverse=True)
        for rank, i in enumerate(idxs):
            label = CODE_TO_LANG_NAME.get(langs[i], langs[i]) if langs else ""
            if label:
                ax.annotate(label, (x[i], y[i]), textcoords="offset points",
                            xytext=(7, 5 + rank * 10), fontsize=7.5, color="#333333",
                            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5, shrinkA=0, shrinkB=4))

    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
    return r, p


def plot_plain(x, y, x_label, y_label, langs, stem, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    r, p = scatter_with_fit(ax, x, y, langs, x_label, y_label,
                             f"{stem}: {x_label} vs. {y_label}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    return r, p


def plot_partial(x, y, ctrl, x_label, y_label, ctrl_label, langs, stem, out_path):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.asarray(ctrl, dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    scatter_with_fit(axes[0, 0], c, x, langs, ctrl_label, x_label,
                      f"(a) Step 1a: fit {x_label} ~ {ctrl_label}")
    x_fit = np.polyfit(c, x, 1)
    x_resid = x - np.polyval(x_fit, c)
    scatter_with_fit(axes[0, 1], c, x_resid, langs, ctrl_label, f"{x_label} residual",
                      f"(b) Step 2a: leftover {x_label}\n(should look flat vs. {ctrl_label})")

    r_raw, p_raw = scatter_with_fit(axes[0, 2], x, y, langs, x_label, y_label,
                                     f"(c) Reference only: RAW {x_label} vs. {y_label}")
    for spine in axes[0, 2].spines.values():
        spine.set_linestyle((0, (4, 3)))
        spine.set_color("#999999")
    axes[0, 2].set_facecolor("#f5f5f5")

    scatter_with_fit(axes[1, 0], c, y, langs, ctrl_label, y_label,
                      f"(d) Step 1b: fit {y_label} ~ {ctrl_label}")
    y_fit = np.polyfit(c, y, 1)
    y_resid = y - np.polyval(y_fit, c)
    scatter_with_fit(axes[1, 1], c, y_resid, langs, ctrl_label, f"{y_label} residual",
                      f"(e) Step 2b: leftover {y_label}\n(should look flat vs. {ctrl_label})")

    r_partial, p_partial = scatter_with_fit(
        axes[1, 2], x_resid, y_resid, langs, f"{x_label} residual", f"{y_label} residual",
        f"(f) Step 3: leftover vs. leftover\n= partial correlation", extra_df_used=1)

    fig.suptitle(f"{stem}: partial correlation walkthrough  --  "
                 f"raw r={r_raw:.3f} (p={p_str(p_raw)})  |  "
                 f"partial r={r_partial:.3f} (p={p_str(p_partial)})",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    return r_raw, p_raw, r_partial, p_partial


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("premium_file", help="A *_premiums_sorted.txt file")
    parser.add_argument("--cols", required=True,
                         help="Two comma-separated column specs, X,Y. Each can be a plain "
                              "column (name or alias) or a ratio A/B. E.g. mean,premium or "
                              "mean/variance,premium. Columns can come from the premiums file "
                              "itself, --langs-csv, or --density-json -- see module docstring.")
    parser.add_argument("--control-for", default=None,
                         help="A third column spec (same rules as --cols) to control for -- "
                              "if given, computes a partial correlation and shows the full "
                              "6-panel step-by-step walkthrough instead of a plain scatter.")
    parser.add_argument("--langs-csv", default="training_setup/langs/langs_chosen.csv",
                         help="CSV with a 'language_code' column plus training-data/typology "
                              "columns (ratio_vs_english, documents, utf8_bytes, "
                              "balanced_allocation_bytes, imbalanced_allocation_bytes, "
                              "ratio_vs_english_imbalanced, Approx_Bytes_Per_Char), merged in by "
                              "language code. Silently skipped if not found -- only an error if "
                              "you then reference a column that would have come from it. Pass "
                              "an empty string to disable.")
    parser.add_argument("--density-json", default="char_density.json",
                         help="JSON from char_density.py ({lang_code: {n_bytes_total, "
                              "n_codepoints, codepoints_per_baseline_codepoint, density_index}}), "
                              "merged in by language code. Same not-found behavior as "
                              "--langs-csv. Pass an empty string to disable.")
    parser.add_argument("--out", default=None, help="Output PNG path (overrides --out-dir entirely -- used exactly as given)")
    parser.add_argument("--out-dir", default="correlation_plots",
                         help="Directory the auto-derived output filename is saved into (default: correlation_plots/). "
                              "Created automatically if it doesn't exist. Ignored if --out is given.")
    args = parser.parse_args()

    present_headers, rows = parse_premium_txt(args.premium_file)
    available_columns = set(present_headers)

    langs_csv_data = {}
    if args.langs_csv and os.path.exists(args.langs_csv):
        langs_csv_data = load_langs_csv(args.langs_csv)
    density_data = {}
    if args.density_json and os.path.exists(args.density_json):
        density_data = load_density_json(args.density_json)

    available_columns |= merge_external_columns(rows, langs_csv_data, density_data)
    available_columns |= add_rank_columns(rows)

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

    stem = os.path.splitext(os.path.basename(args.premium_file))[0]

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

        auto_name = f"{stem}_partial_{x_label.replace('/','-')}_vs_{y_label.replace('/','-')}_ctrl_{c_label.replace('/','-')}.png"
        out_path = args.out or os.path.join(args.out_dir, auto_name)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        r_raw, p_raw, r_partial, p_partial = plot_partial(
            x_final, y_final, c_final, x_label, y_label, c_label, common_langs, stem, out_path)
        print(f"saved {out_path}")
        print(f"raw r({x_label}, {y_label}) = {r_raw:.4f}  p = {p_str(p_raw)}  (n={len(common_langs)})")
        print(f"partial r({x_label}, {y_label} | {c_label}) = {r_partial:.4f}  p = {p_str(p_partial)}  (n={len(common_langs)})")
    else:
        if len(common_langs) < 2:
            print(f"Not enough languages with both fields present ({len(common_langs)}) to plot.", file=sys.stderr)
            sys.exit(1)
        x_final = [x_by_lang[l] for l in common_langs]
        y_final = [y_by_lang[l] for l in common_langs]

        auto_name = f"{stem}_corr_{x_label.replace('/','-')}_vs_{y_label.replace('/','-')}.png"
        out_path = args.out or os.path.join(args.out_dir, auto_name)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        r, p = plot_plain(x_final, y_final, x_label, y_label, common_langs, stem, out_path)
        print(f"saved {out_path}")
        print(f"r({x_label}, {y_label}) = {r:.4f}  p = {p_str(p)}  (n={len(common_langs)})")


if __name__ == "__main__":
    main()