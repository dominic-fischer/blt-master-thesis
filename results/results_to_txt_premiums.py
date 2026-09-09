"""
results_to_txt_premiums.py

For each *_pps_premium column in --csv-in-path, writes a plain-text list
of premiums sorted DESCENDING, restricted to the languages in
--langs-csv's "language_code" column (the 20 languages actually trained
on -- equivalent to the master CSV's "Code_Orig").

GROUPED BY BOUND TYPE (4 folders) -> SUBFOLDERS (by prefix/stem) -> OPTIONAL STEP SUBFOLDER, 
columns keyed by EXACT NUMERIC THRESHOLD: run_patching.py keys its output 
(and therefore results_to_CSV.py's columns) by the literal calibrated 
threshold value, e.g. "raw_entropy_t_1.3340_pps_premium" -- NOT by a bound label. 
To sort these into the 4 bound-type folders, this script cross-references
--summary-csv (the SAME calibration CSV run_patching.py used for this
checkpoint) to look up which numeric value corresponds to which of
low/mid/high/anchor, per case -- reusing run_patching.py's own
load_cases()/threshold_key() directly rather than duplicating that
parsing logic.

SCORE SOURCE (--score-source bytes|chars): mirrors calibrate_thresholds.py/
run_patching.py/results_to_CSV.py. With --score-source=chars, columns in
--csv-in-path carry a "char_" prefix (added by results_to_CSV.py's
COLUMN_PREFIX) ahead of the case name, e.g.
"char_raw_entropy_t_1.3340_pps_premium" -- this prefix is stripped before
parsing into (case, t_key) (see parse_premium_column) and load_cases() is
called with the matching score_source so its case set and threshold
lookups match what run_patching.py actually produced (norm_entropy/
combined are never present in chars-mode, matching KNOWN_CASES below).
Output additionally nests under a "char_level" folder (see Output below)
so byte-mode and chars-mode runs never share a directory even if pointed
at the same --out-dir. This script does NOT chain both score sources
automatically -- run it twice (once per --score-source), pointing at the
matching --csv-in-path/--summary-csv pair each time (see Usage below).

ENTROPY MEAN/VARIANCE: for each language, we additionally compute the
mean and (population) variance of its entropy signal, giving a sense of
how high and how oscillating that language's entropy signal is overall.
These are read from the per-language JSON files under results/base_model/
or results/own_models/<raw_run_stem>/<step>/ -- NOT from --csv-in-path,
which only carries aggregate pps/bpp/premium columns. The JSON directory
is auto-derived from the CSV's filename (or --filename-prefix, reversed
back through RUN_NAME_ALIASES if it's an alias rather than a raw stem)
unless --results-json-dir is given explicitly. Values are computed once
per language (not per case/threshold, since the underlying entropy
signal is shared across all cases within a given score source) and are
the SAME for every output file of a given run + score source.

Which entropy signal is used depends on --score-source, matching the
granularity that mode's patch boundaries were actually thresholded over
(see blt_patcher.py / inspect_results.py):
  - bytes (default): raw per-byte entropy, i.e. bytes_entropies[i][1]
    ("entropy_raw"), flattened across every byte of every sentence.
  - chars: summed raw per-CHARACTER entropy, i.e. chars_entropies[i][1],
    flattened across every character of every sentence. This is NOT the
    same quantity as flattening bytes_entropies in chars mode -- it is
    the actual per-character SUM that char-mode patching thresholds
    over, so using it keeps EntropyMean/EntropyVar consistent with what
    that mode's boundaries were computed from.
If no JSON directory is found, or the relevant key is absent for a
language, entropy columns are simply omitted -- see load_entropy_stats /
default_results_json_dir.

RUN NAME ALIASING: raw run-name stems (e.g.
"entropy_10M_20lang_4gpu_sourcesbalanced_steps10000_ckpt200_customenc_lr4.5e-3")
are long and not meant for human-facing filenames/folders. RUN_NAME_ALIASES
maps known raw stems to short, readable names (e.g. "Balanced-Custom") --
applied via apply_run_name_alias() to the FULL raw run stem (see
resolve_raw_run_stem) BEFORE it's used for any output path or filename,
so aliasing is transparent to the rest of this script's logic. Add new
entries to RUN_NAME_ALIASES as new runs are evaluated; anything not in
the map passes through unchanged (falls back to the raw stem), so this is
purely additive/non-breaking. REVERSE_RUN_NAME_ALIASES is the
automatically-derived inverse mapping, used to recover the raw run stem
(for locating the JSON results directory) when --filename-prefix was
given as a human-readable alias rather than the raw stem itself.

Output:
    results/txt_premiums/t_anchor/<subfolder>/[<step_subfolder>/][<prefix>_]<case>_t_<value>_premiums_sorted.txt
    results/txt_premiums/t_lower_bound/<subfolder>/[<step_subfolder>/][<prefix>_]<case>_t_<value>_premiums_sorted.txt
    results/txt_premiums/t_midpoint/<subfolder>/[<step_subfolder>/][<prefix>_]<case>_t_<value>_premiums_sorted.txt
    results/txt_premiums/t_upper_bound/<subfolder>/[<step_subfolder>/][<prefix>_]<case>_t_<value>_premiums_sorted.txt
  (--score-source=chars nests all of the above one level deeper, under
  results/txt_premiums/char_level/<bound>/... -- see build_out_dir.)

where <subfolder> and <prefix> are BOTH the aliased full raw run stem
(see RUN NAME ALIASING above) -- i.e. --filename-prefix (or the
auto-derived stem from the CSV filename) after being resolved to its
full raw form and passed through apply_run_name_alias(). If "step_X" is
present in the CSV filename, it is additionally nested inside an extra
step subfolder.

Each file contains language code, sorted premium, and the absolute patches-per-sentence 
(pps) and bytes-per-patch (bpp) values, followed by that language's entropy mean and
variance (if the per-language JSON results could be located -- see ENTROPY MEAN/VARIANCE
above).

Each file's FINAL LINE records English's own (non-premium) pps/bpp for
that exact case+threshold -- since English's own premium is trivially
1.0 by definition, this instead gives the actual operating point
(patches/sentence, bytes/patch) the premiums above it are relative to.

Usage:
    python results/results_to_txt_premiums.py --csv-in-path results/results_CSV/<stem>_results.csv --summary-csv calibrated_thresholds/<stem>_thresholds_summary.csv
    python results/results_to_txt_premiums.py --csv-in-path <csv> --summary-csv <csv> --langs-csv training_setup/langs/langs_chosen.csv --filename-prefix <stem_or_alias>
    # Char-level score source (run separately, pointing at the char_level/ CSV + summary):
    python results/results_to_txt_premiums.py --csv-in-path results/results_CSV/char_level/<stem>_results.csv --summary-csv calibrated_thresholds/char_level/<stem>_thresholds_summary.csv --score-source chars --filename-prefix <stem_or_alias>
"""
import argparse
import json
import os
import re
import statistics
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))  # noqa: E402
from model_eval.run_patching import load_cases, threshold_key, BOUND_NAMES, DEFAULT_SUMMARY_CSV

import pandas as pd

DEFAULT_LANGS_CSV = "training_setup/langs/langs_chosen.csv"
DEFAULT_OUT_DIR = "results/txt_premiums"
PREMIUM_SUFFIX = "_pps_premium"
ENGLISH = "eng_Latn"
SCORE_SOURCES = ("bytes", "chars")
# Matches results_to_CSV.py's COLUMN_PREFIX -- the prefix stripped off
# the front of every column name before parsing, in chars-mode.
COLUMN_PREFIX = {"bytes": "", "chars": "char_"}
CHAR_LEVEL_FOLDER = "char_level"

# Which top-level JSON key holds the per-unit entropy list to use for
# EntropyMean/EntropyVar, per score source -- see ENTROPY MEAN/VARIANCE
# in the module docstring for why these are NOT interchangeable.
ENTROPY_JSON_KEY = {"bytes": "bytes_entropies", "chars": "chars_entropies"}

# Must match run_patching.py's CASES/COMBINED keys. Both bytes-mode and
# chars-mode use the same KNOWN_CASES here since norm_entropy/combined
# never appear in a chars-mode CSV in the first place (run_patching.py's
# load_cases already excludes them there) -- nothing extra to filter.
KNOWN_CASES = ["raw_entropy", "raw_monotonicity"]

# Maps run_patching.py's internal bound name -> the folder name requested.
BOUND_FOLDER_NAMES = {
    "low":    "t_lower_bound",
    "mid":    "t_midpoint",
    "high":   "t_upper_bound",
    "anchor": "t_anchor",
}

# Raw run-name stem -> short, human-readable name. See module docstring's
# "RUN NAME ALIASING" section. Matched via exact-substring replacement
# (apply_run_name_alias), so a raw name embedded with an extra suffix
# (e.g. a trailing "_step_0000010000") still gets its known-run portion
# swapped out cleanly, leaving the step suffix intact.
RUN_NAME_ALIASES = {
    "entropy_10M_20lang_4gpu_sourcesbalanced_steps10000_ckpt200_customenc_lr4.5e-3": "Balanced-Custom",
    "entropy_10M_20lang_4gpu_sourcesbalanced_steps10000_ckpt200_lr4.5e-3": "Balanced",
    "entropy_10M_20lang_4gpu_sourcesimbalanced_steps10000_ckpt200_lr4.5e-3": "Imbalanced",
    "base_model":"_Base-Model"
}

# Reverse of RUN_NAME_ALIASES: alias -> raw stem. Lets us go from a
# human-readable name (however --filename-prefix was given) back to the
# raw run directory name under results/own_models/, for locating the
# per-language JSON results (see resolve_raw_run_stem).
REVERSE_RUN_NAME_ALIASES = {v: k for k, v in RUN_NAME_ALIASES.items()}


def apply_run_name_alias(name: str) -> str:
    """Replaces the FIRST matching raw run-name stem found anywhere in
    name with its short alias from RUN_NAME_ALIASES, and strips any
    trailing "_step_<digits>" suffix (e.g. "_step_0000010000") entirely,
    rather than preserving it. Longer keys are checked first so a raw
    name that is itself a prefix of another mapped name (not currently
    the case here, but cheap insurance) can't shadow the more specific
    match. Returns name unchanged (step suffix stripped either way) if no
    known raw stem is found -- unmapped runs simply fall back to their
    raw name rather than erroring."""
    for raw in sorted(RUN_NAME_ALIASES, key=len, reverse=True):
        if raw in name:
            name = name.replace(raw, RUN_NAME_ALIASES[raw])
            break
    return re.sub(r"_step_\d+", "", name)


def resolve_raw_run_stem(csv_stem: str, filename_prefix: str | None) -> str:
    """Recovers the FULL raw run stem (e.g.
    'entropy_10M_20lang_4gpu_sourcesbalanced_steps10000_ckpt200_customenc_lr4.5e-3'),
    used both to locate the per-language JSON folder AND as the basis for
    the output filename/subfolder prefix (before aliasing -- see main()).
    If --filename-prefix was given, it might already be a human-readable
    ALIAS (e.g. 'Balanced-Custom') rather than the raw stem --
    REVERSE_RUN_NAME_ALIASES maps it back. Otherwise, falls back to the
    CSV's own filename with the step suffix and a trailing '_results'
    trimmed off, which is the raw stem as-is."""
    if filename_prefix:
        return REVERSE_RUN_NAME_ALIASES.get(filename_prefix, filename_prefix)
    stem = re.sub(r"_step_\d+", "", csv_stem)
    if stem.endswith("_results"):
        stem = stem[: -len("_results")]
    return stem


def default_results_json_dir(raw_run_stem: str, step_subfolder: str | None) -> str:
    """Derives the per-language JSON results directory from the FULL raw
    run stem (not the aliased/short prefix used for output naming).
    'base_model' is special-cased since it lives directly under
    results/base_model/ rather than results/own_models/<stem>/."""
    if raw_run_stem == "base_model":
        return os.path.join("results", "base_model")
    parts = ["results", "own_models", raw_run_stem]
    if step_subfolder:
        parts.append(step_subfolder)
    return os.path.join(*parts)


def load_entropy_stats(results_json_dir: str, lang_codes: set[str], score_source: str) -> dict[str, tuple[float, float]]:
    """Returns {lang_code: (mean_entropy, var_entropy)}.

    Which JSON key/quantity is used depends on score_source -- see
    ENTROPY MEAN/VARIANCE in the module docstring:
      - 'bytes': flattens raw per-byte entropy (bytes_entropies[i][1],
        i.e. entropy_raw) across every byte of every sentence.
      - 'chars': flattens summed raw per-character entropy
        (chars_entropies[i][1]) across every character of every
        sentence -- the actual quantity char-mode patching thresholds
        over, NOT re-derived from bytes_entropies.

    Languages whose JSON is missing, or which lack the relevant key
    entirely, are silently omitted -- caller reports this against the
    requested set."""
    entropy_key = ENTROPY_JSON_KEY[score_source]
    stats = {}
    for lang in lang_codes:
        fpath = os.path.join(results_json_dir, f"{lang}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath, encoding="utf-8") as f:
            sentences = json.load(f)
        all_entropies = [
            e[1]  # entropy_raw (bytes mode) / summed raw entropy per char (chars mode)
            for sent in sentences
            for e in sent.get(entropy_key, [])
        ]
        if not all_entropies:
            continue
        stats[lang] = (statistics.mean(all_entropies), statistics.pvariance(all_entropies))
    return stats


def load_chosen_languages(langs_csv: str) -> set[str]:
    df = pd.read_csv(langs_csv)
    return set(df["language_code"].astype(str))


def build_key_to_bound(summary_csv: str, score_source: str) -> dict[tuple[str, str], str]:
    """Returns {(case_name, "t_<value>"): bound_name}, reusing
    run_patching.py's own load_cases()/threshold_key() so this always
    matches exactly how run_patching.py itself keyed its output --
    no separate float-formatting logic to drift out of sync. score_source
    must match the --score-source that produced summary_csv, so
    load_cases() applies the same norm_entropy/combined exclusion for
    chars-mode that run_patching.py itself applied."""
    cases = load_cases(summary_csv, score_source)
    key_to_bound = {}
    for case_name, case in cases.items():
        for bound_name in BOUND_NAMES:
            t = case["named_thresholds"][bound_name]
            key_to_bound[(case_name, threshold_key(t))] = bound_name
    return key_to_bound


def parse_premium_column(col: str, score_source: str) -> tuple[str, str] | None:
    """Splits "{prefix}{case}_{t_key}_pps_premium" into (case, t_key),
    where t_key is the literal "t_<value>" string (e.g. "t_1.3340") --
    bound identity is looked up separately via build_key_to_bound, not
    parsed out of the column name itself. prefix is COLUMN_PREFIX[score_source]
    ("" for bytes, "char_" for chars) and is stripped first. Returns None
    if col doesn't end in PREMIUM_SUFFIX, doesn't start with the expected
    prefix, or doesn't match a known case -- defensive against unrelated
    columns / naming drift."""
    if not col.endswith(PREMIUM_SUFFIX):
        return None
    prefix = COLUMN_PREFIX[score_source]
    if not col.startswith(prefix):
        return None
    mode_str = col[len(prefix): -len(PREMIUM_SUFFIX)]  # "{case}_t_{value}"
    for case in KNOWN_CASES:
        case_prefix = f"{case}_"
        if mode_str.startswith(case_prefix):
            t_key = mode_str[len(case_prefix):]  # "t_1.3340"
            return case, t_key
    return None


def build_out_dir(base_out_dir: str, bound_folder: str, subfolder_name: str,
                   step_subfolder: str | None, score_source: str) -> str:
    """out_dir / [char_level /] bound / prefix [/ step_subfolder] --
    chars-mode nests everything one level deeper under CHAR_LEVEL_FOLDER
    so it never shares a directory with a bytes-mode run of the same
    model/checkpoint."""
    parts = [base_out_dir]
    if score_source == "chars":
        parts.append(CHAR_LEVEL_FOLDER)
    parts.append(bound_folder)
    parts.append(subfolder_name)
    if step_subfolder:
        parts.append(step_subfolder)
    return os.path.join(*parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--csv-in-path", 
        required=True,
        help="CSV produced by results_to_CSV.py, containing "
             "*_pps_premium columns and a 'Code_Orig' column."
    )
    parser.add_argument(
        "--summary-csv", 
        default=DEFAULT_SUMMARY_CSV,
        help=f"The SAME calibration CSV used by run_patching.py for "
             f"this checkpoint (and with the same --score-source), used "
             f"here to look up which numeric threshold is which bound "
             f"(default {DEFAULT_SUMMARY_CSV})."
    )
    parser.add_argument(
        "--langs-csv", 
        default=DEFAULT_LANGS_CSV,
        help=f"CSV whose 'language_code' column restricts which "
             f"languages are included (default {DEFAULT_LANGS_CSV})."
    )
    parser.add_argument(
        "--out-dir", 
        default=DEFAULT_OUT_DIR,
        help=f"Base output directory (default {DEFAULT_OUT_DIR}); "
             f"one subfolder per bound type is created under it "
             f"(nested under char_level/ first, for --score-source=chars)."
    )
    parser.add_argument(
        "--score-source",
        choices=SCORE_SOURCES,
        default="bytes",
        help="Must match the --score-source used for --csv-in-path "
             "(results_to_CSV.py) and --summary-csv (calibrate_thresholds.py). "
             "'bytes' (default) expects unprefixed columns; 'chars' expects "
             "'char_'-prefixed columns, nests output under char_level/, and "
             "computes EntropyMean/EntropyVar from chars_entropies instead "
             "of bytes_entropies. See module docstring."
    )
    parser.add_argument(
        "--filename-prefix", 
        default=None,
        help="Optional prefix (e.g. a model/checkpoint stem, or one of its "
             "short RUN_NAME_ALIASES) for the output filename and subfolder. "
             "If omitted, the raw stem is parsed automatically from the "
             "input CSV's filename. Either way, resolved to the full raw "
             "run stem (see resolve_raw_run_stem) and passed through "
             "apply_run_name_alias() before use -- see RUN NAME ALIASING."
    )
    parser.add_argument(
        "--results-json-dir",
        default=None,
        help="Directory containing per-language {lang}.json result files, "
             "used to compute per-language entropy mean/variance. If "
             "omitted, auto-derived from the CSV filename / --filename-prefix "
             "(reversed through RUN_NAME_ALIASES if needed) -- see ENTROPY "
             "MEAN/VARIANCE in the module docstring. If the resulting "
             "directory doesn't exist, entropy columns are simply omitted."
    )
    return parser.parse_args()


def main():
    print("[DEBUG] main() started, parsing args...", flush=True)
    args = parse_args()
    source = args.score_source
    df = pd.read_csv(args.csv_in_path)
    chosen_langs = load_chosen_languages(args.langs_csv)
    key_to_bound = build_key_to_bound(args.summary_csv, source)

    if "Code_Orig" not in df.columns:
        raise ValueError(f"'Code_Orig' column not found in {args.csv_in_path}")

    csv_basename = os.path.basename(args.csv_in_path)
    csv_stem, _ = os.path.splitext(csv_basename)

    # Locate the FULL raw run stem FIRST -- used both for JSON-dir lookup
    # and as the basis for the output prefix/subfolder name, so a short
    # alias (e.g. "Balanced") is correctly recovered even when
    # --filename-prefix was not given explicitly. (Previously, automatic
    # derivation truncated to the first two underscore-parts of the
    # filename BEFORE attempting the alias lookup, so the alias -- keyed
    # on the full stem -- never matched and the raw truncated name leaked
    # through instead.)
    raw_run_stem = resolve_raw_run_stem(csv_stem, args.filename_prefix)
    step_match = re.search(r"step_\d+", csv_stem)
    step_subfolder = step_match.group(0) if step_match else None

    prefix = apply_run_name_alias(raw_run_stem)
    subfolder_name = prefix

    # Locate and load per-language entropy JSON, for the mean/variance
    # columns -- see ENTROPY MEAN/VARIANCE in the module docstring.
    results_json_dir = args.results_json_dir or default_results_json_dir(raw_run_stem, step_subfolder)

    entropy_stats = {}
    if os.path.isdir(results_json_dir):
        entropy_stats = load_entropy_stats(results_json_dir, chosen_langs, source)
        missing_entropy = chosen_langs - set(entropy_stats)
        if missing_entropy:
            print(f"  NOTE: no entropy JSON found for {len(missing_entropy)} language(s) "
                  f"in {results_json_dir} (score_source={source!r}, key={ENTROPY_JSON_KEY[source]!r}): "
                  f"{sorted(missing_entropy)}")
    else:
        print(f"  NOTE: results-json-dir {results_json_dir!r} not found -- "
              f"entropy mean/variance columns will be omitted.")

    filtered = df[df["Code_Orig"].astype(str).isin(chosen_langs)]
    missing = chosen_langs - set(filtered["Code_Orig"].astype(str))
    if missing:
        print(f"  NOTE: {len(missing)} language(s) from {args.langs_csv} not found "
              f"in {args.csv_in_path}: {sorted(missing)}")

    premium_cols = [c for c in df.columns if c.endswith(PREMIUM_SUFFIX)]
    if not premium_cols:
        print(f"No *{PREMIUM_SUFFIX} columns found in {args.csv_in_path} -- nothing to write.")
        return

    print(f"Score source: {source}")
    print(f"Found {len(premium_cols)} premium column(s); "
          f"{len(filtered)}/{len(chosen_langs)} chosen languages present.")

    eng_row = filtered[filtered["Code_Orig"].astype(str) == ENGLISH]
    if len(eng_row) == 0:
        print(f"  NOTE: {ENGLISH} not found in the filtered set -- the English "
              f"reference line will be omitted from every file.")

    for i, col in enumerate(premium_cols):
        parsed = parse_premium_column(col, source)
        if parsed is None:
            print(f"  WARNING: could not parse column {col!r} into (case, t_key) for "
                  f"score-source={source!r} -- skipping (unrecognized case name, wrong "
                  f"prefix, or naming drift vs run_patching.py/results_to_CSV.py).")
            continue
        case, t_key = parsed

        bound_name = key_to_bound.get((case, t_key))
        if bound_name is None:
            print(f"  WARNING: {case}/{t_key} not found in {args.summary_csv}'s calibrated "
                  f"bounds for this case -- skipping (was --summary-csv the same one "
                  f"run_patching.py used for this checkpoint AND score-source?).")
            continue
        bound_folder = BOUND_FOLDER_NAMES.get(bound_name, f"t_{bound_name}")

        out_dir = build_out_dir(args.out_dir, bound_folder, subfolder_name, step_subfolder, source)
        os.makedirs(out_dir, exist_ok=True)

        base_name = f"{case}_{t_key}_premiums_sorted.txt"
        filename = f"{prefix}_{base_name}"
        out_path = os.path.join(out_dir, filename)

        # Retrieve absolute pps and bpp columns alongside premium (same
        # prefix as the premium column itself).
        col_prefix = COLUMN_PREFIX[source]
        mode_str = f"{col_prefix}{case}_{t_key}"
        pps_col = f"{mode_str}_pps"
        bpp_col = f"{mode_str}_bpp"

        has_extra_cols = pps_col in filtered.columns and bpp_col in filtered.columns
        target_cols = ["Code_Orig", col]
        if has_extra_cols:
            target_cols.extend([pps_col, bpp_col])

        rows = filtered[target_cols].dropna().sort_values(col, ascending=False)

        # Generate string values with fixed precision so we can measure exact character widths
        formatted_rows = []
        for _, row in rows.iterrows():
            lang = str(row['Code_Orig'])
            prem = f"{row[col]:.4f}"
            values = [lang, prem]
            if has_extra_cols:
                values.append(f"{row[pps_col]:.4f}")
                values.append(f"{row[bpp_col]:.4f}")
            if entropy_stats:
                if lang in entropy_stats:
                    mean_e, var_e = entropy_stats[lang]
                    values.append(f"{mean_e:.4f}")
                    values.append(f"{var_e:.4f}")
                else:
                    values.append("")
                    values.append("")
            formatted_rows.append(tuple(values))

        # Dynamic alignment: compute max width of each column (including safety margin of 2 spaces)
        headers = ["Language", "Premium"]
        if has_extra_cols:
            headers += ["PPS", "BPP"]
        if entropy_stats:
            headers += ["EntropyMean", "EntropyVar"]

        col_widths = []
        for i, header in enumerate(headers):
            # Find length of longest value in this index across all formatted rows and the header itself
            max_len = max(len(row[i]) for row in formatted_rows) if formatted_rows else 0
            col_widths.append(max(len(header), max_len) + 2)

        with open(out_path, "w", encoding="utf-8") as f:
            # Write dynamic left-aligned headers
            header_str = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)).rstrip()
            f.write(header_str + "\n")

            # Write formatted data rows with matching left alignment
            for row in formatted_rows:
                row_str = "".join(f"{val:<{w}}" for val, w in zip(row, col_widths)).rstrip()
                f.write(row_str + "\n")

            if len(eng_row) > 0:
                if has_extra_cols:
                    eng_pps = eng_row[pps_col].values[0]
                    eng_bpp = eng_row[bpp_col].values[0]
                    f.write(f"\n# {ENGLISH} reference ({case}, {t_key}, {bound_folder}): "
                            f"pps={eng_pps:.4f}\tbpp={eng_bpp:.4f}\n")
                else:
                    print(f"    NOTE: {pps_col}/{bpp_col} not found -- omitting English "
                          f"reference line for {out_path}")

        print(f"  {case} / {t_key} ({bound_folder}/{subfolder_name}{f'/{step_subfolder}' if step_subfolder else ''}): wrote {len(rows)} language(s) -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()