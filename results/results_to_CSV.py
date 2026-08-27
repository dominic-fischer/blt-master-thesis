"""
results_to_CSV.py

For each language in --csv-in-path, loads the corresponding
--results-dir/{Code_Orig}.json and computes mean pps and global bpp
(total_bytes / total_patches) for each case/threshold combination, adding
them as new columns to the CSV.

Also computes pps_premium columns: lang_pps / eng_pps for each case/threshold.

If a language results JSON is missing, the columns for that language will 
be filled with N/A.

SCORE SOURCE (--score-source bytes|chars): mirrors calibrate_thresholds.py
and run_patching.py.
  - bytes (default): reads sentence["eval_modes"], and bpp uses
    vals["patch_lengths"]'s sum (already in bytes -- a byte-mode "unit"
    IS a byte).
  - chars: reads sentence["char_eval_modes"] (must already be populated
    by run_patching.py --score-source=chars), and bpp uses
    vals["patch_lengths_bytes"]'s sum -- NOT patch_lengths_chars, whose
    sum would give bpp in units of characters-per-patch, a different and
    not-directly-comparable quantity. See run_patching.py's OUTPUT SHAPE
    section for why both are stored.
  Column names get a "char_" prefix in chars-mode (e.g.
  char_raw_entropy_t_1.1904_pps) so byte-mode and chars-mode results
  never collide if ever written to the same CSV -- though by convention
  (see results_to_txt_premiums.py / run_eval_and_patch.py) chars-mode
  output goes to a separate char_level/ output path entirely.

Column naming:
  {prefix}{case}_{threshold_key}_pps         e.g. raw_entropy_t_1.1904_pps
  {prefix}{case}_{threshold_key}_bpp         e.g. raw_entropy_t_1.1904_bpp
  {prefix}{case}_{threshold_key}_pps_premium e.g. raw_entropy_t_1.1904_pps_premium
  (prefix is "" for bytes, "char_" for chars)

LANGUAGE FILTERING (--langs-csv): restricts --csv-in-path down to just the
languages listed in a CSV's "language_code" column (e.g. "eng_Latn")
before doing anything else -- so languages you were never going to keep
(e.g. the other ~200 FLORES+ languages when you only trained on 20) don't
even get a "Missing JSON" line. Default: training_setup/langs/langs_chosen.csv.
Pass --langs-csv none to process every language in --csv-in-path instead.

CASE FILTERING (--cases): only top-level case names in this set are
aggregated at all; anything else found in a results JSON (e.g. legacy
"combined"/"norm_entropy" columns from an earlier pipeline stage) is
skipped entirely. Default: "raw_entropy,raw_monotonicity". Pass
--cases all to keep every case name found (useful for debugging what's
actually in a results file).

STALE/PARTIAL THRESHOLD FILTERING: within a kept case, a given threshold
column is only included if it appears in EVERY sentence of that
language's file. If it only appears in a subset (e.g. because a
re-calibration run only covered some sentences, leaving stale entries
from an earlier run mixed in with the new ones), it's dropped and
reported rather than being silently averaged over fewer sentences than
the rest of the columns.

The INPUT csv is left exactly as-is; the new columns are written to a
SEPARATE output csv (--csv-out-path), which defaults to
<csv-in-path stem>_with_results.csv if not given explicitly, so the
input file is never silently overwritten.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import pandas as pd

ENGLISH = "eng_Latn"
SCORE_SOURCES = ("bytes", "chars")
EVAL_MODES_KEY = {"bytes": "eval_modes", "chars": "char_eval_modes"}
COLUMN_PREFIX = {"bytes": "", "chars": "char_"}
DEFAULT_CASES = "raw_entropy,raw_monotonicity"
DEFAULT_LANGS_CSV = "training_setup/langs/langs_chosen.csv"


def default_csv_out_path(csv_in_path: str) -> str:
    """<name>.csv -> <name>_with_results.csv"""
    p = Path(csv_in_path)
    return str(p.with_name(f"{p.stem}_with_results{p.suffix}"))


def parse_cases_arg(cases_arg: str) -> set[str] | None:
    """'all' (any case) -> None (no filtering). Otherwise a comma-separated
    list -> a set of case names to keep."""
    if cases_arg.strip().lower() == "all":
        return None
    cases = {c.strip() for c in cases_arg.split(",") if c.strip()}
    if not cases:
        raise ValueError(f"--cases produced an empty set from {cases_arg!r}")
    return cases


def load_chosen_codes(langs_csv: str) -> set[str] | None:
    """Load the 'language_code' column (e.g. 'eng_Latn') from a CSV of
    chosen languages, used to restrict --csv-in-path to just these
    languages before doing anything else. Returns None (no filtering) if
    langs_csv is falsy or 'none'."""
    if not langs_csv or langs_csv.strip().lower() == "none":
        return None
    path = Path(langs_csv)
    if not path.exists():
        raise FileNotFoundError(f"--langs-csv file not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "language_code" not in (reader.fieldnames or []):
            raise ValueError(
                f"'language_code' column not found in {path}; got {reader.fieldnames}"
            )
        codes = {row["language_code"].strip() for row in reader if row["language_code"].strip()}
    if not codes:
        raise ValueError(f"No language codes found in {path}")
    return codes


def load_results(results_dir: str, code_orig: str) -> list[dict] | None:
    path = Path(results_dir) / f"{code_orig}.json"
    if not path.exists():
        # Instead of raising an error, we return None to signal a missing file
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def aggregate(
    sentences: list[dict],
    source: str,
    cases: set[str] | None,
) -> tuple[dict[str, float], list[str]]:
    """Compute mean pps and global bpp (total_bytes / total_patches) for
    every case/threshold, reading from the source-appropriate eval_modes
    key. Columns are prefixed per COLUMN_PREFIX (empty for bytes).

    Two things are filtered out before aggregation:
      1. Case names not in `cases` (if `cases` is not None) are skipped
         entirely -- e.g. to drop legacy "combined"/"norm_entropy" data
         regardless of whether it's internally consistent.
      2. Within a kept case, any threshold column that doesn't appear in
         EVERY sentence is treated as leftover from an earlier/partial
         run and dropped, rather than being averaged over however many
         sentences happen to have it.

    Returns (result_columns, dropped_column_descriptions).
    """
    modes_key = EVAL_MODES_KEY[source]
    prefix = COLUMN_PREFIX[source]
    total_sentences = len(sentences)
    totals = {}  # col -> [total_patches, total_bytes, n_sentences_seen]

    for s in sentences:
        for case_name, thresholds in s.get(modes_key, {}).items():
            if cases is not None and case_name not in cases:
                continue
            for t_key, vals in thresholds.items():
                col = f"{prefix}{case_name}_{t_key}"
                if col not in totals:
                    totals[col] = [0, 0, 0]
                if source == "bytes":
                    patch_bytes_sum = sum(vals["patch_lengths"])
                else:
                    patch_bytes_sum = sum(vals["patch_lengths_bytes"])
                totals[col][0] += vals["n_patches"]
                totals[col][1] += patch_bytes_sum
                totals[col][2] += 1

    result = {}
    dropped = []
    for col, (total_patches, total_bytes, n_sentences_seen) in totals.items():
        if n_sentences_seen != total_sentences:
            dropped.append(f"{col} ({n_sentences_seen}/{total_sentences} sentences)")
            continue
        result[f"{col}_pps"] = round(total_patches / n_sentences_seen, 4)
        result[f"{col}_bpp"] = round(total_bytes / total_patches, 4)
    return result, dropped


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default="results/restructured",
                         help="Directory of per-language JSON files with eval_modes "
                              "(or char_eval_modes, for --score-source=chars) populated "
                              "(default: results/restructured).")
    parser.add_argument("--csv-in-path", default="floresplus_MASTER.csv",
                         help="Input CSV, read but never modified "
                              "(default: floresplus_MASTER.csv).")
    parser.add_argument("--csv-out-path", default=None,
                         help="Output CSV with the new columns added. If omitted, "
                              "derived from --csv-in-path as <name>_with_results.csv.")
    parser.add_argument("--score-source", choices=SCORE_SOURCES, default="bytes",
                         help="Which eval_modes key to aggregate from: 'bytes' (default, "
                              "reads sentence['eval_modes']) or 'chars' (reads "
                              "sentence['char_eval_modes'], requires run_patching.py "
                              "--score-source=chars to have already populated it). "
                              "Columns get a 'char_' prefix in chars-mode.")
    parser.add_argument("--cases", default=DEFAULT_CASES,
                         help="Comma-separated top-level case names to keep "
                              f"(default: '{DEFAULT_CASES}'). Anything else found in "
                              "the results JSON (e.g. legacy 'combined'/'norm_entropy' "
                              "columns) is skipped entirely. Pass --cases all to keep "
                              "every case name found, unfiltered.")
    parser.add_argument("--langs-csv", default=DEFAULT_LANGS_CSV,
                         help="CSV with a 'language_code' column (e.g. 'eng_Latn') used "
                              "to restrict --csv-in-path down to just these languages "
                              "before doing anything else -- so languages you were never "
                              "going to keep don't even get a 'Missing JSON' line. Pass "
                              "--langs-csv none to disable filtering and process every "
                              f"language in --csv-in-path (default: {DEFAULT_LANGS_CSV}).")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_out_path = args.csv_out_path or default_csv_out_path(args.csv_in_path)
    source = args.score_source
    cases = parse_cases_arg(args.cases)

    df = pd.read_csv(args.csv_in_path)
    print(f"Loaded {len(df)} languages from {args.csv_in_path}")

    chosen_codes = load_chosen_codes(args.langs_csv)
    if chosen_codes is not None:
        before = len(df)
        df = df[df["Code_Orig"].isin(chosen_codes)].reset_index(drop=True)
        missing_from_master = chosen_codes - set(df["Code_Orig"])
        print(f"Filtered to {len(df)} / {before} languages via --langs-csv {args.langs_csv}")
        if missing_from_master:
            print(
                f"  warning: {len(missing_from_master)} chosen language(s) not found "
                f"in {args.csv_in_path}: {sorted(missing_from_master)}"
            )
    else:
        print("No --langs-csv filtering applied; processing every language in --csv-in-path.")

    print(f"Score source: {source}  (reading sentence['{EVAL_MODES_KEY[source]}'])")
    print(f"Cases kept: {'all' if cases is None else sorted(cases)}")

    # Keep track of all keys seen across valid files so we can fill missing ones with NaN
    all_known_keys = set()
    rows_data = []
    total_dropped = 0

    for _, row in df.iterrows():
        code_orig = row["Code_Orig"]
        sentences = load_results(args.results_dir, code_orig)

        if sentences is None:
            # File is missing. We save the baseline row and fill the rest later
            rows_data.append((row.to_dict(), None))
            print(f"  {code_orig}: Missing JSON -> filling with N/A")
        else:
            agg, dropped = aggregate(sentences, source, cases)
            all_known_keys.update(agg.keys())
            rows_data.append((row.to_dict(), agg))
            total_dropped += len(dropped)
            msg = f"  {code_orig}: {len(agg)} result columns"
            if dropped:
                msg += f"  [{len(dropped)} stale/partial column(s) dropped]"
            print(msg)
            for d in dropped:
                print(f"      dropped: {d}")

    # Reconstruct rows ensuring missing ones get NaNs for the aggregated keys
    all_rows = []
    for base_row, agg in rows_data:
        if agg is None:
            # Create a dict of NaNs for all possible result columns we found
            nan_dict = {k: None for k in all_known_keys}
            all_rows.append({**base_row, **nan_dict})
        else:
            all_rows.append({**base_row, **agg})

    out = pd.DataFrame(all_rows)

    # Compute pps premiums relative to English
    eng_row = out[out["Code_Orig"] == ENGLISH]
    if len(eng_row) == 0:
        raise ValueError(f"English row ({ENGLISH}) not found in CSV")
    
    pps_cols = [c for c in out.columns if c.endswith("_pps")]
    for col in pps_cols:
        eng_pps = eng_row[col].values[0]
        # Only compute the premium if English actually has data for this column
        if pd.notna(eng_pps) and eng_pps != 0:
            out[col.replace("_pps", "_pps_premium")] = (out[col] / eng_pps).round(4)
        else:
            out[col.replace("_pps", "_pps_premium")] = None
            
    print(f"  Added {len(pps_cols)} pps_premium columns")
    if total_dropped:
        print(f"  ({total_dropped} stale/partial column instances dropped across all languages)")

    # Keep original columns first, then result columns sorted
    orig_cols   = list(df.columns)
    result_cols = sorted(c for c in out.columns if c not in orig_cols)
    out = out[orig_cols + result_cols]

    out_dir = os.path.dirname(csv_out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(csv_out_path, index=False)
    print(f"\nSaved → {csv_out_path}  ({len(result_cols)} new columns added)")
    print(f"(input CSV {args.csv_in_path} left unmodified)")


if __name__ == "__main__":
    main()