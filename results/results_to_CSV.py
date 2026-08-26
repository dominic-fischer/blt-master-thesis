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

The INPUT csv is left exactly as-is; the new columns are written to a
SEPARATE output csv (--csv-out-path), which defaults to
<csv-in-path stem>_with_results.csv if not given explicitly, so the
input file is never silently overwritten.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

ENGLISH = "eng_Latn"
SCORE_SOURCES = ("bytes", "chars")
EVAL_MODES_KEY = {"bytes": "eval_modes", "chars": "char_eval_modes"}
COLUMN_PREFIX = {"bytes": "", "chars": "char_"}


def default_csv_out_path(csv_in_path: str) -> str:
    """<name>.csv -> <name>_with_results.csv"""
    p = Path(csv_in_path)
    return str(p.with_name(f"{p.stem}_with_results{p.suffix}"))


def load_results(results_dir: str, code_orig: str) -> list[dict] | None:
    path = Path(results_dir) / f"{code_orig}.json"
    if not path.exists():
        # Instead of raising an error, we return None to signal a missing file
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def aggregate(sentences: list[dict], source: str) -> dict[str, float]:
    """Compute mean pps and global bpp (total_bytes / total_patches) for
    every case/threshold, reading from the source-appropriate eval_modes
    key. Columns are prefixed per COLUMN_PREFIX (empty for bytes)."""
    modes_key = EVAL_MODES_KEY[source]
    prefix = COLUMN_PREFIX[source]
    totals = {}  # key -> [total_patches, total_bytes, n_sentences]

    for s in sentences:
        for case_name, thresholds in s.get(modes_key, {}).items():
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
    for col, (total_patches, total_bytes, n_sentences) in totals.items():
        result[f"{col}_pps"] = round(total_patches / n_sentences, 4)
        result[f"{col}_bpp"] = round(total_bytes / total_patches, 4)
    return result


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
    return parser.parse_args()


def main():
    args = parse_args()
    csv_out_path = args.csv_out_path or default_csv_out_path(args.csv_in_path)
    source = args.score_source

    df = pd.read_csv(args.csv_in_path)
    print(f"Loaded {len(df)} languages from {args.csv_in_path}")
    print(f"Score source: {source}  (reading sentence['{EVAL_MODES_KEY[source]}'])")

    # Keep track of all keys seen across valid files so we can fill missing ones with NaN
    all_known_keys = set()
    rows_data = []

    for _, row in df.iterrows():
        code_orig = row["Code_Orig"]
        sentences = load_results(args.results_dir, code_orig)
        
        if sentences is None:
            # File is missing. We save the baseline row and fill the rest later
            rows_data.append((row.to_dict(), None))
            print(f"  {code_orig}: Missing JSON -> filling with N/A")
        else:
            agg = aggregate(sentences, source)
            all_known_keys.update(agg.keys())
            rows_data.append((row.to_dict(), agg))
            print(f"  {code_orig}: {len(agg)} result columns")

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