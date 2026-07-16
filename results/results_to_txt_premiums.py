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

Output:
    results/txt_premiums/t_anchor/<subfolder>/[<step_subfolder>/][<prefix>_]<case>_t_<value>_premiums_sorted.txt
    results/txt_premiums/t_lower_bound/<subfolder>/[<step_subfolder>/][<prefix>_]<case>_t_<value>_premiums_sorted.txt
    results/txt_premiums/t_midpoint/<subfolder>/[<step_subfolder>/][<prefix>_]<case>_t_<value>_premiums_sorted.txt
    results/txt_premiums/t_upper_bound/<subfolder>/[<step_subfolder>/][<prefix>_]<case>_t_<value>_premiums_sorted.txt

where <subfolder> and <prefix> are determined by --filename-prefix if specified,
otherwise automatically parsed as the portion of the CSV filename preceding the 
second underscore. If "step_X" is present in the CSV filename, it is nested inside 
an extra step subfolder.

Each file contains language code, sorted premium, and the absolute patches-per-sentence 
(pps) and bytes-per-patch (bpp) values.

Each file's FINAL LINE records English's own (non-premium) pps/bpp for
that exact case+threshold -- since English's own premium is trivially
1.0 by definition, this instead gives the actual operating point
(patches/sentence, bytes/patch) the premiums above it are relative to.

Usage:
    python results/results_to_txt_premiums.py --csv-in-path results/results_CSV/<stem>_results.csv --summary-csv calibrated_thresholds/<stem>_thresholds_summary.csv
    python results/results_to_txt_premiums.py --csv-in-path <csv> --summary-csv <csv> --langs-csv training_setup/langs/langs_chosen.csv --filename-prefix <stem>
"""
import argparse
import os
import re
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))  # noqa: E402
from model_eval.run_patching import load_cases, threshold_key, BOUND_NAMES, DEFAULT_SUMMARY_CSV

import pandas as pd

DEFAULT_LANGS_CSV = "training_setup/langs/langs_chosen.csv"
DEFAULT_OUT_DIR = "results/txt_premiums"
PREMIUM_SUFFIX = "_pps_premium"
ENGLISH = "eng_Latn"

# Must match run_patching.py's CASES/COMBINED keys.
KNOWN_CASES = ["raw_entropy", "raw_monotonicity", "norm_entropy", "combined"]

# Maps run_patching.py's internal bound name -> the folder name requested.
BOUND_FOLDER_NAMES = {
    "low":    "t_lower_bound",
    "mid":    "t_midpoint",
    "high":   "t_upper_bound",
    "anchor": "t_anchor",
}


def load_chosen_languages(langs_csv: str) -> set[str]:
    df = pd.read_csv(langs_csv)
    return set(df["language_code"].astype(str))


def build_key_to_bound(summary_csv: str) -> dict[tuple[str, str], str]:
    """Returns {(case_name, "t_<value>"): bound_name}, reusing
    run_patching.py's own load_cases()/threshold_key() so this always
    matches exactly how run_patching.py itself keyed its output --
    no separate float-formatting logic to drift out of sync."""
    cases = load_cases(summary_csv)
    key_to_bound = {}
    for case_name, case in cases.items():
        for bound_name in BOUND_NAMES:
            t = case["named_thresholds"][bound_name]
            key_to_bound[(case_name, threshold_key(t))] = bound_name
    return key_to_bound


def parse_premium_column(col: str) -> tuple[str, str] | None:
    """Splits "{case}_{t_key}_pps_premium" into (case, t_key), where
    t_key is the literal "t_<value>" string (e.g. "t_1.3340") -- bound
    identity is looked up separately via build_key_to_bound, not parsed
    out of the column name itself. Returns None if col doesn't end in
    PREMIUM_SUFFIX or doesn't match a known case -- defensive against
    unrelated columns / naming drift."""
    if not col.endswith(PREMIUM_SUFFIX):
        return None
    mode_str = col[: -len(PREMIUM_SUFFIX)]  # "{case}_t_{value}"
    for case in KNOWN_CASES:
        prefix = f"{case}_"
        if mode_str.startswith(prefix):
            t_key = mode_str[len(prefix):]  # "t_1.3340"
            return case, t_key
    return None


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
             f"this checkpoint, used here to look up which numeric "
             f"threshold is which bound (default {DEFAULT_SUMMARY_CSV})."
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
             f"one subfolder per bound type is created under it."
    )
    parser.add_argument(
        "--filename-prefix", 
        default=None,
        help="Optional prefix (e.g. a model/checkpoint stem) for the "
             "output filename and subfolder. If omitted, parsed automatically "
             "from the input CSV's filename before the second underscore."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv_in_path)
    chosen_langs = load_chosen_languages(args.langs_csv)
    key_to_bound = build_key_to_bound(args.summary_csv)

    if "Code_Orig" not in df.columns:
        raise ValueError(f"'Code_Orig' column not found in {args.csv_in_path}")

    csv_basename = os.path.basename(args.csv_in_path)
    csv_stem, _ = os.path.splitext(csv_basename)

    # Check if there is a "step_X" pattern in the filename
    step_match = re.search(r"step_\d+", csv_stem)
    step_subfolder = step_match.group(0) if step_match else None

    # Determine automatic prefix / subfolder name if not explicitly specified
    if args.filename_prefix:
        prefix = args.filename_prefix
        subfolder_name = args.filename_prefix
    else:
        parts = csv_stem.split("_")
        if len(parts) >= 2:
            prefix = f"{parts[0]}_{parts[1]}"
        else:
            prefix = csv_stem
        subfolder_name = prefix

    filtered = df[df["Code_Orig"].astype(str).isin(chosen_langs)]
    missing = chosen_langs - set(filtered["Code_Orig"].astype(str))
    if missing:
        print(f"  NOTE: {len(missing)} language(s) from {args.langs_csv} not found "
              f"in {args.csv_in_path}: {sorted(missing)}")

    premium_cols = [c for c in df.columns if c.endswith(PREMIUM_SUFFIX)]
    if not premium_cols:
        print(f"No *{PREMIUM_SUFFIX} columns found in {args.csv_in_path} -- nothing to write.")
        return

    print(f"Found {len(premium_cols)} premium column(s); "
          f"{len(filtered)}/{len(chosen_langs)} chosen languages present.")

    eng_row = filtered[filtered["Code_Orig"].astype(str) == ENGLISH]
    if len(eng_row) == 0:
        print(f"  NOTE: {ENGLISH} not found in the filtered set -- the English "
              f"reference line will be omitted from every file.")

    for col in premium_cols:
        parsed = parse_premium_column(col)
        if parsed is None:
            print(f"  WARNING: could not parse column {col!r} into (case, t_key) -- "
                  f"skipping (unrecognized case name or naming drift vs run_patching.py).")
            continue
        case, t_key = parsed

        bound_name = key_to_bound.get((case, t_key))
        if bound_name is None:
            print(f"  WARNING: {case}/{t_key} not found in {args.summary_csv}'s calibrated "
                  f"bounds for this case -- skipping (was --summary-csv the same one "
                  f"run_patching.py used for this checkpoint?).")
            continue
        bound_folder = BOUND_FOLDER_NAMES.get(bound_name, f"t_{bound_name}")

        # Construct path hierarchy: out_dir / bound / prefix [/ step_subfolder]
        if step_subfolder:
            out_dir = os.path.join(args.out_dir, bound_folder, subfolder_name, step_subfolder)
        else:
            out_dir = os.path.join(args.out_dir, bound_folder, subfolder_name)
            
        os.makedirs(out_dir, exist_ok=True)

        base_name = f"{case}_{t_key}_premiums_sorted.txt"
        filename = f"{prefix}_{base_name}"
        out_path = os.path.join(out_dir, filename)

        # Retrieve absolute pps and bpp columns alongside premium
        mode_str = f"{case}_{t_key}"
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
            if has_extra_cols:
                pps = f"{row[pps_col]:.4f}"
                bpp = f"{row[bpp_col]:.4f}"
                formatted_rows.append((lang, prem, pps, bpp))
            else:
                formatted_rows.append((lang, prem))

        # Dynamic alignment: compute max width of each column (including safety margin of 2 spaces)
        headers = ["Language", "Premium", "PPS", "BPP"] if has_extra_cols else ["Language", "Premium"]
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