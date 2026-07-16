"""
results_to_txt_premiums.py

For each *_pps_premium column in --csv-in-path, writes a plain-text list
of premiums sorted DESCENDING, restricted to the languages in
--langs-csv's "language_code" column (the 20 languages actually trained
on -- equivalent to the master CSV's "Code_Orig").

Output:
    results/txt_premiums/<mode>/[<prefix>_]premiums_sorted.txt
where <mode> is derived from the column name by stripping the trailing
"_pps_premium" suffix (e.g. column "raw_monotonicity_t_0.0366_pps_premium"
-> mode "raw_monotonicity_t_0.0366"). <prefix>, if given via
--filename-prefix (e.g. a model/checkpoint stem), keeps different runs'
outputs from silently overwriting each other in the same mode folder --
omit it for a simple "premiums_sorted.txt" per mode.

Usage:
    python results/results_to_txt_premiums.py --csv-in-path results/results_CSV/<stem>_results.csv
    python results/results_to_txt_premiums.py --csv-in-path <csv> --langs-csv training_setup/langs/langs_chosen.csv --filename-prefix <stem>
"""
import argparse
import os

import pandas as pd

DEFAULT_LANGS_CSV = "training_setup/langs/langs_chosen.csv"
DEFAULT_OUT_DIR = "results/txt_premiums"
PREMIUM_SUFFIX = "_pps_premium"


def load_chosen_languages(langs_csv: str) -> set[str]:
    df = pd.read_csv(langs_csv)
    return set(df["language_code"].astype(str))


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv-in-path", required=True,
                         help="CSV produced by results_to_CSV.py, containing "
                              "*_pps_premium columns and a 'Code_Orig' column.")
    parser.add_argument("--langs-csv", default=DEFAULT_LANGS_CSV,
                         help=f"CSV whose 'language_code' column restricts which "
                              f"languages are included (default {DEFAULT_LANGS_CSV}).")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                         help=f"Base output directory (default {DEFAULT_OUT_DIR}); "
                              f"one subfolder per mode is created under it.")
    parser.add_argument("--filename-prefix", default=None,
                         help="Optional prefix (e.g. a model/checkpoint stem) for the "
                              "output filename, so different runs don't overwrite each "
                              "other's premiums_sorted.txt in the same mode folder.")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv_in_path)
    chosen_langs = load_chosen_languages(args.langs_csv)

    if "Code_Orig" not in df.columns:
        raise ValueError(f"'Code_Orig' column not found in {args.csv_in_path}")

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

    filename = f"{args.filename_prefix}_premiums_sorted.txt" if args.filename_prefix else "premiums_sorted.txt"

    for col in premium_cols:
        mode = col[:-len(PREMIUM_SUFFIX)]
        mode_dir = os.path.join(args.out_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        rows = filtered[["Code_Orig", col]].dropna().sort_values(col, ascending=False)
        out_path = os.path.join(mode_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in rows.iterrows():
                f.write(f"{row['Code_Orig']}\t{row[col]:.4f}\n")

        print(f"  {mode}: wrote {len(rows)} language(s) -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()