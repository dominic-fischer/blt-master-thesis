"""
lang_data_ratios_imbalanced.py

Sibling to data_to_params_ratio.py, but for the IMBALANCED case: instead of
capping every language at a content-equalized allocation (bounded by the
smallest-content language, Chichewa), this uses each language's actual
RAW available utf8_bytes, in its natural proportions -- i.e. what training
would look like if every language just used all the data it actually has,
with no equalization across languages.

Three things this computes:

  1. share_of_total: each language's raw utf8_bytes as a % of the sum
     across all measured languages.

  2. ratio_vs_english_imbalanced: ratio of each language's raw utf8_bytes to a baseline
     language's raw utf8_bytes (--baseline english or --baseline chichewa).
     This is what you'd plug into bytelatent's data.sources: {lang: weight}
     for a genuinely IMBALANCED (proportional-to-availability) training
     run, as opposed to the balanced script's ratio_vs_english-based
     weights. NOTE: mathematically, it does not matter which baseline you
     pick -- SamplingIterator renormalizes weights by dividing by their
     sum before drawing, so any constant rescaling (which is all switching
     baseline does) cancels out and produces IDENTICAL sampling behavior.
     Defaults to english purely because it's already your established
     reference point elsewhere in this project.

  3. imbalanced_allocation_bytes: every language's raw utf8_bytes scaled
     DOWN by the same constant factor so the new total exactly matches
     --target-total (default: 11,146,890,535, the balanced corpus's total
     from data_to_params_ratio.py) -- i.e. same total training bytes as
     the balanced condition, but keeping the natural (imbalanced) shape
     instead of equalizing content across languages. This lets you compare
     "balanced" vs "imbalanced" training at an identical total compute
     budget, varying only how that budget is distributed across languages.

     Given the actual numbers, this is a stark illustration of what
     imbalance really means: Chichewa's imbalanced_allocation_bytes comes
     out to roughly 51,472 bytes (~50 KiB) -- essentially nothing, versus
     its full ~413MB balanced allocation.

Usage:
    python lang_data_ratios_imbalanced.py
    python lang_data_ratios_imbalanced.py --baseline chichewa
    python lang_data_ratios_imbalanced.py --target-total 11146890535
    python lang_data_ratios_imbalanced.py --no-write
"""

import argparse
import csv


def load_all_languages(langs_csv):
    """Returns a list of {language, language_code, utf8_bytes (or None for
    English/unmeasured)} for every row in langs_chosen.csv."""
    rows = []
    with open(langs_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row.get("utf8_bytes", "n/a")
            n_bytes = int(raw) if raw not in ("n/a", "", None) else None
            rows.append({
                "language": row["Language"],
                "language_code": row["language_code"],
                "utf8_bytes": n_bytes,
            })
    return rows


def write_columns_to_langs_csv(langs_csv, rows_by_code, columns):
    """
    Writes `columns` (a list of column names, each already present as a
    key in each dict in rows_by_code) into langs_chosen.csv IN PLACE --
    only those columns are added/overwritten; every other row/column is
    preserved exactly as read. Matched on language_code, same pattern as
    the other langs_chosen.csv-updating scripts.
    """
    with open(langs_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for col in columns:
        if col not in fieldnames:
            fieldnames.append(col)

    updated = 0
    for row in rows:
        code = row.get("language_code")
        if code in rows_by_code:
            for col in columns:
                row[col] = rows_by_code[code][col]
            updated += 1

    with open(langs_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs-csv", default="training_setup/langs/langs_chosen.csv")
    parser.add_argument("--baseline", choices=["english", "chichewa"], default="english",
                         help="Baseline for the ratio_vs_english_imbalanced ratio column. "
                              "Mathematically doesn't affect sampling behavior "
                              "(SamplingIterator renormalizes), purely for "
                              "readability. Default: english.")
    parser.add_argument("--target-total", type=int, default=11_146_890_535,
                         help="Total bytes to downsize the imbalanced corpus "
                              "to, preserving natural proportions. Default: "
                              "11,146,890,535 (the balanced corpus's total "
                              "from data_to_params_ratio.py), so both "
                              "conditions use the identical total training "
                              "budget.")
    parser.add_argument("--no-write", action="store_true",
                         help="Only print results; don't touch --langs-csv.")
    args = parser.parse_args()

    langs = load_all_languages(args.langs_csv)

    measured = [row for row in langs if row["utf8_bytes"] is not None]
    measured.sort(key=lambda r: r["utf8_bytes"], reverse=True)

    skipped = [row["language"] for row in langs if row["utf8_bytes"] is None]
    if skipped:
        print(f"Skipping (no measured utf8_bytes yet): {skipped}")
        print()

    total_raw_bytes = sum(row["utf8_bytes"] for row in measured)

    baseline_code = "eng_Latn" if args.baseline == "english" else "nya_Latn"
    baseline_row = next((r for r in measured if r["language_code"] == baseline_code), None)
    if baseline_row is None:
        raise ValueError(f"Baseline language {baseline_code!r} not found among measured languages")
    baseline_bytes = baseline_row["utf8_bytes"]

    scale = args.target_total / total_raw_bytes

    for row in measured:
        row["share_of_total"] = row["utf8_bytes"] / total_raw_bytes * 100
        row["ratio_vs_english_imbalanced"] = row["utf8_bytes"] / baseline_bytes
        row["imbalanced_allocation_bytes"] = round(row["utf8_bytes"] * scale)

    total_weight = sum(row["ratio_vs_english_imbalanced"] for row in measured)
    for row in measured:
        norm_weight = row["ratio_vs_english_imbalanced"] / total_weight
        # Expected number of draws before this source is picked once, under
        # i.i.d. weighted sampling -- makes the practical (not just
        # mathematical) reachability of small weights concrete.
        row["expected_draws_to_sample_once"] = 1 / norm_weight if norm_weight > 0 else float("inf")

    print(f"ratio_vs_english_imbalanced baseline: {args.baseline} ({baseline_code}, {baseline_bytes:,} bytes) "
          f"-- NOTE: choice of baseline doesn't change actual sampling behavior, "
          f"SamplingIterator renormalizes weights before drawing")
    print(f"Downsizing to target_total={args.target_total:,} (scale factor = {scale:.8f})")
    print()
    print("WARNING: ratio_vs_english_imbalanced values are printed in scientific notation "
          "deliberately -- a fixed-decimal format (e.g. .4f) rounds small "
          "weights like Chichewa's to a displayed 0.0000, which is NOT the "
          "same as the stored value. If you ever hand-copy a weight from "
          "this table into a yaml sources: block, copying a rounded 0.0000 "
          "makes that language's sampling probability a PERMANENT, exact "
          "zero (SamplingIterator's np.random.choice will never select it, "
          "not just rarely) -- categorically different from a tiny but "
          "nonzero weight, which is merely improbable. Always read weights "
          "off the CSV column (full float precision) or this table's "
          "scientific-notation display, never a rounded decimal.")
    print()

    print(f"{'Language':<20}{'share_of_total':>16}{'source_weight':>16}"
          f"{'raw utf8_bytes':>22}{'imbalanced_allocation_bytes':>30}{'expected_draws_to_sample_once':>32}")
    print("-" * 136)
    total_imbalanced = 0
    for row in measured:
        draws = row["expected_draws_to_sample_once"]
        draws_str = f"{draws:,.0f}" if draws != float("inf") else "inf"
        print(f"{row['language']:<20}{row['share_of_total']:>15.2f}%"
              f"{row['ratio_vs_english_imbalanced']:>16.6e}{row['utf8_bytes']:>22,}"
              f"{row['imbalanced_allocation_bytes']:>30,}{draws_str:>32}")
        total_imbalanced += row["imbalanced_allocation_bytes"]
    print("-" * 136)
    print(f"{'TOTAL (' + str(len(measured)) + ' languages)':<52}{total_raw_bytes:>22,}"
          f"{total_imbalanced:>30,}")
    print(f"  raw total    = {total_raw_bytes/1024**3:,.2f} GiB = {total_raw_bytes/1024**4:,.4f} TiB")
    print(f"  imbalanced total (downsized, should ~= target_total) = {total_imbalanced:,}")

    if args.no_write:
        print(f"\n(--no-write set: NOT writing to {args.langs_csv})")
    else:
        rows_by_code = {row["language_code"]: row for row in measured}
        updated = write_columns_to_langs_csv(
            args.langs_csv, rows_by_code,
            columns=["ratio_vs_english_imbalanced", "imbalanced_allocation_bytes"],
        )
        print(f"\n ratio_vs_english_imbalanced / imbalanced_allocation_bytes "
              f"to {updated} row(s) in {args.langs_csv}")


if __name__ == "__main__":
    main()