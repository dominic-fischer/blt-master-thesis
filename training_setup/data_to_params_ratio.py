"""
data_to_params_ratio.py

Computes a single TOTAL training byte count across ALL 20 chosen languages,
using content-equalized allocation capped by the smallest-content language,
then reports the resulting bytes/param ratio for each model size in
model_configs_computed.csv.

Method:
    1. Find the smallest-content language by RATIO-ADJUSTED bytes
       (utf8_bytes / ratio_vs_english) -- this is Chichewa (nya_Latn), per
       the actual numbers in langs_chosen.csv.
    2. Reserve --val-bytes (raw bytes) out of that language's total, giving
       its usable training bytes.
    3. Convert that to a language-agnostic "content budget", in
       English-equivalent bytes, by dividing by ITS OWN ratio_vs_english:
           content_budget = chichewa_usable_bytes / chichewa_ratio
    4. For every one of the 20 languages (including English, ratio=1.0),
       allocate raw_bytes = content_budget * that_language's_ratio_vs_english.
       This gives every language the SAME AMOUNT OF ACTUAL CONTENT, at a
       raw byte cost that varies by how many bytes that language's script/
       encoding needs per unit of content -- rather than giving every
       language the same raw byte count (which would be an unequal amount
       of real content across languages).
    5. Sum the 20 per-language raw byte allocations. THIS total (not just
       Chichewa's own bytes) is the actual total training corpus size fed
       to the model across all languages combined, so it's what the
       bytes/param ratio should be computed against.

Why this matters: the model reads raw bytes regardless of language, so
total compute/steps depend on the SUM of raw bytes across all 20 languages
-- but the allocation across languages needs to be content-equalized
(step 4) to be a fair comparison, rather than naively giving every
language equal raw bytes.

Usage:
    python data_to_params_ratio.py
    python data_to_params_ratio.py --val-bytes 5000000
    python data_to_params_ratio.py --langs-csv training_setup/langs/langs_chosen.csv \
                                    --configs-csv training_setup/model_configs_computed.csv
"""

import argparse
import csv


def load_all_languages(langs_csv):
    """Returns a list of {language, language_code, utf8_bytes (or None for
    English/unmeasured), ratio_vs_english} for every row in langs_chosen.csv."""
    rows = []
    with open(langs_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row.get("utf8_bytes", "n/a")
            n_bytes = int(raw) if raw not in ("n/a", "", None) else None
            rows.append({
                "language": row["Language"],
                "language_code": row["language_code"],
                "utf8_bytes": n_bytes,
                "ratio_vs_english": float(row["ratio_vs_english"]),
            })
    return rows


def find_smallest_by_adjusted_bytes(langs):
    """Ranks by RATIO-ADJUSTED bytes (utf8_bytes / ratio_vs_english), among
    languages that actually have a measured utf8_bytes value."""
    candidates = [
        {**row, "adjusted_bytes": row["utf8_bytes"] / row["ratio_vs_english"]}
        for row in langs
        if row["utf8_bytes"] is not None
    ]
    if not candidates:
        raise ValueError("No languages with a measured utf8_bytes value found")
    return min(candidates, key=lambda r: r["adjusted_bytes"])


def compute_total_training_bytes(langs, smallest, val_bytes):
    """Content-equalized allocation across ALL languages (including English),
    capped by the smallest-content language's usable bytes. Returns
    (total_training_bytes, content_budget, per_language_allocations)."""
    usable_bytes = smallest["utf8_bytes"] - val_bytes
    if usable_bytes <= 0:
        raise ValueError(
            f"--val-bytes ({val_bytes:,}) >= smallest language's total "
            f"({smallest['utf8_bytes']:,}). Nothing left for training."
        )
    content_budget = usable_bytes / smallest["ratio_vs_english"]

    allocations = []
    total = 0
    for row in langs:
        allocated = content_budget * row["ratio_vs_english"]
        allocations.append({
            "language": row["language"],
            "language_code": row["language_code"],
            "ratio_vs_english": row["ratio_vs_english"],
            "allocated_bytes": allocated,
        })
        total += allocated

    return total, content_budget, allocations


def load_model_configs(configs_csv):
    configs = []
    skipped = []
    with open(configs_csv, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):  # start=2: row 1 is the header
            raw = row.get("total_params", "")
            cleaned = raw.replace(",", "").strip()
            if not cleaned.lstrip("-").isdigit():
                skipped.append((i, row.get("config_name", "?"), raw))
                continue
            configs.append({
                "config_name": row["config_name"],
                "total_params": int(cleaned),
            })

    if skipped:
        print(f"WARNING: skipped {len(skipped)} row(s) with non-numeric total_params "
              f"(likely a stray header/annotation row in the CSV):")
        for line_no, name, raw in skipped:
            print(f"  line {line_no}: config_name={name!r} total_params={raw!r}")
        print()

    if not configs:
        raise ValueError(f"No valid rows with numeric total_params found in {configs_csv}")

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs-csv", default="training_setup/langs/langs_chosen.csv")
    parser.add_argument("--configs-csv", default="training_setup/model_configs_computed.csv")
    parser.add_argument("--val-bytes", type=int, default=5_000_000,
                         help="Bytes reserved for validation, subtracted from "
                              "the smallest-content language's total before "
                              "computing the content budget. Default: 5,000,000.")
    parser.add_argument("--show-allocations", action="store_true",
                         help="Print the per-language raw byte allocation breakdown.")
    args = parser.parse_args()

    langs = load_all_languages(args.langs_csv)
    smallest = find_smallest_by_adjusted_bytes(langs)

    print(f"Smallest language (by RATIO-ADJUSTED bytes): {smallest['language']} ({smallest['language_code']})")
    print(f"  ratio_vs_english:     {smallest['ratio_vs_english']:.4f}")
    print(f"  Raw utf8_bytes:       {smallest['utf8_bytes']:,}")
    print(f"  Minus validation:     -{args.val_bytes:,}")
    print()

    total_training_bytes, content_budget, allocations = compute_total_training_bytes(
        langs, smallest, args.val_bytes
    )

    print(f"{'Language':<20}{'ratio_vs_english':>18}{'raw bytes allocated':>22}")
    print("-" * 60)
    for a in allocations:
        print(f"{a['language']:<20}{a['ratio_vs_english']:>18.4f}{a['allocated_bytes']:>22,.0f}")
    print("-" * 60)
    print(f"{'TOTAL (' + str(len(langs)) + ' languages)':<38}{total_training_bytes:>22,.0f}")
    print(f"  = {total_training_bytes/1024**2:,.2f} MiB = {total_training_bytes/1024**3:,.4f} GiB "
          f"(ONE epoch, unique bytes)")
    print()

    configs = load_model_configs(args.configs_csv)

    print(f"{'Size':<15}{'total_params':>15}{'total_bytes/param':>20}  note")
    print("-" * 95)
    for c in configs:
        ratio = total_training_bytes / c["total_params"]
        flag = ""
        if ratio < 20:
            flag = "likely severely undertrained / heavy repetition needed"
        elif ratio < 80:
            flag = "below rough Chinchilla-equivalent byte target (~80-100)"
        elif ratio <= 100:
            flag = "in the ~80-100 target range"
        else:
            flag = "above ~100 -- likely more data than strictly needed for this size"
        print(f"{c['config_name']:<15}{c['total_params']:>15,}{ratio:>20,.2f}  {flag}")


if __name__ == "__main__":
    main()