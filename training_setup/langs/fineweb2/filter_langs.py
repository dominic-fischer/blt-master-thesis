"""
Filter fineweb2-language-distribution.csv down to:
  - rows whose "subset" is one of the languages in langs_chosen.csv
  - split == "train"
  - only the columns: subset, split, name, documents, utf8_bytes, utf8_bytes_human

Also merges documents / utf8_bytes / utf8_bytes_human back into langs_chosen.csv,
matched on language_code (with the cmn_Hans -> cmn_Hani override).

Usage:
    python filter_fineweb2.py

Adjust the paths below to match your repo layout.
"""

import csv

FINEWEB2_CSV = "training_setup/langs/fineweb2/fineweb2_language_distribution.csv"
LANGS_CHOSEN_CSV = "training_setup/langs/langs_chosen.csv"
OUTPUT_CSV = "training_setup/langs/fineweb2/fineweb2_filtered.csv"

KEEP_COLUMNS = ["subset", "split", "name", "documents", "utf8_bytes", "utf8_bytes_human"]
MERGE_COLUMNS = ["documents", "utf8_bytes", "utf8_bytes_human"]  # columns pulled into langs_chosen.csv


def resolve_subset_code(language_code):
    """Map a langs_chosen.csv language_code to its fineweb2 subset code."""
    if language_code == "cmn_Hans":
        return "cmn_Hani"  # override to match fineweb2's config name
    return language_code


def load_chosen_subsets(path):
    """Read the language_code column from langs_chosen.csv into a set of fineweb2 subset codes."""
    subsets = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = resolve_subset_code(row.get("language_code", "").strip())
            if code and code != "n/a":
                subsets.add(code)
    return subsets


def filter_fineweb2(fineweb2_path, chosen_subsets, output_path):
    with open(fineweb2_path, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = [
            {col: row[col] for col in KEEP_COLUMNS}
            for row in reader
            if row["subset"] in chosen_subsets and row["split"] == "train"
        ]

    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=KEEP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def update_langs_chosen(langs_chosen_path, filtered_rows):
    """Write documents / utf8_bytes / utf8_bytes_human back into langs_chosen.csv, keyed on subset code."""
    by_subset = {row["subset"]: row for row in filtered_rows}

    with open(langs_chosen_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        base_fieldnames = reader.fieldnames
        rows = list(reader)

    new_fieldnames = base_fieldnames + [c for c in MERGE_COLUMNS if c not in base_fieldnames]

    for row in rows:
        subset_code = resolve_subset_code(row.get("language_code", "").strip())
        match = by_subset.get(subset_code)
        for col in MERGE_COLUMNS:
            row[col] = match[col] if match else "n/a"

    with open(langs_chosen_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main():
    chosen_subsets = load_chosen_subsets(LANGS_CHOSEN_CSV)
    print(f"Loaded {len(chosen_subsets)} chosen subset codes: {sorted(chosen_subsets)}")

    filtered_rows = filter_fineweb2(FINEWEB2_CSV, chosen_subsets, OUTPUT_CSV)
    print(f"Wrote {len(filtered_rows)} rows to {OUTPUT_CSV}")

    missing = chosen_subsets - {r["subset"] for r in filtered_rows}
    if missing:
        print(f"WARNING: no train row found for: {sorted(missing)}")

    updated_rows = update_langs_chosen(LANGS_CHOSEN_CSV, filtered_rows)
    print(f"Updated {len(updated_rows)} rows in {LANGS_CHOSEN_CSV} with {MERGE_COLUMNS}")


if __name__ == "__main__":
    main()