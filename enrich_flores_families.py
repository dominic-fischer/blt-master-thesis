"""
enrich_flores_families.py

Adds language family information to a FLORES+ language list CSV using Glottolog.

Usage:
    python enrich_flores_families.py input.csv output.csv

Input CSV must have at least a 'Glottocode' column.
Expected format (FLORES+ standard):
    Code,Script,Glottocode,Name,Notes

Output adds:
    Family          - top-level family (e.g. Indo-European)
    Subfamily_1     - first subbranch (e.g. Germanic)
    Subfamily_2     - second subbranch (e.g. North Germanic)
    Full_Classification - full lineage as human-readable names joined by ' > '

Requirements: pip install pandas
"""

# cat,Latn,vale1252,Valencian,devtestonly
# kaa,Latn,kara1467,Karakalpak,devtestonly
# khk,Mong,halh1238,Halh Mongolian (traditional Mongolian script),devtestonly
# udm,Cyrl,udmu1245,Udmurt,dev only
# these should be ignored

ignore_glottocodes = {"udmu1245"}


import sys
import pandas as pd
import urllib.request

GLOTTOLOG_LANGUAGES = (
    "https://raw.githubusercontent.com/glottolog/glottolog-cldf/master/cldf/languages.csv"
)
GLOTTOLOG_VALUES = (
    "https://raw.githubusercontent.com/glottolog/glottolog-cldf/master/cldf/values.csv"
)


def download_glottolog():
    print("Downloading Glottolog languages.csv...")
    glottolog = pd.read_csv(urllib.request.urlopen(GLOTTOLOG_LANGUAGES))
    print(f"  {len(glottolog)} languoids loaded")

    print("Downloading Glottolog values.csv (large, ~100MB, please wait)...")
    content = urllib.request.urlopen(GLOTTOLOG_VALUES).read().decode("utf-8")

    print("  Parsing classification paths...")
    class_rows = {}
    for line in content.split("\n")[1:]:
        if ",classification," not in line:
            continue
        parts = line.split(",")
        if len(parts) >= 4 and parts[3]:
            class_rows[parts[1]] = parts[3]

    print(f"  {len(class_rows)} classification paths found")
    return glottolog, class_rows


def expand_path(glottocode, class_rows, id_to_name):
    path_str = class_rows.get(glottocode, "")
    if not path_str:
        return "", "", "", ""

    parts = path_str.split("/")
    names = [id_to_name.get(p, p) for p in parts]

    family = names[0] if len(names) > 0 else ""
    sub1   = names[1] if len(names) > 1 else ""
    sub2   = names[2] if len(names) > 2 else ""
    full   = " > ".join(names)

    return family, sub1, sub2, full


def main(input_path, output_path):
    flores_df = pd.read_csv(input_path)
    if "Glottocode" not in flores_df.columns:
        raise ValueError("Input CSV must have a 'Glottocode' column")

    glottolog, class_rows = download_glottolog()
    id_to_name = dict(zip(glottolog["ID"], glottolog["Name"]))

    print(f"\nEnriching {len(flores_df)} rows...")
    results = []
    for _, row in flores_df.iterrows():
        gc = row["Glottocode"]
        if gc in ignore_glottocodes or row["Notes"] == "devtestonly":
            print(f"Skipping {gc} ({row['Name']}) due to ignore list or devtestonly note")
            continue
        family, sub1, sub2, full = expand_path(gc, class_rows, id_to_name)
        results.append({
            **row.to_dict(),
            "Family": family,
            "Subfamily_1": sub1,
            "Subfamily_2": sub2,
            "Full_Classification": full,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    print("\nFamily distribution:")
    print(out_df["Family"].value_counts().to_string())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python enrich_flores_families.py input.csv output.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])