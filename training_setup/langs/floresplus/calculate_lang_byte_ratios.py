"""
Count total bytes per language from results/restructured/*.json and compute
each language's byte ratio relative to English, all in a single CSV.

Expects files named like:
    eng_Latn.json
    hin_Deva.json
    ...
Each file is a JSON list of dicts, each with a "bytes_entropies" key (a list).

total_bytes(L) = sum of len(entry["bytes_entropies"]) across all entries in L's file
ratio_vs_english(L) = total_bytes(L) / total_bytes(English)

Asserts every file has exactly 997 entries (expected FLORES+ devtest size) -
fails loudly if any language file is missing sentences, since that would
silently skew the byte ratio.
"""

import json
import re
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results/base_model")
OUTPUT_CSV = Path("training_setup/langs/floresplus/byte_counts_and_ratios.csv")

REFERENCE_CODE = "eng_Latn"  # adjust if your English file uses a different code
EXPECTED_N_ENTRIES = 997

FILENAME_PATTERN = re.compile(r"^(?P<code>.+)\.json$")


def count_bytes_in_file(path: Path) -> tuple[int, int]:
    """Returns (total_bytes, n_entries) for a single language file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON list at the top level")

    total_bytes = 0
    n_entries = 0
    n_missing_key = 0

    for entry in data:
        if "bytes_entropies" not in entry:
            n_missing_key += 1
            continue
        total_bytes += len(entry["bytes_entropies"])
        n_entries += 1

    if n_missing_key > 0:
        print(f"  [warning] {path.name}: {n_missing_key} entries missing 'bytes_entropies' key")

    return total_bytes, n_entries


def main():
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {RESULTS_DIR.resolve()}")

    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files found in {RESULTS_DIR.resolve()}")

    rows = []
    mismatches = []

    for path in files:
        match = FILENAME_PATTERN.match(path.name)
        code = match.group("code") if match else path.stem

        total_bytes, n_entries = count_bytes_in_file(path)

        if n_entries != EXPECTED_N_ENTRIES:
            mismatches.append((code, n_entries))

        rows.append({
            "language_code": code,
            "filename": path.name,
            "n_entries": n_entries,
            "total_bytes": total_bytes,
        })
        print(f"{code:30s} n_entries={n_entries:6,d}  total_bytes={total_bytes:12,d}")

    # Fail loudly if any file doesn't have exactly 997 entries -
    # a silent mismatch here would skew every ratio downstream.
    if mismatches:
        msg = "\n".join(f"  {code}: {n} entries (expected {EXPECTED_N_ENTRIES})" for code, n in mismatches)
        raise AssertionError(
            f"{len(mismatches)} file(s) do not have exactly {EXPECTED_N_ENTRIES} entries:\n{msg}"
        )
    print(f"\n[OK] All {len(rows)} files have exactly {EXPECTED_N_ENTRIES} entries.")

    df = pd.DataFrame(rows)

    ref_rows = df[df["language_code"] == REFERENCE_CODE]
    if ref_rows.empty:
        raise ValueError(
            f"Reference language '{REFERENCE_CODE}' not found. "
            f"Available codes include: {sorted(df['language_code'].unique())[:10]}..."
        )
    ref_bytes = ref_rows.iloc[0]["total_bytes"]
    if ref_bytes == 0:
        raise ValueError(f"Reference language '{REFERENCE_CODE}' has 0 total_bytes - cannot compute ratios")

    df["ratio_vs_english"] = df["total_bytes"] / ref_bytes
    df = df.sort_values("ratio_vs_english", ascending=False).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nReference: {REFERENCE_CODE} ({ref_bytes:,} bytes)")
    for _, row in df.iterrows():
        marker = "  <- reference" if row["language_code"] == REFERENCE_CODE else ""
        print(f"{row['language_code']:30s} ratio={row['ratio_vs_english']:.4f}  "
              f"total_bytes={row['total_bytes']:12,d}{marker}")

    print(f"\nSaved to {OUTPUT_CSV}")
    return df


if __name__ == "__main__":
    main()