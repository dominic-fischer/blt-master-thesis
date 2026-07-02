"""
For each of the 20 selected languages (training_setup/langs/langs_chosen.csv,
column FLORES_Code), check how many bytes are actually available in FineWeb2.

Streams through each language's FineWeb2 config and sums UTF-8 byte lengths
of the "text" field, stopping early once a cutoff is hit (default 100M
bytes) - past that point we almost certainly have "enough" and don't need
to burn time/bandwidth counting further.

If a language's FineWeb2 subset is exhausted *before* hitting the cutoff,
that's flagged clearly - those are your bottleneck languages that will cap
how much data you can use across the whole balanced condition.

Requires: pip install datasets --break-system-packages
Run locally (needs internet access to huggingface.co).
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

FINEWEB2_REPO = "HuggingFaceFW/fineweb-2"
FINEWEB_REPO = "HuggingFaceFW/fineweb"  # English lives here, not in fineweb-2
CHOSEN_LANGS_CSV = Path("training_setup/langs/langs_chosen.csv")
OUTPUT_CSV = Path("training_setup/langs/fineweb2_available_bytes.csv")

CUTOFF_BYTES = 100_000_000  # 100M byte cap - stop counting once we hit this

# FLORES+ codes that don't map 1:1 onto FineWeb2 config names.
# FineWeb2 uses cmn_Hani for Mandarin, not zho_Hans (the FLORES+ code).
CODE_OVERRIDES = {
    "cmn_Hans": "cmn_Hani",
}

ENGLISH_CODE = "eng_Latn"


def resolve_fineweb2_config(flores_code):
    """Map a FLORES+ code to the actual FineWeb2 config name, if they differ."""
    return CODE_OVERRIDES.get(flores_code, flores_code)


def count_available_bytes(repo, config_name, cutoff=CUTOFF_BYTES):
    """
    Streams through a dataset (fineweb or fineweb-2), summing UTF-8 byte
    length of the 'text' field. Stops early if `cutoff` is reached.
    Returns (total_bytes, n_docs, hit_cutoff).
    """
    if config_name is None:
        ds = load_dataset(repo, split="train", streaming=True)
    else:
        ds = load_dataset(repo, name=config_name, split="train", streaming=True)

    total_bytes = 0
    n_docs = 0
    hit_cutoff = False

    for row in ds:
        n_bytes = len(row["text"].encode("utf-8"))
        total_bytes += n_bytes
        n_docs += 1

        if total_bytes >= cutoff:
            hit_cutoff = True
            break

    return total_bytes, n_docs, hit_cutoff


def main():
    if not CHOSEN_LANGS_CSV.exists():
        raise FileNotFoundError(f"Not found: {CHOSEN_LANGS_CSV.resolve()}")

    chosen = pd.read_csv(CHOSEN_LANGS_CSV)
    if "FLORES_Code" not in chosen.columns:
        raise ValueError(f"Expected column 'FLORES_Code' in {CHOSEN_LANGS_CSV}, "
                          f"found columns: {list(chosen.columns)}")

    codes = chosen["FLORES_Code"].dropna().unique().tolist()
    print(f"Checking FineWeb2 availability for {len(codes)} languages "
          f"(cutoff = {CUTOFF_BYTES:,} bytes)\n")

    rows = []
    for code in codes:
        print(f"--- {code} ---")

        if code == ENGLISH_CODE:
            repo, config_name = FINEWEB_REPO, None
            print(f"  (English: routing to {FINEWEB_REPO} instead of fineweb-2)")
        else:
            resolved = resolve_fineweb2_config(code)
            if resolved != code:
                print(f"  (mapped {code} -> {resolved} for FineWeb2)")
            repo, config_name = FINEWEB2_REPO, resolved

        try:
            total_bytes, n_docs, hit_cutoff = count_available_bytes(repo, config_name)
        except Exception as e:
            print(f"  [ERROR] {code}: {e}")
            rows.append({
                "FLORES_Code": code,
                "fineweb_repo": repo,
                "fineweb_config": config_name,
                "available_bytes": None,
                "n_docs_scanned": None,
                "hit_cutoff": None,
                "status": f"ERROR: {e}",
            })
            continue

        status = "OK (hit cutoff)" if hit_cutoff else "BOTTLENECK (exhausted before cutoff)"
        print(f"  bytes={total_bytes:,}  docs={n_docs:,}  {status}")

        rows.append({
            "FLORES_Code": code,
            "fineweb_repo": repo,
            "fineweb_config": config_name,
            "available_bytes": total_bytes,
            "n_docs_scanned": n_docs,
            "hit_cutoff": hit_cutoff,
            "status": status,
        })

    df = pd.DataFrame(rows).sort_values("available_bytes", ascending=True, na_position="first")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved to {OUTPUT_CSV}\n")

    bottlenecks = df[df["hit_cutoff"] == False]
    if not bottlenecks.empty:
        print("=== BOTTLENECK LANGUAGES (fewer than cutoff bytes available) ===")
        for _, row in bottlenecks.iterrows():
            print(f"  {row['FLORES_Code']:20s} only {row['available_bytes']:,} bytes available")
        print(f"\nYour balanced-condition per-language budget is capped by the smallest "
              f"of these: {bottlenecks['available_bytes'].min():,} bytes.")
    else:
        print("All languages have at least the cutoff amount available.")

    return df


if __name__ == "__main__":
    main()