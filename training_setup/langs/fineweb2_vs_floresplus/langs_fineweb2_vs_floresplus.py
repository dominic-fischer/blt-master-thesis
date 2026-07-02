"""
FineWeb2 <-> FLORES+ language comparison + download helper.

Requires (pip install):
    huggingface_hub
    datasets
    pandas

Run locally (needs internet access to huggingface.co).
"""

import re
import pandas as pd
from huggingface_hub import HfApi

FINEWEB2_REPO = "HuggingFaceFW/fineweb-2"
FLORES_CSV = "floresplus_MASTER_CSV.csv"   # adjust path if needed
FLORES_CODE_COL = "Code_Orig"


# ---------------------------------------------------------------------------
# 1. Get all FineWeb2 language configs (subsets are named like "eng_Latn")
# ---------------------------------------------------------------------------
def get_fineweb2_languages():
    api = HfApi()
    info = api.dataset_info(FINEWEB2_REPO)

    # Language/script codes show up as top-level folders in the repo,
    # e.g. "data/eng_Latn/train/000_00000.parquet"
    pattern = re.compile(r"^data/([a-z]{3}_[A-Za-z]{4})/")
    langs = set()
    for sibling in info.siblings:
        m = pattern.match(sibling.rfilename)
        if m:
            langs.add(m.group(1))

    return sorted(langs)


# ---------------------------------------------------------------------------
# 2. Load FLORES+ codes from your CSV
# ---------------------------------------------------------------------------
def get_floresplus_codes():
    df = pd.read_csv(FLORES_CSV)
    return sorted(set(df[FLORES_CODE_COL].dropna().astype(str)))


# ---------------------------------------------------------------------------
# 3. Compare
# ---------------------------------------------------------------------------
def compare(fineweb2_langs, flores_codes):
    fw2_set = set(fineweb2_langs)
    flores_set = set(flores_codes)

    both = sorted(fw2_set & flores_set)
    only_fw2 = sorted(fw2_set - flores_set)
    only_flores = sorted(flores_set - fw2_set)

    print(f"FineWeb2 languages:   {len(fw2_set)}")
    print(f"FLORES+ languages:    {len(flores_set)}")
    print(f"In both:              {len(both)}")
    print(f"Only in FineWeb2:     {len(only_fw2)}")
    print(f"Only in FLORES+:      {len(only_flores)}")

    return {
        "both": both,
        "only_fineweb2": only_fw2,
        "only_floresplus": only_flores,
    }


# ---------------------------------------------------------------------------
# 4. Download a specific FineWeb2 language subset
# ---------------------------------------------------------------------------
def download_fineweb2_language(lang_code, split="train", streaming=True):
    """
    lang_code example: 'eng_Latn', 'fra_Latn', 'cmn_Hani'
    Set streaming=False to download the full subset locally (can be large!).
    """
    from datasets import load_dataset

    ds = load_dataset(
        FINEWEB2_REPO,
        name=lang_code,
        split=split,
        streaming=streaming,
    )
    return ds


if __name__ == "__main__":
    fw2_langs = get_fineweb2_languages()
    print("Sample FineWeb2 codes:", fw2_langs[:10])

    flores_codes = get_floresplus_codes()
    print("Sample FLORES+ codes:", flores_codes[:10])

    results = compare(fw2_langs, flores_codes)

    # Example: save results to CSV files
    pd.Series(results["both"]).to_csv("training_setup/langs/langs_in_both.csv", index=False, header=["code"])
    pd.Series(results["only_fineweb2"]).to_csv("training_setup/langs/langs_only_fineweb2.csv", index=False, header=["code"])
    pd.Series(results["only_floresplus"]).to_csv("training_setup/langs/langs_only_floresplus.csv", index=False, header=["code"])

    # Example download (streaming, doesn't pull whole dataset to disk):
    # ds = download_fineweb2_language("eng_Latn", streaming=True)
    # for example in ds.take(3):
    #     print(example)