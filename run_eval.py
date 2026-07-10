"""
eval_flores.py
Run BLT patching over a diverse subset of FLORES+ languages and save results to JSON.
Usage:
    python eval_flores.py --hf_weights/entropy_model
Output: one JSON file per language at results/{lang_code}.json
"""
import argparse
import csv
import json
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import subprocess
from blt_patcher import load_patcher, patch_text

# ── config ────────────────────────────────────────────────────────────────────
LIMIT = None  # set to an int for quick testing
REPO = "facebook/blt-1b"
SPLIT = "dev"
OUTPUT_DIR = "results/own_models"
FLORES_DATASET = "openlanguagedata/flores_plus"

# read the languages from floresplus_MASTER_CSV.csv, like this: Code_Orig : Name
with open("floresplus_MASTER_CSV.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    LANGUAGES = {row["Code_Orig"]: row["Name"] for row in reader}


# ── helpers ───────────────────────────────────────────────────────────────────
def normalize_scores(scores: list[float]) -> list[float]:
    arr = torch.tensor(scores)
    mean = arr.mean()
    std = arr.std()
    return ((arr - mean) / (std + 1e-8)).tolist()


def load_language(lang_code: str):
    return load_dataset(FLORES_DATASET, lang_code, split=SPLIT)


def run_language(lang_code: str, lang_name: str, eng_dataset, tokenizer, patcher):
    print(f"\nProcessing {lang_name} ({lang_code})...")
    if lang_code == "eng_Latn":
        dataset = eng_dataset
    else:
        try:
            dataset = load_language(lang_code)
        except Exception as e:
            print(f"  Skipping: {e}")
            return None

    assert len(dataset) == len(eng_dataset), (
        f"Length mismatch for {lang_code}: {len(dataset)} vs {len(eng_dataset)}"
    )

    results = []
    for i, (row, eng_row) in enumerate(
        tqdm(zip(dataset, eng_dataset), total=len(dataset), desc=lang_name)
    ):
        if LIMIT is not None and i >= LIMIT:
            break
        text = row["text"]
        eng_text = eng_row["text"]
        result = patch_text(text, tokenizer, patcher)
        text_bytes = result["text_bytes"]
        scores = result["scores"]
        norm = normalize_scores(scores)
        bytes_entropies = [
            [b, round(e, 6), round(en, 6)]
            for b, e, en in zip(text_bytes, scores, norm)
        ]
        results.append({
            "id": i,
            "text": text,
            "text_en": eng_text,
            "n_bytes": result["n_bytes"],
            "bytes_entropies": bytes_entropies
        })
    return results


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BLT patching over FLORES+ languages."
    )
    parser.add_argument(
        "--entropy_repo",
        type=str,
        required=True,
        help="Repo id (or path) of the entropy model to use for patching, "
             "e.g. --hf_weights/entropy_model",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading patcher...")
    tokenizer, patcher = load_patcher(repo=REPO, entropy_repo=args.entropy_repo)
    print(f"Loading English ({SPLIT})...")
    eng_dataset = load_language("eng_Latn")

    for lang_code, lang_name in LANGUAGES.items():
        results = run_language(lang_code, lang_name, eng_dataset, tokenizer, patcher)
        if results is None:
            continue
        out_path = os.path.join(OUTPUT_DIR, f"{lang_code}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(results)} entries → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()