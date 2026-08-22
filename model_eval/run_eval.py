"""
run_eval.py
Run BLT patching over a diverse subset of FLORES+ languages and save results to JSON.
Usage (from repo root):
    python model_eval/run_eval.py --entropy_repo dumps/entropy_10M_20lang_4gpu_sourcesbalanced_steps3000_ckpt200_lr4.5e-3/checkpoints/0000003000/consolidated
    python model_eval/run_eval.py --entropy_repo hf_weights/entropy_model --results-dir results/own_models/my_custom_name
    python model_eval/run_eval.py --entropy_repo <repo> --custom-encoding-path training_setup/custom_encoding.json
Output: one JSON file per language at
    results/own_models/<run_name>/step_<step>/{lang_code}.json
where <run_name> is the path component immediately after "dumps/" in
--entropy_repo (e.g. "entropy_10M_20lang_4gpu_sourcesbalanced_steps3000_ckpt200_lr4.5e-3")
and <step> is the checkpoint number (from "checkpoints/<step>/..." in the
path, if present) -- so results from different runs AND different
checkpoints of the same run don't silently overwrite each other under one
flat directory. Pass --results-dir to override this auto-derivation
entirely and write somewhere specific instead.

Must be invoked with the repo root as the current working directory
(same requirement as before this script lived under model_eval/) --
floresplus_MASTER.csv, --results-dir's default, etc. are all resolved
relative to cwd, not to this file's own location.

CUSTOM ENCODING: if the entropy model at --entropy_repo was trained with
launch_training.py's --custom-encoding-path, pass the SAME path here via
--custom-encoding-path -- FLORES+ text will then be encoded to bytes the
same way (see blt_patcher.py's patch_text/_text_to_raw_bytes) before
being fed to the entropy model. A mismatch silently produces meaningless
entropy scores and patch boundaries. Omit for a model trained on plain
UTF-8.
"""
import argparse
import csv
import json
import os
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))  # repo root, for blt_patcher + launch_training

import torch
from datasets import load_dataset
from tqdm import tqdm
import subprocess
from blt_patcher import load_patcher, patch_text
from results_paths import results_dir_for
from launch_training import get_free_gpu_ids, DEFAULT_FREE_MEM_THRESHOLD_MIB, DEFAULT_FREE_UTIL_THRESHOLD_PCT

# ── config ────────────────────────────────────────────────────────────────────
LIMIT = None  # set to an int for quick testing
REPO = "facebook/blt-1b"
SPLIT = "dev"
FLORES_DATASET = "openlanguagedata/flores_plus"
DEFAULT_LANGS_CSV = "training_setup/langs/langs_chosen.csv"

# read the languages from floresplus_MASTER.csv, like this: Code_Orig : Name
with open("floresplus_MASTER.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    LANGUAGES = {row["Code_Orig"]: row["Name"] for row in reader}


# ── helpers ───────────────────────────────────────────────────────────────────
def load_trained_lang_codes(langs_csv: str) -> set[str]:
    """Returns the set of language_code values from langs_csv (the 20
    languages actually used in training -- equivalent to Code_Orig in
    floresplus_MASTER.csv)."""
    with open(langs_csv, newline="", encoding="utf-8") as f:
        return {row["language_code"] for row in csv.DictReader(f)}


def normalize_scores(scores: list[float]) -> list[float]:
    arr = torch.tensor(scores, dtype=torch.float32)  # force fp16 here, not on the model
    mean = arr.mean()
    std = arr.std()
    return ((arr - mean) / (std + 1e-8)).tolist()


def load_language(lang_code: str):
    return load_dataset(FLORES_DATASET, lang_code, split=SPLIT)


def run_language(lang_code: str, lang_name: str, eng_dataset, tokenizer, patcher,
                  custom_encoding: dict | None):
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
        result = patch_text(text, tokenizer, patcher, custom_encoding=custom_encoding)
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
             "e.g. --entropy_repo dumps/entropy_10M_..._lr4.5e-3/checkpoints/0000003000/consolidated",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override the auto-derived results directory "
             "(results/own_models/<run>/step_<step>/). If omitted, derived "
             "from --entropy_repo via results_paths.derive_run_subdir.",
    )
    parser.add_argument(
        "--only-trained-langs",
        action="store_true",
        help="Only evaluate the languages actually used in training (from "
             "--langs-csv's 'language_code' column) instead of the full FLORES+ "
             "set in floresplus_MASTER.csv -- much faster when you only "
             "care about the 20 trained languages, e.g. for feeding into "
             "results_to_txt_premiums.py, which already restricts to these "
             "same languages at the reporting stage.",
    )
    parser.add_argument(
        "--langs-csv",
        type=str,
        default=DEFAULT_LANGS_CSV,
        help=f"CSV whose 'language_code' column defines the trained-language "
             f"set, used only when --only-trained-langs is set (default "
             f"{DEFAULT_LANGS_CSV}).",
    )
    parser.add_argument(
        "--custom-encoding-path",
        type=str,
        default=None,
        help="Path to the SAME custom_encoding.json passed to "
             "launch_training.py's --custom-encoding-path when the entropy "
             "model at --entropy_repo was trained. FLORES+ text will be "
             "encoded to bytes using this mapping instead of plain UTF-8 "
             "(see blt_patcher.py's patch_text). MUST match training exactly, "
             "or entropy scores / patch boundaries will be meaningless. Omit "
             "for a model trained on plain UTF-8.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run the (expensive) FLORES+ eval even for languages whose output "
             "JSON already exists in the results dir. By default, existing files "
             "are left as-is and skipped, since the eval itself is the slow part "
             "of this pipeline -- re-running run_patching.py afterward already "
             "picks up existing results incrementally without needing this.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Explicit physical GPU index to use, skipping auto-detection.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU, skipping GPU auto-detection entirely (much slower).",
    )
    parser.add_argument(
        "--free-mem-threshold-mib",
        type=int,
        default=DEFAULT_FREE_MEM_THRESHOLD_MIB,
        help="Only used for GPU auto-detection: a GPU counts as free if its used "
             f"memory is below this, in MiB (default {DEFAULT_FREE_MEM_THRESHOLD_MIB}).",
    )
    parser.add_argument(
        "--free-util-threshold-pct",
        type=int,
        default=DEFAULT_FREE_UTIL_THRESHOLD_PCT,
        help="Only used for GPU auto-detection: a GPU counts as free if its "
             f"utilization is below this percent (default {DEFAULT_FREE_UTIL_THRESHOLD_PCT}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.results_dir if args.results_dir is not None else results_dir_for(args.entropy_repo)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results under: {output_dir}/")

    languages = LANGUAGES
    if args.only_trained_langs:
        trained_codes = load_trained_lang_codes(args.langs_csv)
        languages = {code: name for code, name in LANGUAGES.items() if code in trained_codes}
        missing = trained_codes - set(languages)
        print(f"--only-trained-langs set: restricting to {len(languages)}/{len(LANGUAGES)} "
              f"languages (from {args.langs_csv}).")
        if missing:
            print(f"  NOTE: {len(missing)} code(s) from {args.langs_csv} not found in "
                  f"floresplus_MASTER.csv, skipped: {sorted(missing)}")

    # Split into "already have output, skip" vs "actually need to run" BEFORE
    # loading the patcher/model at all -- if everything's already done, this
    # avoids even that cost, not just the per-language FLORES+ eval itself.
    to_process = {}
    already_done = []
    for lang_code, lang_name in languages.items():
        out_path = os.path.join(output_dir, f"{lang_code}.json")
        if os.path.exists(out_path) and not args.force:
            already_done.append(lang_code)
        else:
            to_process[lang_code] = lang_name

    if already_done:
        print(f"Skipping {len(already_done)} already-evaluated language(s) "
              f"(pass --force to re-run anyway): {sorted(already_done)}")

    if not to_process:
        print("All requested languages already evaluated -- nothing to do.")
        return

    if args.cpu:
        print("Forcing CPU (--cpu set) -- this will be much slower.")
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using explicit GPU {args.gpu}.")
    else:
        try:
            gpu_ids = get_free_gpu_ids(1, args.free_mem_threshold_mib, args.free_util_threshold_pct)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
            print(f"Auto-detected idle GPU {gpu_ids[0]}, using it.")
        except SystemExit as e:
            print(f"No idle GPU found ({e}) -- falling back to CPU, this will be much "
                  f"slower. Pass --gpu to force a specific one, or free up a GPU.")

    print("Loading patcher...")
    tokenizer, patcher, custom_encoding = load_patcher(
        repo=REPO, entropy_repo=args.entropy_repo,
        custom_encoding_path=args.custom_encoding_path,
    )
    if custom_encoding is not None:
        print(f"Using CUSTOM ENCODING from {args.custom_encoding_path}")
    print(f"Loading English ({SPLIT})...")
    eng_dataset = load_language("eng_Latn")

    for lang_code, lang_name in to_process.items():
        results = run_language(lang_code, lang_name, eng_dataset, tokenizer, patcher, custom_encoding)
        if results is None:
            continue
        out_path = os.path.join(output_dir, f"{lang_code}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(results)} entries → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()