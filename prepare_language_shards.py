"""
prepare_language_shards.py

Streams FineWeb (English) / FineWeb2 (everything else) and writes each
language's fairness-adjusted byte allocation (from langs_chosen.csv, e.g. the
"Medium_bytes" column produced by add_language_allocations.py) out as JSONL
chunk shards, in the exact layout bytelatent's dataloader expects:

    <root_dir>/<language_code>/<language_code>.chunk.00.jsonl
    <root_dir>/<language_code>/<language_code>.chunk.01.jsonl
    ...
    <root_dir>/<language_code>/<language_code>.chunk.07.jsonl   (n_chunks total)

Why 8 chunks: bytelatent's find_and_sanitize_chunks() requires
world_size % n_chunks == 0. With 8 GPUs, 8 chunks (one per rank) is the
simplest fit -- adjust --n-chunks if you train on a different GPU count.

Special cases (same as check_fineweb_availability.py):
  - eng_Latn routes to HuggingFaceFW/fineweb instead of fineweb-2.
  - cmn_Hans is queried in fineweb-2 under the config name cmn_Hani.

Each source line is written through unmodified (already has "text" + "id"/
"url", matching what bytelatent's get_text()/get_id_key() expect) -- no
schema conversion needed since file_format="json" is used, not "arrow".

Usage:
    python prepare_language_shards.py \
        training_setup/langs/langs_chosen.csv \
        <column_name, e.g. Medium_bytes> \
        <output_root_dir> \
        [--n-chunks 8] [--val-docs 200]
"""

import argparse
import csv
import json
import os

from datasets import load_dataset


FINEWEB2_DATASET = "HuggingFaceFW/fineweb-2"
FINEWEB_EN_DATASET = "HuggingFaceFW/fineweb"
FINEWEB_EN_CONFIG = "default"  # adjust to match whatever config your original
                                 # fineweb2_data_amounts.py check used for English

LANG_CODE_OVERRIDES = {
    "cmn_Hans": "cmn_Hani",
}


def resolve_dataset(language_code: str) -> tuple[str, str | None]:
    if language_code == "eng_Latn":
        return FINEWEB_EN_DATASET, FINEWEB_EN_CONFIG
    fineweb2_config = LANG_CODE_OVERRIDES.get(language_code, language_code)
    return FINEWEB2_DATASET, fineweb2_config


def prepare_language(language_code: str, target_bytes: int, root_dir: str,
                      n_chunks: int, val_docs: int) -> None:
    dataset_name, dataset_config = resolve_dataset(language_code)
    out_dir = os.path.join(root_dir, language_code)
    os.makedirs(out_dir, exist_ok=True)

    chunk_paths = [
        os.path.join(out_dir, f"{language_code}.chunk.{i:02d}.jsonl")
        for i in range(n_chunks)
    ]
    val_path = os.path.join(out_dir, f"{language_code}.val.jsonl")
    chunk_files = [open(p, "w") for p in chunk_paths]
    val_file = open(val_path, "w")

    print(f"--- {language_code} -> {dataset_name} ({dataset_config}) "
          f"target={target_bytes:,} bytes ---")

    try:
        ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        for f in chunk_files:
            f.close()
        val_file.close()
        return

    cumulative_bytes = 0
    docs_written = 0
    try:
        for doc in ds:
            text = doc.get("text", "")
            n_bytes = len(text.encode("utf-8"))

            if docs_written < val_docs:
                # first val_docs docs go to the held-out validation file,
                # not counted against the training byte budget
                val_file.write(json.dumps(doc, ensure_ascii=False) + "\n")
                docs_written += 1
                continue

            if cumulative_bytes >= target_bytes:
                break

            chunk_idx = (docs_written - val_docs) % n_chunks
            chunk_files[chunk_idx].write(json.dumps(doc, ensure_ascii=False) + "\n")
            cumulative_bytes += n_bytes
            docs_written += 1
    except Exception as e:
        print(f"  ERROR while streaming: {e} "
              f"(after {docs_written} docs, {cumulative_bytes:,} bytes)")
    finally:
        for f in chunk_files:
            f.close()
        val_file.close()

    status = "OK" if cumulative_bytes >= target_bytes else "SHORTFALL"
    print(f"  {status}: wrote {cumulative_bytes:,} / {target_bytes:,} bytes "
          f"across {n_chunks} chunks, {val_docs} val docs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("langs_csv")
    parser.add_argument("byte_column", help="e.g. Medium_bytes, Repo-scale_bytes")
    parser.add_argument("output_root_dir")
    parser.add_argument("--n-chunks", type=int, default=8)
    parser.add_argument("--val-docs", type=int, default=200)
    args = parser.parse_args()

    with open(args.langs_csv, newline="") as f:
        reader = csv.DictReader(f)
        lang_rows = list(reader)

    os.makedirs(args.output_root_dir, exist_ok=True)
    for row in lang_rows:
        language_code = row["language_code"]
        target_bytes = int(row[args.byte_column])
        prepare_language(language_code, target_bytes, args.output_root_dir,
                          args.n_chunks, args.val_docs)

    print(f"\nDone. Shards written under {args.output_root_dir}/<language_code>/")


if __name__ == "__main__":
    main()