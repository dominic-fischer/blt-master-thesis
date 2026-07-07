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

Why n_chunks matters: bytelatent's find_and_sanitize_chunks() requires
world_size % n_chunks == 0, and silently DISCARDS excess chunks if
n_chunks > world_size, or SHARES a chunk across multiple ranks if
n_chunks < world_size -- so n_chunks should equal your actual GPU count
for full, non-duplicated data coverage. This script defaults to 8; pass
--n-chunks to match your launch.

Validation split: targets an exact BYTE count (--val-bytes, default
500,000), not a fixed document count. A fixed document count would make
the actual bytes collected depend on how long that language's documents
happen to be -- exactly the problem eval_entropy_bpb.py's
--target-bytes-per-lang was built to avoid on the eval side. Sizing this
script's val split by bytes (with headroom above eval's own
--target-bytes-per-lang, e.g. 500k written here vs. 400k evaluated there)
means eval should never hit its "ran out of validation data" shortfall
warning, without having to guess a per-language document count that
happens to translate to enough bytes.

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
        [output_root_dir] \
        [--n-chunks 8] [--val-bytes 500000]

If output_root_dir is omitted, it's auto-derived as
data/lang_shards_<size>_<n_chunks>gpu/ (e.g. data/lang_shards_tiny_4gpu),
matching the convention launch_training.py expects -- so the two scripts
stay in sync without manually typing matching paths.

Example:
    python prepare_language_shards.py \
        training_setup/langs/langs_chosen.csv Tiny_bytes --n-chunks 4
    # -> writes to data/lang_shards_tiny_4gpu/
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
                      n_chunks: int, val_bytes: int) -> None:
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
          f"target={target_bytes:,} bytes, val_target={val_bytes:,} bytes ---")

    try:
        ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        for f in chunk_files:
            f.close()
        val_file.close()
        return

    cumulative_bytes = 0
    cumulative_val_bytes = 0
    val_docs_written = 0
    train_docs_written = 0
    try:
        for doc in ds:
            text = doc.get("text", "")
            n_bytes = len(text.encode("utf-8"))

            if cumulative_val_bytes < val_bytes:
                # fill the held-out validation file first, by BYTES not
                # document count, so eval later can rely on every language
                # actually having val_bytes worth of data available --
                # not counted against the training byte budget
                val_file.write(json.dumps(doc, ensure_ascii=False) + "\n")
                cumulative_val_bytes += n_bytes
                val_docs_written += 1
                continue

            if cumulative_bytes >= target_bytes:
                break

            chunk_idx = train_docs_written % n_chunks
            chunk_files[chunk_idx].write(json.dumps(doc, ensure_ascii=False) + "\n")
            cumulative_bytes += n_bytes
            train_docs_written += 1
    except Exception as e:
        print(f"  ERROR while streaming: {e} "
              f"(after {train_docs_written} train docs, {cumulative_bytes:,} bytes; "
              f"{val_docs_written} val docs, {cumulative_val_bytes:,} val bytes)")
    finally:
        for f in chunk_files:
            f.close()
        val_file.close()

    train_status = "OK" if cumulative_bytes >= target_bytes else "SHORTFALL"
    val_status = "OK" if cumulative_val_bytes >= val_bytes else "SHORTFALL"
    print(f"  train {train_status}: wrote {cumulative_bytes:,} / {target_bytes:,} bytes "
          f"across {n_chunks} chunks ({train_docs_written} docs)")
    print(f"  val   {val_status}: wrote {cumulative_val_bytes:,} / {val_bytes:,} bytes "
          f"({val_docs_written} docs)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("langs_csv")
    parser.add_argument("byte_column", help="e.g. Tiny_bytes, Small_bytes, Medium_bytes")
    parser.add_argument("output_root_dir", nargs="?", default=None,
                         help="Optional. If omitted, auto-derived as "
                              "data/lang_shards_<size>_<n_chunks>gpu/ from "
                              "byte_column and --n-chunks, matching the "
                              "convention launch_training.py expects.")
    parser.add_argument("--n-chunks", type=int, default=8)
    parser.add_argument("--val-bytes", type=int, default=500_000,
                         help="Target bytes for the held-out validation "
                              "split, per language (not a document count). "
                              "Default 500,000 gives headroom above "
                              "eval_entropy_bpb.py's own "
                              "--target-bytes-per-lang (default 400,000), "
                              "so eval shouldn't hit its shortfall warning.")
    args = parser.parse_args()

    if args.output_root_dir is None:
        size = args.byte_column.removesuffix("_bytes").lower()
        args.output_root_dir = os.path.join("data", f"lang_shards_{size}_{args.n_chunks}gpu")
        print(f"No output_root_dir given -- auto-derived: {args.output_root_dir}")

    with open(args.langs_csv, newline="") as f:
        reader = csv.DictReader(f)
        lang_rows = list(reader)

    os.makedirs(args.output_root_dir, exist_ok=True)
    for row in lang_rows:
        language_code = row["language_code"]
        target_bytes = int(row[args.byte_column])
        prepare_language(language_code, target_bytes, args.output_root_dir,
                          args.n_chunks, args.val_bytes)

    print(f"\nDone. Shards written under {args.output_root_dir}/<language_code>/")


if __name__ == "__main__":
    main()