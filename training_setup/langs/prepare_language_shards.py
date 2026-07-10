"""
prepare_language_shards.py

Streams FineWeb (English) / FineWeb2 (everything else) and writes each
language's BALANCED, content-equalized byte allocation (from
langs_chosen.csv's "balanced_allocation_bytes" column -- see
add_balanced_allocation.py) out as JSONL chunk shards, in the layout
bytelatent's dataloader expects:

    <root_dir>/<language_code>/<language_code>.chunk.00.jsonl
    <root_dir>/<language_code>/<language_code>.chunk.01.jsonl
    ...
    <root_dir>/<language_code>/<language_code>.chunk.07.jsonl   (n_chunks total)

IMPORTANT DESIGN CHANGE from the previous version: every language (not just
Chichewa) is capped at its own small, ratio-proportional
balanced_allocation_bytes value -- e.g. German gets ~434MB written, NOT the
~1.65TB actually available in FineWeb2. This is deliberate: bytelatent's
SamplingIterator draws documents from each language's chunks with
probability proportional to data.sources[language_code] (set to that
language's ratio_vs_english -- see the sources: block printed at the end
of this script), and LoopingIterator wraps every source unconditionally,
restarting it from the beginning once exhausted (confirmed by reading
bytelatent/args.py + bytelatent/data/iterators/{sampling,looping}_iterator.py
directly). Since every language's shard size is proportional to its own
sampling weight, every source empties out and loops back at roughly the
SAME point in training -- that's what makes the corpus "balanced": at any
point you stop training (e.g. via early stopping), the cumulative bytes
consumed per language should track these same ratios, not just at a clean
epoch boundary.

Validation split: targets an exact BYTE count (--val-bytes, default
5,000,000, same value used for Chichewa in data_to_params_ratio.py), not a
fixed document count -- a fixed document count would make actual bytes
collected depend on that language's average document length. This is
ADDITIONAL to balanced_allocation_bytes, not subtracted from it
(validation for the limiting language, Chichewa, was already accounted
for separately when balanced_allocation_bytes was computed).

Special cases (same as check_fineweb_availability.py):
  - eng_Latn routes to HuggingFaceFW/fineweb instead of fineweb-2.
  - cmn_Hans is queried in fineweb-2 under the config name cmn_Hani.

Each source line is written through unmodified (already has "text" + "id"/
"url", matching what bytelatent's get_text()/get_id_key() expect) -- no
schema conversion needed since file_format="json" is used, not "arrow".

Usage:
    python prepare_language_shards.py \
        training_setup/langs/langs_chosen.csv \
        [output_root_dir] \
        [--n-chunks 8] [--val-bytes 5000000] \
        [--byte-column balanced_allocation_bytes]

If output_root_dir is omitted, it's auto-derived as
data/lang_shards_balanced_<n_chunks>gpu/ -- no longer size-specific, since
every model size now shares the exact same balanced shard set (see
launch_training.py, which should point data.root_dir at this shared
directory for every --size).

Example:
    python prepare_language_shards.py training_setup/langs/langs_chosen.csv --n-chunks 4
    # -> writes to data/lang_shards_balanced_4gpu/
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
          f"target={target_bytes:,} bytes (balanced allocation), "
          f"val_target={val_bytes:,} bytes ---")

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
                # document count
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
    parser.add_argument("langs_csv", nargs="?", default="training_setup/langs/langs_chosen.csv",
                         help="CSV with languages to shard, including "
                              "balanced_allocation_bytes column (see "
                              "data_to_params_ratio.py). Default: "
                              "training_setup/langs/langs_chosen.csv")
    parser.add_argument("output_root_dir", nargs="?", default=None,
                         help="Optional. If omitted, auto-derived as "
                              "data/lang_shards_balanced_<n_chunks>gpu/ -- "
                              "shared across every model size.")
    parser.add_argument("--byte-column", default="balanced_allocation_bytes",
                         help="Column in langs_csv giving each language's "
                              "training byte target. Default: "
                              "balanced_allocation_bytes (see "
                              "add_balanced_allocation.py).")
    parser.add_argument("--n-chunks", type=int, default=4)
    parser.add_argument("--val-bytes", type=int, default=5_000_000,
                         help="Target bytes for the held-out validation "
                              "split, per language (not a document count). "
                              "Default 5,000,000, matching the --val-bytes "
                              "used for Chichewa in data_to_params_ratio.py, "
                              "so validation set size is consistent across "
                              "every language, not just the limiting one.")
    args = parser.parse_args()

    if args.output_root_dir is None:
        # Derive a short name from the byte column itself (e.g.
        # "balanced_allocation_bytes" -> "balanced",
        # "imbalanced_allocation_bytes" -> "imbalanced") so different
        # --byte-column runs land in DIFFERENT directories by default,
        # instead of all silently colliding on lang_shards_balanced_*.
        short_name = (
            args.byte_column
            .removesuffix("_allocation_bytes")
            .removesuffix("_bytes")
        )
        args.output_root_dir = os.path.join("data", f"lang_shards_{short_name}_{args.n_chunks}gpu")
        print(f"No output_root_dir given -- auto-derived: {args.output_root_dir}")

    with open(args.langs_csv, newline="") as f:
        reader = csv.DictReader(f)
        lang_rows = list(reader)

    os.makedirs(args.output_root_dir, exist_ok=True)

    sources_for_yaml = {}
    for row in lang_rows:
        language_code = row["language_code"]
        raw = row.get(args.byte_column, "n/a")
        if raw in ("n/a", "", None):
            print(f"SKIPPING {language_code}: no value in column {args.byte_column!r} "
                  f"(run add_balanced_allocation.py first)")
            continue
        target_bytes = int(raw)
        prepare_language(language_code, target_bytes, args.output_root_dir,
                          args.n_chunks, args.val_bytes)
        # Use the SAME value that determined target_bytes as the sampling
        # weight too -- not ratio_vs_english specifically. Whatever
        # --byte-column you sharded with (balanced_allocation_bytes,
        # imbalanced_allocation_bytes, or anything else) is, by
        # construction, already proportional to the weight you want:
        # balanced_allocation_bytes = content_budget * ratio_vs_english,
        # and imbalanced_allocation_bytes = raw_utf8_bytes * a constant
        # scale factor. SamplingIterator renormalizes weights before
        # drawing, so a constant scale factor (content_budget, or the
        # imbalanced scale factor) cancels out -- using target_bytes
        # itself as the weight reproduces the exact same normalized
        # sampling proportions as ratio_vs_english / source_weight would,
        # with zero risk of the printed weights coming from a DIFFERENT
        # column than the one that actually capped the data on disk.
        sources_for_yaml[language_code] = target_bytes

    print(f"\nDone. Shards written under {args.output_root_dir}/<language_code>/")
    print(f"\nPaste this into your yaml's data.sources (weights = {args.byte_column}, "
          f"the SAME column used to cap each language's shard size; "
          f"SamplingIterator renormalizes, so relative proportions are all that matter):")
    print("data:")
    print(f"  root_dir: {args.output_root_dir}")
    print("  sources:")
    for code, weight in sources_for_yaml.items():
        print(f"    {code}: {weight}")


if __name__ == "__main__":
    main()