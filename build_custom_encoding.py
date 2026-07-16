"""
build_custom_encoding.py

STEP 1 of the custom-encoding effort: scans every character that appears
in the training shards, validation shards, AND FLORES+ evaluation data
(restricted to the 20 languages in --langs-csv) to build a fixed-width
custom byte encoding, as an alternative to UTF-8.

This must be EXHAUSTIVE -- any character missing from this inventory
would have no code to map to when STEP 2 (a separate, later change)
actually re-encodes text using this mapping before feeding it to the
model, instead of raw UTF-8 bytes.

Sources scanned, ALL of them, for EVERY language in --langs-csv:
  1. Training shards:   <shard-root>/<lang>/<lang>.chunk.*.jsonl, for EACH
                        --shard-roots entry (both "balanced" and
                        "imbalanced" by default -- these can genuinely
                        contain different documents per language, since
                        prepare_language_shards.py builds them from
                        separate streaming passes)
  2. Validation shards: <shard-root>/<lang>/<lang>.val.jsonl, same roots
  3. FLORES+ eval data: openlanguagedata/flores_plus, dev split, per
                        language (same source model_eval/run_eval.py uses)

DECIDING BYTE WIDTH: if the total unique-character count fits in 2^16
(65,536), every character gets a 2-byte code; otherwise, if it fits in
2^24 (16,777,216), 3 bytes are used instead. Given this corpus spans
scripts like Han, Hangul, Thai, Georgian, Tamil, etc., 2 bytes may not
be enough -- this script measures the real count rather than assuming.

CODE ASSIGNMENT: characters are sorted by Unicode codepoint and assigned
codes 0..N-1 in that order, purely for determinism (re-running this
script against the exact same data always reproduces an identical
mapping) -- not for any frequency-based compression benefit. If you'd
rather assign smaller codes to more frequent characters instead, that's
a straightforward follow-up change, but wasn't asked for here.

PERFORMANCE NOTE: this reads every line of every training/validation
shard for every language (the training corpus alone was ~11GB across
all languages combined, per launch_training.py's own printed corpus
size) -- expect this to take a while on the full corpus. Run it in a
tmux pane rather than interactively, and consider --skip-flores for a
quick dry run first (NOT safe for a final encoding, since a character
seen only in FLORES+ would then be missing).

Output (--out-path, default training_setup/custom_encoding.json):
  {
    "num_chars": <int>,
    "bytes_per_char": 2 or 3,
    "char_to_code": {char: int, ...},
    "code_to_char": {"<int as string>": char, ...}  (JSON object keys are always strings)
  }

Usage:
    python training_setup/build_custom_encoding.py
    python training_setup/build_custom_encoding.py --shard-roots data/lang_shards_balanced_4gpu data/lang_shards_imbalanced_4gpu --langs-csv training_setup/langs/langs_chosen.csv --out-path training_setup/custom_encoding.json
    python training_setup/build_custom_encoding.py --skip-flores   # quick dry run only
"""
import argparse
import csv
import glob
import json
import os

from datasets import load_dataset

FLORES_DATASET = "openlanguagedata/flores_plus"
FLORES_SPLIT = "dev"

DEFAULT_SHARD_ROOTS = [
    "data/lang_shards_balanced_4gpu",
    "data/lang_shards_imbalanced_4gpu",
]
DEFAULT_LANGS_CSV = "training_setup/langs/langs_chosen.csv"
DEFAULT_OUT_PATH = "training_setup/custom_encoding.json"

TWO_BYTE_MAX = 2 ** 16     # 65,536
THREE_BYTE_MAX = 2 ** 24   # 16,777,216


def load_lang_codes(langs_csv: str) -> list[str]:
    with open(langs_csv, newline="", encoding="utf-8") as f:
        return [row["language_code"] for row in csv.DictReader(f)]


def scan_jsonl_file(path: str, chars: set) -> int:
    """Updates chars in place with every character found in each line's
    "text" field. Returns the number of lines processed (0 if the file
    doesn't exist)."""
    if not os.path.exists(path):
        return 0
    n_lines = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get("text", "")
            if text:
                chars.update(text)
            n_lines += 1
    return n_lines


def scan_training_and_val(shard_roots: list[str], lang_code: str, chars: set) -> None:
    for shard_root in shard_roots:
        lang_dir = os.path.join(shard_root, lang_code)
        if not os.path.isdir(lang_dir):
            print(f"    {shard_root}: directory not found, skipping "
                  f"({lang_dir})")
            continue

        chunk_paths = sorted(glob.glob(os.path.join(lang_dir, f"{lang_code}.chunk.*.jsonl")))
        val_path = os.path.join(lang_dir, f"{lang_code}.val.jsonl")

        total_lines = 0
        for chunk_path in chunk_paths:
            total_lines += scan_jsonl_file(chunk_path, chars)
        val_lines = scan_jsonl_file(val_path, chars)
        total_lines += val_lines

        print(f"    {shard_root}: {len(chunk_paths)} chunk file(s), {total_lines} "
              f"line(s) scanned ({val_lines} from val)")


def scan_flores(lang_code: str, chars: set) -> None:
    try:
        ds = load_dataset(FLORES_DATASET, lang_code, split=FLORES_SPLIT)
    except Exception as e:
        print(f"    FLORES+: SKIPPED ({e})")
        return
    for row in ds:
        text = row.get("text", "")
        if text:
            chars.update(text)
    print(f"    FLORES+: {len(ds)} sentence(s) scanned")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shard-roots", nargs="+", default=DEFAULT_SHARD_ROOTS,
                         help=f"Root(s) of the training/validation shards to scan -- "
                              f"scans ALL of them (default: {DEFAULT_SHARD_ROOTS}, i.e. "
                              f"both balanced and imbalanced). A root that doesn't "
                              f"exist is skipped with a note rather than erroring, "
                              f"e.g. if you only ever built one of the two.")
    parser.add_argument("--langs-csv", default=DEFAULT_LANGS_CSV,
                         help=f"CSV whose 'language_code' column lists the languages to "
                              f"scan (default {DEFAULT_LANGS_CSV}).")
    parser.add_argument("--out-path", default=DEFAULT_OUT_PATH,
                         help=f"Where to save the character inventory + encoding "
                              f"(default {DEFAULT_OUT_PATH}).")
    parser.add_argument("--skip-flores", action="store_true",
                         help="Skip scanning FLORES+ eval data (train+val shards only) -- "
                              "useful for a quick dry run, but NOT safe for a real final "
                              "encoding, since any character seen only in FLORES+ would "
                              "then have no code assigned.")
    return parser.parse_args()


def main():
    args = parse_args()
    lang_codes = load_lang_codes(args.langs_csv)
    print(f"Scanning {len(lang_codes)} language(s) from {args.langs_csv}\n")

    chars = set()
    for lang_code in lang_codes:
        print(f"  {lang_code}:")
        scan_training_and_val(args.shard_roots, lang_code, chars)
        if not args.skip_flores:
            scan_flores(lang_code, chars)
        print(f"    running total unique characters: {len(chars)}")

    n_chars = len(chars)
    print(f"\nTotal unique characters across all sources: {n_chars}")

    if n_chars <= TWO_BYTE_MAX:
        bytes_per_char = 2
        capacity = TWO_BYTE_MAX
    elif n_chars <= THREE_BYTE_MAX:
        bytes_per_char = 3
        capacity = THREE_BYTE_MAX
    else:
        raise SystemExit(
            f"{n_chars} unique characters exceeds even 3-byte capacity "
            f"({THREE_BYTE_MAX:,}) -- this script only supports 2 or 3 bytes per "
            f"character; a 4-byte scheme would be needed instead."
        )
    print(f"-> fits in {bytes_per_char} bytes per character (capacity: {capacity:,})")

    # Deterministic ordering (by codepoint) so re-running this script on the
    # exact same data always reproduces the identical mapping.
    sorted_chars = sorted(chars)
    char_to_code = {ch: i for i, ch in enumerate(sorted_chars)}
    code_to_char = {i: ch for i, ch in enumerate(sorted_chars)}

    out = {
        "num_chars": n_chars,
        "bytes_per_char": bytes_per_char,
        "char_to_code": char_to_code,
        "code_to_char": code_to_char,
    }
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {args.out_path}")


if __name__ == "__main__":
    main()