"""
add_char_entropies.py

NEW STEP 2 of the pipeline (run after run_eval.py, before
calibrate_thresholds.py). Adds a "chars_entropies" key to every sentence
in <results-dir>/{lang_code}.json, grouping each sentence's
"bytes_entropies" (per-byte [byte, raw_entropy, norm_entropy]) into
per-character entries:

    chars_entropies: [
        [char, summed_raw_entropy, n_bytes, [byte1, byte2, ...]],
        ...
    ]

Only the RAW entropy (bytes_entropies[i][1]) is summed -- the normalized
score (bytes_entropies[i][2]) is intentionally dropped, since summing
already-normalized (mean-0) per-byte scores doesn't correspond to a
meaningful per-character quantity.

GROUPING DEPENDS ON ENCODING:
  - UTF-8 (default): a byte starts a new character iff it is NOT a
    continuation byte, i.e. (byte & 0xC0) != 0x80. All following
    continuation bytes belong to that character. This matches how
    patch_text/_text_to_raw_bytes in blt_patcher.py encodes plain text,
    and needs no knowledge of the lead byte's declared sequence length --
    only whether each byte is a continuation byte or not.
  - Custom fixed-width encoding (e.g. --custom-encoding-path's 2-byte
    scheme): grouped in fixed chunks of --bytes-per-char (default 2),
    no byte inspection needed.

ENCODING AUTO-DETECTION: if --custom-encoding-path / --bytes-per-char is
not passed explicitly, this script checks whether "_customenc" appears
anywhere in --results-dir (matching the run-naming convention used
elsewhere in this pipeline, e.g. results_to_txt_premiums.py's
RUN_NAME_ALIASES). If found, assumes the fixed-width custom encoding
(default 2 bytes/char). Pass --bytes-per-char or --force-utf8 explicitly
to override auto-detection for a one-off results dir that doesn't follow
the naming convention.

SANITY CHECKS (per sentence, raises on failure -- these should never
fail for well-formed input, so a failure indicates a real bug in the
grouping or a mismatch between the assumed encoding and the one actually
used to produce bytes_entropies):
  1. len(chars_entropies) == len(text)         -- one entry per character
  2. sum(summed_raw_entropy across chars) == sum(bytes_entropies[i][1])
     (within floating-point tolerance)         -- no bytes dropped/duplicated

Output: updates <results-dir>/{lang_code}.json in place (indented),
adding "chars_entropies" to each sentence. Existing keys (including
"eval_modes" if run_patching.py already ran) are left untouched.

Usage:
    python model_eval/add_char_entropies.py --results-dir results/own_models/entropy_..._lr4.5e-3/step_0000006000
    # Custom 2-bytes-per-character encoding, explicit override:
    python model_eval/add_char_entropies.py --results-dir <dir> --bytes-per-char 2
    # Force UTF-8 grouping even if "_customenc" appears in the path:
    python model_eval/add_char_entropies.py --results-dir <dir> --force-utf8
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

CUSTOM_ENC_MARKER = "_customenc"
DEFAULT_CUSTOM_BYTES_PER_CHAR = 2
FLOAT_TOLERANCE = 1e-3  # sum of many rounded (6dp) floats; allow small drift


def is_continuation_byte(b: int) -> bool:
    return (b & 0xC0) == 0x80


def group_chars_utf8(bytes_entropies: list) -> list:
    """Groups per-byte [byte, raw_entropy, norm_entropy] entries into
    per-character [char_placeholder_bytes, summed_raw_entropy, n_bytes,
    [byte,...]] using UTF-8 continuation-byte detection. Returns groups
    of raw byte lists; caller decodes to actual characters against the
    sentence's text separately (see build_chars_entropies)."""
    groups = []
    current = []
    for entry in bytes_entropies:
        b = entry[0]
        if current and not is_continuation_byte(b):
            groups.append(current)
            current = []
        current.append(entry)
    if current:
        groups.append(current)
    return groups


def group_chars_fixed_width(bytes_entropies: list, bytes_per_char: int) -> list:
    groups = []
    for i in range(0, len(bytes_entropies), bytes_per_char):
        groups.append(bytes_entropies[i:i + bytes_per_char])
    return groups


def build_chars_entropies(text: str, bytes_entropies: list, custom_bytes_per_char: int | None) -> list:
    if custom_bytes_per_char is not None:
        groups = group_chars_fixed_width(bytes_entropies, custom_bytes_per_char)
    else:
        groups = group_chars_utf8(bytes_entropies)

    if len(groups) != len(text):
        raise ValueError(
            f"Grouping produced {len(groups)} character group(s) but text has "
            f"{len(text)} character(s) -- encoding mismatch (custom_bytes_per_char="
            f"{custom_bytes_per_char}) or malformed UTF-8 input. Text: {text!r}"
        )

    chars_entropies = []
    for ch, group in zip(text, groups):
        raw_sum = sum(entry[1] for entry in group)
        byte_list = [entry[0] for entry in group]
        chars_entropies.append([ch, round(raw_sum, 6), len(byte_list), byte_list])

    total_from_chars = sum(c[1] for c in chars_entropies)
    total_from_bytes = sum(entry[1] for entry in bytes_entropies)
    if abs(total_from_chars - total_from_bytes) > FLOAT_TOLERANCE:
        raise ValueError(
            f"Summed char entropy ({total_from_chars:.6f}) does not match summed "
            f"byte entropy ({total_from_bytes:.6f}) for text: {text!r}"
        )

    return chars_entropies


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Directory of per-language JSON files produced by run_eval.py "
             "(e.g. results/own_models/<run>/step_<step>/). Updated in place.",
    )
    parser.add_argument(
        "--bytes-per-char", type=int, default=None,
        help="Explicit fixed bytes-per-character for a custom encoding "
             f"(e.g. {DEFAULT_CUSTOM_BYTES_PER_CHAR}). Overrides auto-detection. "
             "Omit to auto-detect from --results-dir, or pass --force-utf8 to "
             "force UTF-8 grouping regardless of the path.",
    )
    parser.add_argument(
        "--force-utf8", action="store_true",
        help="Force UTF-8 continuation-byte grouping even if the results-dir "
             f"path contains {CUSTOM_ENC_MARKER!r}. Overrides auto-detection "
             "and --bytes-per-char.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute chars_entropies even for sentences that already have "
             "it (default: skip sentences that already have the key).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.force_utf8:
        custom_bytes_per_char = None
        print("Forcing UTF-8 grouping (--force-utf8 set).")
    elif args.bytes_per_char is not None:
        custom_bytes_per_char = args.bytes_per_char
        print(f"Using explicit fixed-width grouping: {custom_bytes_per_char} bytes/char.")
    elif CUSTOM_ENC_MARKER in args.results_dir:
        custom_bytes_per_char = DEFAULT_CUSTOM_BYTES_PER_CHAR
        print(f"Auto-detected {CUSTOM_ENC_MARKER!r} in --results-dir -- using fixed-width "
              f"grouping: {custom_bytes_per_char} bytes/char. Pass --bytes-per-char to "
              f"override, or --force-utf8 to force UTF-8 grouping instead.")
    else:
        custom_bytes_per_char = None
        print("No custom-encoding marker detected -- using UTF-8 continuation-byte grouping.")

    paths = sorted(Path(args.results_dir).glob("*.json"))
    print(f"Found {len(paths)} language file(s) in {args.results_dir}\n")

    for path in paths:
        lang_code = path.stem
        with open(path, encoding="utf-8") as f:
            sentences = json.load(f)

        n_done = 0
        n_skipped = 0
        for sentence in tqdm(sentences, desc=lang_code, leave=False):
            if "chars_entropies" in sentence and not args.force:
                n_skipped += 1
                continue
            sentence["chars_entropies"] = build_chars_entropies(
                sentence["text"], sentence["bytes_entropies"], custom_bytes_per_char
            )
            n_done += 1

        with open(path, "w", encoding="utf-8") as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)

        skip_note = f" ({n_skipped} already had chars_entropies, skipped)" if n_skipped else ""
        print(f"  {lang_code}: added chars_entropies to {n_done} sentence(s){skip_note}")

    print("\nDone.")


if __name__ == "__main__":
    main()