"""
add_char_entropies.py

STEP 2 of the pipeline (run after run_eval.py, before
calibrate_thresholds.py). Adds a "chars_entropies" key to every sentence
in <results-dir>/{lang_code}.json, one entry per character:

    chars_entropies: [
        [char, next_char_entropy, n_bytes, [byte1, byte2, ...]],
        ...
    ]

CONVENTION -- "next character", mirroring bytes_entropies:
run_eval.py stores bytes_entropies[i] = [byte_i, H_i, norm_i], where H_i is
the entropy of the model's prediction AFTER reading byte i, i.e. the
uncertainty about byte i+1 (the prediction for byte 0, from BOS, is not
stored). chars_entropies keeps exactly this convention one level up:
chars_entropies[c][1] is the uncertainty about character c+1, i.e. the sum
of the H values that predict each of character c+1's bytes:

    chars_entropies[c][1] = H[end(c)-1] + ... + H[end(c+1)-2]
                            (last byte of c ... second-to-last byte of c+1)

so every score covers the bytes of exactly ONE character. Downstream
nothing changes: calibrate_thresholds.py / run_patching.py apply BLT's
one-position shift (a high score at c cuts before c+1), which now means
exactly "character c+1 is surprising, start a patch there".

(An earlier version summed H over character c's OWN bytes, which is the
uncertainty about c's continuation bytes plus c+1's first byte -- mixing
two characters. For 1-byte characters the two are identical, so Latin
script is essentially unaffected; multi-byte scripts and every character
of the fixed-width custom encoding are.)

EDGES:
  - Last character: its score is only H at its last byte (the prediction
    past the end of the sentence), like the last entry of bytes_entropies.
  - First character: the H values predicting its continuation bytes
    (H[0] .. H[end(0)-2]) have no preceding character slot and are
    dropped, just as the BOS prediction for byte 0 is not stored. Nothing
    is dropped when the first character is 1 byte.

Only the RAW entropy (bytes_entropies[i][1]) is summed -- the normalized
score (bytes_entropies[i][2]) is intentionally dropped, since summing
already-normalized (mean-0) per-byte scores doesn't correspond to a
meaningful per-character quantity.

GROUPING DEPENDS ON ENCODING:
  - UTF-8 (default): a byte starts a new character iff it is NOT a
    continuation byte, i.e. (byte & 0xC0) != 0x80.
  - Custom fixed-width encoding (e.g. --custom-encoding-path's 2-byte
    scheme): grouped in fixed chunks of --bytes-per-char (default 2).

ENCODING AUTO-DETECTION: if --bytes-per-char is not passed explicitly,
this script checks whether "_customenc" appears anywhere in --results-dir.
If found, assumes the fixed-width custom encoding (default 2 bytes/char).
Pass --bytes-per-char or --force-utf8 explicitly to override.

VERSIONING / SKIPPING: each sentence processed with this convention gets
"chars_entropies_convention": "next_char". Sentences that already carry
that marker are skipped (unless --force); sentences with an older
chars_entropies (no marker) are ALWAYS recomputed, so a plain re-run
upgrades a results dir without needing --force.

SANITY CHECKS (per sentence, raises on failure):
  1. len(chars_entropies) == len(text)         -- one entry per character
  2. sum(char scores) + dropped first-char part == sum(bytes_entropies[i][1])
     (within floating-point tolerance)         -- no bytes lost/duplicated

Output: updates <results-dir>/{lang_code}.json in place (compact JSON,
no indentation, to keep files small).
Existing keys are left untouched. After changing chars_entropies, the
char-level calibration/patching (steps 3-7 with --score-source chars, and
run_cumulative_patching.py --score-source chars) must be re-run.

Usage:
    python model_eval/add_char_entropies.py --results-dir results/own_models/<run>/step_<step>
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
CONVENTION_KEY = "chars_entropies_convention"
CONVENTION = "next_char"


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

    h = [entry[1] for entry in bytes_entropies]
    ends, pos = [], 0
    for group in groups:
        pos += len(group)
        ends.append(pos)          # end(c): exclusive byte end of character c

    n_chars = len(groups)
    chars_entropies = []
    for c, (ch, group) in enumerate(zip(text, groups)):
        if c < n_chars - 1:
            # H values predicting the bytes of character c+1
            score = sum(h[ends[c] - 1: ends[c + 1] - 1])
        else:
            # last character: only the prediction past the end of the sentence
            score = h[ends[c] - 1]
        byte_list = [entry[0] for entry in group]
        chars_entropies.append([ch, round(score, 6), len(byte_list), byte_list])

    # H predicting the first character's continuation bytes has no slot
    dropped = sum(h[0: ends[0] - 1]) if n_chars else 0.0
    total_from_chars = sum(c[1] for c in chars_entropies)
    total_from_bytes = sum(h)
    if abs(total_from_chars + dropped - total_from_bytes) > FLOAT_TOLERANCE:
        raise ValueError(
            f"Char scores ({total_from_chars:.6f}) + dropped first-char part "
            f"({dropped:.6f}) do not match summed byte entropy "
            f"({total_from_bytes:.6f}) for text: {text!r}"
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
        help="Recompute chars_entropies even for sentences already computed "
             f"with the current convention (default: skip those; sentences "
             f"with an older chars_entropies are always recomputed).",
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
            if sentence.get(CONVENTION_KEY) == CONVENTION and not args.force:
                n_skipped += 1
                continue
            sentence["chars_entropies"] = build_chars_entropies(
                sentence["text"], sentence["bytes_entropies"], custom_bytes_per_char
            )
            sentence[CONVENTION_KEY] = CONVENTION
            n_done += 1

        with open(path, "w", encoding="utf-8") as f:
            json.dump(sentences, f, ensure_ascii=False, separators=(",", ":"))

        skip_note = f" ({n_skipped} already up to date, skipped)" if n_skipped else ""
        print(f"  {lang_code}: added chars_entropies to {n_done} sentence(s){skip_note}")

    print("\nDone.")


if __name__ == "__main__":
    main()