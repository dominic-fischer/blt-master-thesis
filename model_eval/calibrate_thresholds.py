"""
calibrate_thresholds.py

Runs English sentences across a range of thresholds for all cases:
  - raw entropy mode
  - raw monotonicity mode
  - normalised entropy mode        (bytes score-source only, see below)
  - combined mode                  (bytes score-source only, see below)

For each case, finds:
  - The threshold at exactly --pps-low patches/sentence (lower bound)
  - The threshold at exactly --pps-high patches/sentence (upper bound)
  - The midpoint between them
  - The exact anchor threshold that gives --target-pps patches/sentence (binary search)

For combined mode specifically:
  - t is fixed by finding the entropy threshold that gives pps=--pps-low at t_add=0
  - t_add is then swept to find the upper bound, midpoint, and anchor

WHY RE-RUN THIS PER MODEL/CHECKPOINT: a fixed threshold value doesn't
correspond to the same operating point across two different models --
entropy score distributions can differ in scale/sharpness, so the same
literal threshold could mean "English at 4 bytes/patch" for one model
and "English at 6 bytes/patch" for another. Recalibrating per model so
English hits the SAME target pps each time is what makes premiums
(lang_pps / eng_pps) actually comparable across models -- otherwise
you're comparing two different anchor points and calling it one metric.

WHY PPS, NOT BPP, FOR ALL FOUR THRESHOLDS: all four calibration targets
(low/mid/high/anchor) are PPS-based, not BPP-based. pps (patches per
sentence) is a sentence-level count that's meaningful regardless of what
a "unit" is (a byte or a character) or how many bytes that unit spans,
which is exactly why the same --target-pps/--pps-low/--pps-high DEFAULTS
are reused unchanged for --score-source=chars (see SCORE SOURCE below) --
we still want the same number of patches per sentence, i.e. the same
compression rate, just discovered by thresholding over a different score
sequence. BPP, by contrast, scales with the encoding's bytes-per-unit, so
it's only comparable across encodings/granularities when computed the
same way -- see the byte-length-per-patch reconstruction in
eval_english() for score-source=chars.

SCORE SOURCE (--score-source bytes|chars): controls which per-sentence
score sequence thresholds are searched over.
  - bytes (default, original behavior): scores = sentence["bytes_entropies"][i][score_idx],
    one score per byte. bpp = total_bytes / total_patches, and a "patch"
    already spans bytes directly.
  - chars: scores = sentence["chars_entropies"][i][1] (the per-character
    SUMMED raw entropy added by add_char_entropies.py -- run that script
    first). Requires "chars_entropies" to already exist in the results
    dir's JSON files. Only score_idx=1 (raw_entropy, raw_monotonicity)
    is valid here, since chars_entropies has no normalized-score or
    combined-mode equivalent (see add_char_entropies.py's module
    docstring for why the normalized per-byte score isn't summed) --
    norm_entropy and combined are automatically excluded from
    --score-source=chars runs even if requested via other flags,
    with a printed note. bpp is reconstructed by summing each
    patch's characters' byte counts (chars_entropies[i][2]), since a
    "patch" here spans characters of variable byte width, not bytes
    directly -- see byte_lengths_for_patches().
  --search-high-override, if given, replaces every case's search_high
  upper bound (see CASES) -- summing 1-4 per-byte raw entropies into one
  per-character score shifts the score's scale/range upward versus a
  single byte's score, so the byte-tuned search_high values below may not
  bracket the true threshold anymore. A --score-source=chars run should
  pass a wider bound (e.g. 6.0) to compensate; --search-high-override is
  independent of --score-source so it can also be used to widen bounds
  for an unusual byte-mode checkpoint if ever needed.

TARGET_PPS/PPS_LOW/PPS_HIGH default to values matching the base model's
original UTF-8 calibration (32.77 anchor; 36/23 as the pps equivalents of
the former 3.5/5.5 bpp bounds) -- change these only if you deliberately
want a different (non-comparable-to-base) target, e.g. doubling
--target-pps (and --pps-low/--pps-high) for a checkpoint trained with a
fixed 2-bytes-per-character custom encoding, to compensate for English
now taking roughly twice as many raw bytes for the same sentence content.
(This doubling is about the CUSTOM ENCODING changing bytes-per-character,
not about --score-source=chars -- the two are independent: a custom-
encoding checkpoint evaluated with --score-source=chars still wants the
SAME target-pps as its own --score-source=bytes run, per the "SCORE
SOURCE" section above, since pps doesn't change meaning between
granularities the way bpp does.)

Output: printed to stdout + saved to --out-path (default
calibrated_thresholds/thresholds_summary.csv)

Usage:
    python model_eval/calibrate_thresholds.py --results-dir results/base_model --out-path calibrated_thresholds/base_model_thresholds_summary.csv
    python model_eval/calibrate_thresholds.py --results-dir results/own_models/<run>/step_<step> --out-path calibrated_thresholds/<stem>_thresholds_summary.csv
    # Custom 2-bytes-per-character encoding -- double the pps targets:
    python model_eval/calibrate_thresholds.py --results-dir <dir> --target-pps 65.54 --pps-low 72 --pps-high 46
    # Char-level score source (requires chars_entropies already added):
    python model_eval/calibrate_thresholds.py --results-dir <dir> --score-source chars --search-high-override 6.0 --out-path calibrated_thresholds/char_level/<stem>_thresholds_summary.csv
"""

import argparse
import json
import os
import csv
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from bytelatent.data.patcher import (
    find_entropy_patch_start_ids,
    patch_lengths_from_start_ids,
)

# ── config ────────────────────────────────────────────────────────────────────
ENGLISH = "eng_Latn"
DEFAULT_TARGET_PPS = 32.77
DEFAULT_PPS_LOW = 36.0
DEFAULT_PPS_HIGH = 23.0
DEFAULT_OUT_PATH = "calibrated_thresholds/thresholds_summary.csv"
SCORE_SOURCES = ("bytes", "chars")
# Only score_idx=1 (raw entropy / raw monotonicity) has a char-level
# equivalent -- chars_entropies has no normalized score, so norm_entropy
# (score_idx=2) is meaningless for --score-source=chars. See module
# docstring's SCORE SOURCE section.
CHAR_COMPATIBLE_SCORE_IDX = {1}

CASES = {
    "raw_entropy": {
        "score_idx":    1,
        "monotonicity": False,
        "search_low":   0.5,
        "search_high":  4.0,
    },
    "raw_monotonicity": {
        "score_idx":    1,
        "monotonicity": True,
        "search_low":   0.1,
        "search_high":  3.0,
    },
    "norm_entropy": {
        "score_idx":    2,
        "monotonicity": False,
        "search_low":   -2.0,
        "search_high":  3.0,
    },
}

COMBINED = {
    "score_idx":     1,
    "t_search_low":  0.5,
    "t_search_high": 4.0,
    "t_add_max":     3.0,
}


# ── score-source helpers ──────────────────────────────────────────────────────

def get_scores_and_byte_counts(sentence: dict, score_idx: int, source: str) -> tuple[list[float], list[int]]:
    """Returns (scores, byte_counts) for one sentence, aligned unit-for-unit
    (one score + one byte count per "unit" -- a byte in bytes-mode, a
    character in chars-mode). byte_counts is used only to reconstruct
    each patch's byte length afterward (see byte_lengths_for_patches);
    for bytes-mode every unit IS one byte, so byte_counts is all 1s and
    reconstruction is a no-op identity."""
    if source == "bytes":
        scores = [be[score_idx] for be in sentence["bytes_entropies"]]
        byte_counts = [1] * len(scores)
    else:  # chars
        # chars_entropies[i] = [char, summed_raw_entropy, n_bytes, byte_list]
        # score_idx is validated to be 1 (summed_raw_entropy) by caller.
        scores = [ce[1] for ce in sentence["chars_entropies"]]
        byte_counts = [ce[2] for ce in sentence["chars_entropies"]]
    return scores, byte_counts


def byte_lengths_for_patches(patch_lengths_units: list[int], byte_counts: list[int]) -> list[int]:
    """Given patch lengths in UNITS (bytes for bytes-mode, characters for
    chars-mode) and the per-unit byte_counts they were computed over,
    returns each patch's length in actual BYTES -- needed for bpp in
    chars-mode, where a patch spans a variable number of bytes depending
    on which characters (ASCII vs multi-byte) it contains."""
    lengths_bytes = []
    idx = 0
    for pl in patch_lengths_units:
        lengths_bytes.append(sum(byte_counts[idx:idx + pl]))
        idx += pl
    return lengths_bytes


# ── patch helpers ─────────────────────────────────────────────────────────────

def compute_patch_lengths(
    scores: list[float],
    threshold: float,
    monotonicity: bool,
) -> list[int]:
    """Returns patch lengths in UNITS (one unit = one score/entry, i.e. one
    byte in bytes-mode or one character in chars-mode)."""
    bos_score = 99.0
    scores_with_bos = torch.tensor([[bos_score] + scores])
    patch_start_ids = find_entropy_patch_start_ids(
        scores_with_bos,
        threshold=threshold,
        threshold_add=None,
        monotonicity=monotonicity,
    )
    n_units = len(scores)
    try:
        patch_lengths = patch_lengths_from_start_ids(patch_start_ids, n_units + 1)
        return [l for l in patch_lengths[0].tolist()[1:] if l > 0]
    except AssertionError:
        return [n_units]


def compute_patch_lengths_combined(
    scores: list[float],
    t: float,
    t_add: float,
) -> list[int]:
    bos_score = 99.0
    scores_with_bos = torch.tensor([[bos_score] + scores])
    patch_start_ids = find_entropy_patch_start_ids(
        scores_with_bos,
        threshold=t,
        threshold_add=t_add,
        monotonicity=False,
    )
    n_units = len(scores)
    try:
        patch_lengths = patch_lengths_from_start_ids(patch_start_ids, n_units + 1)
        return [l for l in patch_lengths[0].tolist()[1:] if l > 0]
    except AssertionError:
        return [n_units]


# ── eval helpers ──────────────────────────────────────────────────────────────

def eval_english(sentences, threshold, score_idx, monotonicity, source):
    """Returns (mean_pps, mean_bpp) for English at a given threshold.
    pps is patches/sentence regardless of source (unit-count-agnostic).
    bpp always uses reconstructed BYTE lengths per patch (byte_lengths_for_patches),
    which is an identity pass-through for source="bytes"."""
    all_pps, all_bpp = [], []
    for s in sentences:
        scores, byte_counts = get_scores_and_byte_counts(s, score_idx, source)
        lengths_units = compute_patch_lengths(scores, threshold, monotonicity)
        lengths_bytes = byte_lengths_for_patches(lengths_units, byte_counts)
        n_patches = len(lengths_units)
        n_bytes   = sum(lengths_bytes)
        all_pps.append(n_patches)
        all_bpp.append(n_bytes / n_patches if n_patches > 0 else 0.0)
    return sum(all_pps) / len(all_pps), sum(all_bpp) / len(all_bpp)


def eval_english_combined(sentences, t, t_add, score_idx, source):
    """Returns (mean_pps, mean_bpp) for English in combined mode.
    Combined mode is bytes-only (see CHAR_COMPATIBLE checks in main()),
    but source is threaded through for symmetry / future-proofing."""
    all_pps, all_bpp = [], []
    for s in sentences:
        scores, byte_counts = get_scores_and_byte_counts(s, score_idx, source)
        lengths_units = compute_patch_lengths_combined(scores, t, t_add)
        lengths_bytes = byte_lengths_for_patches(lengths_units, byte_counts)
        n_patches = len(lengths_units)
        n_bytes   = sum(lengths_bytes)
        all_pps.append(n_patches)
        all_bpp.append(n_bytes / n_patches if n_patches > 0 else 0.0)
    return sum(all_pps) / len(all_pps), sum(all_bpp) / len(all_bpp)


# ── binary search helpers ─────────────────────────────────────────────────────

def binary_search_pps(sentences, target_pps, score_idx, monotonicity, low, high, source, tolerance=0.05):
    """Binary search for threshold giving target_pps. Higher threshold → lower pps."""
    mid = (low + high) / 2
    for _ in range(50):
        mid = (low + high) / 2
        mean_pps, mean_bpp = eval_english(sentences, mid, score_idx, monotonicity, source)
        if abs(mean_pps - target_pps) < tolerance:
            return mid, mean_pps, mean_bpp
        if mean_pps > target_pps:
            low = mid
        else:
            high = mid
        if high - low < 1e-6:
            break
    mean_pps, mean_bpp = eval_english(sentences, mid, score_idx, monotonicity, source)
    return mid, mean_pps, mean_bpp


def binary_search_t_for_combined_pps(sentences, target_pps, score_idx, t_add, low, high, source, tolerance=0.05):
    """Binary search over t (entropy threshold) in combined mode at fixed
    t_add, targeting a PPS value (used to fix t for the lower bound)."""
    mid = (low + high) / 2
    for _ in range(50):
        mid = (low + high) / 2
        mean_pps, mean_bpp = eval_english_combined(sentences, mid, t_add, score_idx, source)
        if abs(mean_pps - target_pps) < tolerance:
            return mid, mean_pps, mean_bpp
        # Higher entropy threshold t → fewer patch boundaries → lower pps
        if mean_pps > target_pps:
            low = mid
        else:
            high = mid
        if high - low < 1e-6:
            break
    mean_pps, mean_bpp = eval_english_combined(sentences, mid, t_add, score_idx, source)
    return mid, mean_pps, mean_bpp


def binary_search_t_add(sentences, target, score_idx, t, low, high, source, mode="pps", tolerance=0.05):
    """Binary search over t_add at fixed t. mode='bpp' or 'pps'."""
    mid = (low + high) / 2
    for _ in range(50):
        mid = (low + high) / 2
        mean_pps, mean_bpp = eval_english_combined(sentences, t, mid, score_idx, source)
        value = mean_bpp if mode == "bpp" else mean_pps
        if abs(value - target) < tolerance:
            return mid, mean_pps, mean_bpp
        if mode == "bpp":
            if mean_bpp < target:
                low = mid   # higher t_add → fewer boundaries → higher bpp
            else:
                high = mid
        else:  # pps: higher t_add → fewer boundaries → lower pps
            if mean_pps > target:
                low = mid
            else:
                high = mid
        if high - low < 1e-6:
            break
    mean_pps, mean_bpp = eval_english_combined(sentences, t, mid, score_idx, source)
    return mid, mean_pps, mean_bpp


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", required=True,
                         help="Directory of per-language JSON files (from run_eval.py) "
                              "containing this model/checkpoint's eng_Latn.json.")
    parser.add_argument("--out-path", default=DEFAULT_OUT_PATH,
                         help=f"Where to save the calibrated thresholds CSV "
                              f"(default {DEFAULT_OUT_PATH}).")
    parser.add_argument("--score-source", choices=SCORE_SOURCES, default="bytes",
                         help="Which per-sentence score sequence to threshold over: "
                              "'bytes' (default, per-byte scores from bytes_entropies) or "
                              "'chars' (per-character SUMMED raw entropy from "
                              "chars_entropies -- run add_char_entropies.py first). "
                              "See module docstring's SCORE SOURCE section.")
    parser.add_argument("--search-high-override", type=float, default=None,
                         help="Replace every case's search_high (see CASES) with this "
                              "value. Recommended for --score-source=chars, e.g. 6.0, "
                              "since summed per-character scores can range higher than "
                              "single-byte scores.")
    parser.add_argument("--target-pps", type=float, default=DEFAULT_TARGET_PPS,
                         help=f"Anchor target: patches-per-sentence for English "
                              f"(default {DEFAULT_TARGET_PPS}, matching the base model's "
                              f"original UTF-8 calibration -- e.g. double this for a "
                              f"checkpoint trained with a fixed 2-bytes-per-character "
                              f"custom encoding, since PPS scales with a sentence's raw "
                              f"byte count for a fixed encoding, see module docstring). "
                              f"NOT affected by --score-source -- pps targets stay the "
                              f"same across bytes/chars runs for the same checkpoint.")
    parser.add_argument("--pps-low", type=float, default=DEFAULT_PPS_LOW,
                         help=f"Lower-bound target: patches-per-sentence for English "
                              f"(default {DEFAULT_PPS_LOW}; higher pps = finer patching, "
                              f"see module docstring for why this is pps- rather than "
                              f"bpp-targeted).")
    parser.add_argument("--pps-high", type=float, default=DEFAULT_PPS_HIGH,
                         help=f"Upper-bound target: patches-per-sentence for English "
                              f"(default {DEFAULT_PPS_HIGH}; lower pps = coarser patching).")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    source = args.score_source

    # Check if the output CSV is specifically base_model_thresholds_summary.csv
    is_base_model = Path(args.out_path).name == "base_model_thresholds_summary.csv"

    eng_path = Path(args.results_dir) / f"{ENGLISH}.json"
    with open(eng_path, encoding="utf-8") as f:
        sentences = json.load(f)

    if source == "chars" and not sentences[0].get("chars_entropies"):
        raise ValueError(
            f"--score-source=chars requires 'chars_entropies' in {eng_path} -- "
            f"run add_char_entropies.py on {args.results_dir} first."
        )

    print(f"Loaded {len(sentences)} English sentences from {eng_path}")
    print(f"Score source            : {source}")
    print(f"Target pps (anchor)     : {args.target_pps}")
    print(f"Target pps (low/high)   : {args.pps_low} / {args.pps_high}\n")

    summary_rows = []

    # Include norm_entropy only if saving to base_model_thresholds_summary.csv
    active_cases = CASES if is_base_model else {
        k: v for k, v in CASES.items() if k in ("raw_entropy", "raw_monotonicity")
    }
    if source == "chars":
        excluded = [k for k, c in active_cases.items() if c["score_idx"] not in CHAR_COMPATIBLE_SCORE_IDX]
        if excluded:
            print(f"NOTE: --score-source=chars has no equivalent for {excluded} "
                  f"(chars_entropies stores no normalized/combined score) -- excluding.\n")
        active_cases = {k: v for k, v in active_cases.items() if k not in excluded}

    # ── standard cases ────────────────────────────────────────────────────────
    for case_name, case in active_cases.items():
        print(f"── {case_name} ──")
        idx   = case["score_idx"]
        mono  = case["monotonicity"]
        slow  = case["search_low"]
        shigh = args.search_high_override if args.search_high_override is not None else case["search_high"]

        t_low, pps_low, bpp_low = binary_search_pps(
            sentences, args.pps_low, idx, mono, slow, shigh, source)
        print(f"  Lower bound  (pps≈{args.pps_low}): threshold={t_low:.4f}  pps={pps_low:.2f}  bpp={bpp_low:.4f}")

        t_high, pps_high, bpp_high = binary_search_pps(
            sentences, args.pps_high, idx, mono, slow, shigh, source)
        print(f"  Upper bound  (pps≈{args.pps_high}): threshold={t_high:.4f}  pps={pps_high:.2f}  bpp={bpp_high:.4f}")

        t_mid = (t_low + t_high) / 2
        pps_mid, bpp_mid = eval_english(sentences, t_mid, idx, mono, source)
        print(f"  Midpoint                  : threshold={t_mid:.4f}  pps={pps_mid:.2f}  bpp={bpp_mid:.4f}")

        t_anchor, pps_anchor, bpp_anchor = binary_search_pps(
            sentences, args.target_pps, idx, mono, slow, shigh, source)
        print(f"  Anchor       (pps≈{args.target_pps}): threshold={t_anchor:.4f}  pps={pps_anchor:.2f}  bpp={bpp_anchor:.4f}")

        thresholds = sorted([t_low, t_mid, t_high, t_anchor])
        print(f"\n  Final 4 thresholds: {[f'{t:.4f}' for t in thresholds]}\n")

        summary_rows.append({
            "case":            case_name,
            "fixed_t":         "",
            "t_low":           t_low,   "pps_low":    pps_low,    "bpp_low":    bpp_low,
            "t_mid":           t_mid,   "pps_mid":    pps_mid,    "bpp_mid":    bpp_mid,
            "t_high":          t_high,  "pps_high":   pps_high,   "bpp_high":   bpp_high,
            "t_anchor":        t_anchor,"pps_anchor":  pps_anchor, "bpp_anchor": bpp_anchor,
            "final_thresholds": thresholds,
        })

    # ── combined mode (only executed for base_model_thresholds_summary.csv,
    #    and only for score-source=bytes -- combined has no char equivalent) ──
    if is_base_model and source == "bytes":
        print("── combined (raw entropy + monotonicity delta) ──")
        idx       = COMBINED["score_idx"]
        t_sl      = COMBINED["t_search_low"]
        t_sh      = args.search_high_override if args.search_high_override is not None else COMBINED["t_search_high"]
        t_add_max = COMBINED["t_add_max"]

        # step 1: fix t so that pps=--pps-low at t_add=0 (lower bound)
        t_fixed, pps_lb, bpp_lb = binary_search_t_for_combined_pps(
            sentences, args.pps_low, idx, t_add=0.0, low=t_sl, high=t_sh, source=source)
        print(f"  Fixed t (pps≈{args.pps_low} at t_add=0): t={t_fixed:.4f}  pps={pps_lb:.2f}  bpp={bpp_lb:.4f}")

        # step 2: upper bound — find t_add giving pps=--pps-high
        t_add_high, pps_ub, bpp_ub = binary_search_t_add(
            sentences, args.pps_high, idx, t_fixed, low=0.0, high=t_add_max, source=source, mode="pps")
        print(f"  Upper bound  (pps≈{args.pps_high}): t_add={t_add_high:.4f}  pps={pps_ub:.2f}  bpp={bpp_ub:.4f}")

        # step 3: midpoint
        t_add_mid = t_add_high / 2
        pps_mid, bpp_mid = eval_english_combined(sentences, t_fixed, t_add_mid, idx, source)
        print(f"  Midpoint                  : t_add={t_add_mid:.4f}  pps={pps_mid:.2f}  bpp={bpp_mid:.4f}")

        # step 4: anchor — find t_add giving target pps
        t_add_anchor, pps_anchor, bpp_anchor = binary_search_t_add(
            sentences, args.target_pps, idx, t_fixed, low=0.0, high=t_add_max, source=source, mode="pps")
        print(f"  Anchor       (pps≈{args.target_pps}): t_add={t_add_anchor:.4f}  pps={pps_anchor:.2f}  bpp={bpp_anchor:.4f}")

        t_adds = sorted([0.0, t_add_mid, t_add_high, t_add_anchor])
        print(f"\n  Fixed t={t_fixed:.4f}, final 4 t_add values: {[f'{t:.4f}' for t in t_adds]}\n")

        summary_rows.append({
            "case":             "combined",
            "fixed_t":          t_fixed,
            "t_low":            0.0,         "pps_low":    pps_lb,      "bpp_low":    bpp_lb,
            "t_mid":            t_add_mid,   "pps_mid":    pps_mid,     "bpp_mid":    bpp_mid,
            "t_high":           t_add_high,  "pps_high":   pps_ub,      "bpp_high":   bpp_ub,
            "t_anchor":         t_add_anchor,"pps_anchor":  pps_anchor,  "bpp_anchor": bpp_anchor,
            "final_thresholds": t_adds,
        })
    elif is_base_model and source == "chars":
        print("── combined mode skipped: no char-level equivalent (see module docstring) ──\n")

    # ── save summary ──────────────────────────────────────────────────────────
    with open(args.out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case", "fixed_t",
            "t_low",    "pps_low",    "bpp_low",
            "t_mid",    "pps_mid",    "bpp_mid",
            "t_high",   "pps_high",   "bpp_high",
            "t_anchor", "pps_anchor", "bpp_anchor",
            "final_thresholds",
        ])
        for r in summary_rows:
            writer.writerow([
                r["case"],
                f"{r['fixed_t']:.4f}" if r["fixed_t"] != "" else "",
                f"{r['t_low']:.4f}",    f"{r['pps_low']:.2f}",    f"{r['bpp_low']:.4f}",
                f"{r['t_mid']:.4f}",    f"{r['pps_mid']:.2f}",    f"{r['bpp_mid']:.4f}",
                f"{r['t_high']:.4f}",   f"{r['pps_high']:.2f}",   f"{r['bpp_high']:.4f}",
                f"{r['t_anchor']:.4f}", f"{r['pps_anchor']:.2f}", f"{r['bpp_anchor']:.4f}",
                str([f"{t:.4f}" for t in r["final_thresholds"]]),
            ])
    print(f"Summary saved → {args.out_path}")


if __name__ == "__main__":
    main()