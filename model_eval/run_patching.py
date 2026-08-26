"""
run_patching.py

Computes patch lengths for all cases and thresholds and stores results
in the restructured JSON files, under "eval_modes" (--score-source=bytes,
default) or "char_eval_modes" (--score-source=chars).

Cases and thresholds are read from --summary-csv (default
calibrated_thresholds/thresholds_summary.csv, or a per-checkpoint
recalibrated CSV from calibrate_thresholds.py -- pass the CSV produced by
the matching --score-source run of that script; a chars-mode CSV and a
bytes-mode CSV are NOT interchangeable, since they were calibrated over
different score sequences). For standard cases, the threshold column
values (t_low, t_mid, t_high, t_anchor) are used directly as the
patching threshold. For the combined case (bytes-only, no char-mode
equivalent -- see calibrate_thresholds.py), fixed_t is the entropy
threshold and t_low/t_mid/t_high/t_anchor are threshold_add values.

SCORE SOURCE (--score-source bytes|chars): mirrors calibrate_thresholds.py.
  - bytes (default): thresholds a per-BYTE score sequence
    (sentence["bytes_entropies"][i][score_idx]). Patch lengths are
    naturally in bytes already -- one unit = one byte.
  - chars: thresholds the per-CHARACTER summed raw entropy sequence
    (sentence["chars_entropies"][i][1], added by add_char_entropies.py --
    run that first). Patch lengths come out in CHARACTERS, not bytes, so
    each patch's actual byte length is reconstructed by summing the
    n_bytes (chars_entropies[i][2]) of the characters it spans. Both are
    stored per patch (see OUTPUT SHAPE below) so downstream bpp
    computation doesn't need to re-derive byte lengths from
    chars_entropies itself. Only cases with score_idx=1 (raw_entropy,
    raw_monotonicity) are valid for chars-mode -- norm_entropy and
    combined are skipped with a warning if present in --summary-csv,
    since chars_entropies has no normalized-score or combined-mode
    equivalent.

OUTPUT SHAPE:
  --score-source=bytes (unchanged from before):
      eval_modes[case][t_key] = {
          "n_patches": int, "avg_bytes_per_patch": float,
          "patch_lengths": [int, ...],      # in bytes
      }
  --score-source=chars (new):
      char_eval_modes[case][t_key] = {
          "n_patches": int, "avg_bytes_per_patch": float,
          "patch_lengths_chars": [int, ...],  # in characters
          "patch_lengths_bytes": [int, ...],  # in bytes (reconstructed)
      }
  The two top-level keys ("eval_modes" / "char_eval_modes") never
  collide, so a bytes-mode and a chars-mode run can both be applied to
  the same results-dir JSON files without clobbering each other.

KEYED BY EXACT NUMERIC THRESHOLD (as originally): entries are stored as
<eval_modes_key>[case_name]["t_<value>"], e.g. "t_1.3340" -- the literal
calibrated threshold, not a bound label. Bound identity (low/mid/high/
anchor) is intentionally NOT baked in here; it's reconstructed downstream
by results_to_txt_premiums.py, which cross-references the same
--summary-csv this script used to figure out which numeric value
corresponds to which bound, per case. This keeps --summary-csv as the
single source of truth for that mapping instead of duplicating it into
the patching output. (Iterating the 4 named bounds from the CSV, rather
than blindly deduplicating via a set as an earlier version of this script
did, still avoids silently losing an entry if two bounds happen to
coincide numerically for a given case -- the second one just correctly
reuses the already-computed result instead of redoing it.)

STALE CLEANUP -- TWO LEVELS, applied independently within whichever of
eval_modes/char_eval_modes this run's --score-source targets (the other
key, if present from a prior run of the opposite --score-source, is left
untouched): any CASE (raw_entropy, raw_monotonicity, etc.) not present in
the CSV is removed entirely. Additionally, WITHIN each still-valid case,
any THRESHOLD KEY (e.g. "t_0.5928") not among that case's current 4
named-bound values in the CSV is also removed. This second level matters
whenever a checkpoint gets recalibrated (e.g. a change in --target-pps/
--pps-low/--pps-high, or switching calibration methodology entirely, as
happened when this script's bpp-based bounds became pps-based) -- the OLD
threshold values computed under the previous calibration would otherwise
silently persist forever, since they still sit under a valid case name
and only the case-name-level cleanup ran previously. Leftover stale
threshold keys don't corrupt anything by themselves, but they do produce
extra *_pps_premium columns downstream (via results_to_CSV.py) that
results_to_txt_premiums.py then can't map to a bound and skips with a
warning -- cleaning them up here removes that noise at the source.

Output: updates <results-dir>/{lang_code}.json in place (indented)

Usage (from repo root):
    python model_eval/run_patching.py --results-dir results/own_models/entropy_10M_..._lr4.5e-3/step_0000006000
    python model_eval/run_patching.py --results-dir <dir> --summary-csv calibrated_thresholds/thresholds_summary.csv
    # Char-level score source (requires chars_entropies + a chars-mode calibration CSV):
    python model_eval/run_patching.py --results-dir <dir> --score-source chars --summary-csv calibrated_thresholds/char_level/<stem>_thresholds_summary.csv
"""

import argparse
import csv
import json
import os
import torch
from pathlib import Path
from tqdm import tqdm
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))  # noqa: E402
from bytelatent.data.patcher import (
    find_entropy_patch_start_ids,
    patch_lengths_from_start_ids,
)

DEFAULT_SUMMARY_CSV = "calibrated_thresholds/base_model_thresholds_summary.csv"
SCORE_SOURCES = ("bytes", "chars")
EVAL_MODES_KEY = {"bytes": "eval_modes", "chars": "char_eval_modes"}
# Only score_idx=1 (raw entropy) has a per-character equivalent -- see
# module docstring's SCORE SOURCE section.
CHAR_COMPATIBLE_SCORE_IDX = {1}

# maps case name → which score index to use
SCORE_IDX = {
    "raw_entropy":      1,
    "raw_monotonicity": 1,
    "norm_entropy":     2,
    "combined":         1,
}

# maps case name → monotonicity flag (combined uses threshold_add instead)
MONOTONICITY = {
    "raw_entropy":      False,
    "raw_monotonicity": True,
    "norm_entropy":     False,
    "combined":         False,
}


# ── load cases from CSV ───────────────────────────────────────────────────────

BOUND_NAMES = ["low", "mid", "high", "anchor"]


def load_cases(csv_path: str, score_source: str = "bytes") -> dict:
    """
    Returns a dict keyed by case name. Each value is a dict with:
      - score_idx:         int
      - monotonicity:      bool
      - fixed_threshold:   float or None  (only for combined)
      - named_thresholds:  dict[str, float], keys "low"/"mid"/"high"/"anchor"
                           (t_add values for combined, else t values)

    For score_source="chars", cases whose score_idx has no char-level
    equivalent (see CHAR_COMPATIBLE_SCORE_IDX) are silently excluded --
    they should not appear in a properly-generated chars-mode
    --summary-csv anyway (calibrate_thresholds.py already excludes them
    there), but this guards against a stale/mismatched CSV being passed
    in by mistake.
    """
    cases = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["case"]
            if name not in SCORE_IDX:
                continue
            if score_source == "chars" and SCORE_IDX[name] not in CHAR_COMPATIBLE_SCORE_IDX:
                continue
            named_thresholds = {
                "low":    float(row["t_low"]),
                "mid":    float(row["t_mid"]),
                "high":   float(row["t_high"]),
                "anchor": float(row["t_anchor"]),
            }
            fixed_t = float(row["fixed_t"]) if row["fixed_t"] else None
            cases[name] = {
                "score_idx":        SCORE_IDX[name],
                "monotonicity":     MONOTONICITY[name],
                "fixed_threshold":  fixed_t,
                "named_thresholds": named_thresholds,
            }
    return cases


# ── score / byte-count extraction ─────────────────────────────────────────────

def get_scores_and_byte_counts(sentence: dict, score_idx: int, source: str) -> tuple[list[float], list[int]]:
    """See calibrate_thresholds.py's identical helper -- kept in sync
    conceptually, duplicated here rather than imported to keep this
    script runnable standalone against just bytelatent.data.patcher."""
    if source == "bytes":
        scores = [be[score_idx] for be in sentence["bytes_entropies"]]
        byte_counts = [1] * len(scores)
    else:  # chars
        scores = [ce[1] for ce in sentence["chars_entropies"]]
        byte_counts = [ce[2] for ce in sentence["chars_entropies"]]
    return scores, byte_counts


def byte_lengths_for_patches(patch_lengths_units: list[int], byte_counts: list[int]) -> list[int]:
    lengths_bytes = []
    idx = 0
    for pl in patch_lengths_units:
        lengths_bytes.append(sum(byte_counts[idx:idx + pl]))
        idx += pl
    return lengths_bytes


# ── patch helper ──────────────────────────────────────────────────────────────

def compute_patch_lengths(
    scores: list[float],
    threshold: float,
    threshold_add: float | None,
    monotonicity: bool,
) -> list[int]:
    """Returns patch lengths in UNITS (one unit = one score entry, i.e.
    one byte in bytes-mode or one character in chars-mode)."""
    bos_score = 99.0
    scores_with_bos = torch.tensor([[bos_score] + scores])
    patch_start_ids = find_entropy_patch_start_ids(
        scores_with_bos,
        threshold=threshold,
        threshold_add=threshold_add,
        monotonicity=monotonicity,
    )
    n_units = len(scores)
    try:
        patch_lengths = patch_lengths_from_start_ids(patch_start_ids, n_units + 1)
        return [l for l in patch_lengths[0].tolist()[1:] if l > 0]
    except AssertionError:
        return [n_units]


def threshold_key(threshold: float) -> str:
    return f"t_{threshold:.4f}"


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute patch lengths for all cases/thresholds over an "
                    "existing run_eval.py results directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory of per-language JSON files produced by run_eval.py "
             "(e.g. results/own_models/<run>/step_<step>/). Updated in place.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=DEFAULT_SUMMARY_CSV,
        help=f"CSV defining cases and thresholds (default {DEFAULT_SUMMARY_CSV}). "
             f"Must be from a calibrate_thresholds.py run with the SAME "
             f"--score-source as this run.",
    )
    parser.add_argument(
        "--score-source",
        choices=SCORE_SOURCES,
        default="bytes",
        help="Which per-sentence score sequence to threshold over -- must "
             "match the --score-source used to produce --summary-csv. "
             "'bytes' (default) writes to sentence['eval_modes']; 'chars' "
             "writes to sentence['char_eval_modes'] and requires "
             "chars_entropies to already exist (run add_char_entropies.py "
             "first). See module docstring.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    source = args.score_source
    modes_key = EVAL_MODES_KEY[source]

    cases = load_cases(args.summary_csv, source)
    valid_case_names = set(cases.keys())

    # Per case, the set of threshold keys ("t_<value>") that are current
    # under this --summary-csv -- used for the threshold-level stale
    # cleanup (see module docstring's STALE CLEANUP section). Computed
    # once here, outside the per-sentence loop, since it's the same for
    # every sentence/language.
    valid_keys_per_case = {
        case_name: {threshold_key(case["named_thresholds"][b]) for b in BOUND_NAMES}
        for case_name, case in cases.items()
    }

    print(f"Score source: {source}  (writing to sentence['{modes_key}'])")
    print(f"Loaded {len(cases)} case(s) from {args.summary_csv}:")
    for name, case in cases.items():
        bounds_str = ", ".join(f"{b}={case['named_thresholds'][b]:.4f}" for b in BOUND_NAMES)
        if case["fixed_threshold"] is not None:
            print(f"  {name}: fixed_t={case['fixed_threshold']:.4f}, t_add: {bounds_str}")
        else:
            print(f"  {name}: {bounds_str}")
    print()

    paths = sorted(Path(args.results_dir).glob("*.json"))
    print(f"Found {len(paths)} language files in {args.results_dir}\n")

    for path in paths:
        lang_code = path.stem
        with open(path, encoding="utf-8") as f:
            sentences = json.load(f)

        if source == "chars" and sentences and "chars_entropies" not in sentences[0]:
            print(f"  {lang_code}: SKIPPED -- no 'chars_entropies' found; "
                  f"run add_char_entropies.py on {args.results_dir} first.")
            continue

        stale_threshold_count = 0
        for sentence in tqdm(sentences, desc=lang_code, leave=False):
            if modes_key not in sentence:
                sentence[modes_key] = {}

            # Level 1: remove stale CASES not present in the CSV at all.
            stale_cases = [k for k in sentence[modes_key] if k not in valid_case_names]
            for k in stale_cases:
                del sentence[modes_key][k]

            for case_name, case in cases.items():
                if case_name not in sentence[modes_key]:
                    sentence[modes_key][case_name] = {}

                # Level 2: WITHIN this still-valid case, remove stale
                # THRESHOLD KEYS left over from a previous calibration.
                stale_keys = [
                    k for k in sentence[modes_key][case_name]
                    if k not in valid_keys_per_case[case_name]
                ]
                for k in stale_keys:
                    del sentence[modes_key][case_name][k]
                    stale_threshold_count += 1

                scores, byte_counts = get_scores_and_byte_counts(sentence, case["score_idx"], source)
                is_combined = case["fixed_threshold"] is not None

                for bound_name in BOUND_NAMES:
                    t = case["named_thresholds"][bound_name]
                    key = threshold_key(t)
                    if key in sentence[modes_key][case_name]:
                        continue  # already computed (this exact value), skip

                    if is_combined:
                        threshold     = case["fixed_threshold"]
                        threshold_add = t
                    else:
                        threshold     = t
                        threshold_add = None

                    lengths_units = compute_patch_lengths(
                        scores,
                        threshold=threshold,
                        threshold_add=threshold_add,
                        monotonicity=case["monotonicity"],
                    )
                    lengths_bytes = byte_lengths_for_patches(lengths_units, byte_counts)
                    n_patches = len(lengths_units)
                    n_bytes   = sum(lengths_bytes)
                    avg_bpp   = n_bytes / n_patches if n_patches > 0 else 0.0

                    if source == "bytes":
                        sentence[modes_key][case_name][key] = {
                            "n_patches":           n_patches,
                            "avg_bytes_per_patch": round(avg_bpp, 4),
                            "patch_lengths":       lengths_units,  # units == bytes here
                        }
                    else:
                        sentence[modes_key][case_name][key] = {
                            "n_patches":           n_patches,
                            "avg_bytes_per_patch": round(avg_bpp, 4),
                            "patch_lengths_chars": lengths_units,
                            "patch_lengths_bytes": lengths_bytes,
                        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)

        stale_note = f" (removed {stale_threshold_count} stale threshold entr{'y' if stale_threshold_count == 1 else 'ies'})" if stale_threshold_count else ""
        print(f"  {lang_code}: done{stale_note}")

    print("\nDone.")


if __name__ == "__main__":
    main()