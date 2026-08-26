"""
run_eval_and_patch.py

Convenience launcher: runs, in order, for a single checkpoint --
    1. model_eval/run_eval.py             -- entropy scores over FLORES+ per language
    2. model_eval/add_char_entropies.py   -- groups per-byte scores into per-character
                                             summed scores (chars_entropies), needed
                                             for the char-level granularity in steps 3-7
    3. model_eval/calibrate_thresholds.py -- recalibrates t_low/t_mid/t_high/t_anchor
                                             for THIS model, so English hits the same
                                             pps targets the base model was
                                             calibrated against (see step 3 below for
                                             why this matters)
    4. model_eval/run_patching.py         -- patch lengths per case/bound, using the
                                             calibration from step 3
    5. results/results_to_CSV.py          -- aggregates into pps/bpp columns on the master CSV
    6. results/results_to_JS.py           -- converts that CSV to results/results_JS/*.js
    7. results/results_to_txt_premiums.py -- per-bound-type sorted premium .txt files

-- so you don't have to work out the results directory or output
filenames by hand and invoke seven scripts manually (times two, for both
granularities -- see GRANULARITY below).

WHY STEP 3 (CALIBRATION) MATTERS: premium = lang_pps / eng_pps only means
the same thing across two models if English is anchored to the same pps
target each time -- a fixed literal threshold value does NOT guarantee
that, since entropy score distributions can differ in scale/sharpness
between models. Recalibrating per checkpoint (using this checkpoint's own
eng_Latn.json, already produced by step 1) keeps premiums comparable
across different models/checkpoints. Pass --summary-csv explicitly to
skip this and reuse a fixed CSV instead (e.g. the base model's own
calibration) -- doing so means premiums will NOT be directly comparable
to a run that DID recalibrate, so only do this deliberately.

GRANULARITY (--granularity both|bytes|chars, default "both"): steps 3-7
each operate over a "score source" -- either per-BYTE entropy scores
(bytes_entropies, the original behavior) or per-CHARACTER summed entropy
scores (chars_entropies, added by step 2). By default this launcher runs
steps 3-7 ONCE PER GRANULARITY, back to back, so a single invocation
produces both the byte-level and char-level calibration/patching/CSV/JS/
txt-premium outputs -- e.g. to compare whether char-level patching helps
with cross-language fairness versus byte-level. Pass --granularity bytes
or --granularity chars to run only one side (e.g. to iterate faster on
one, or because step 2 was already run separately). Step 1 is always
single-pass regardless (it produces the raw per-byte scores both
granularities are derived from); step 2 is skipped automatically for
--granularity bytes, since nothing downstream needs chars_entropies in
that case.

Byte-mode and char-mode outputs never collide: run_patching.py writes
byte-mode results into each sentence's "eval_modes" key and char-mode
results into "char_eval_modes" (same results-dir, same JSON files, both
can coexist), and steps 3/5/6/7 write char-mode output to a parallel
"char_level" subfolder alongside the byte-mode output (see OUTPUT NAMING
below) rather than overwriting it.

CHAR-MODE SEARCH BOUNDS: summing 1-4 correlated per-byte raw entropy
scores into one per-character score shifts the score's scale upward
versus a single byte's score, so calibrate_thresholds.py's byte-tuned
search_high bounds (~3-4) may not bracket the true char-level threshold.
This launcher always passes --search-high-override 6.0 to the chars-mode
calibration pass to compensate (not currently exposed as a separate CLI
flag here -- edit CHAR_SEARCH_HIGH_OVERRIDE below if you need a different
value). --target-pps/--pps-low/--pps-high, by contrast, are NOT
regranularized -- pps is a sentence-level patch count independent of
what a "unit" is, so the same targets apply to both granularities for a
given checkpoint (see calibrate_thresholds.py's module docstring).

CUSTOM ENCODING: if the checkpoint at --entropy_repo was trained with
launch_training.py's --custom-encoding-path, pass the SAME path here too
via --custom-encoding-path -- forwarded ONLY to step 1 (run_eval.py),
which is the only step that ever converts raw text to bytes (see
blt_patcher.py's patch_text). Step 2 (add_char_entropies.py) instead
AUTO-DETECTS the custom fixed-width encoding from whether "_customenc"
appears in the results directory path (matching this repo's run-naming
convention -- see add_char_entropies.py's module docstring), so no flag
threading is needed there. Steps 3-7 operate purely on already-computed
scores/JSON and never re-encode text, so they need no encoding awareness
at all -- but step 3's --target-pps/--pps-low/--pps-high SHOULD be scaled
to match the encoding's bytes-per-character (e.g. doubled for a fixed
2-bytes-per-char custom encoding), since these are English-sentence-
content targets and PPS scales with raw byte count for a fixed encoding
-- see calibrate_thresholds.py's module docstring for the empirical
justification. (This doubling is unrelated to --granularity: it's the
same doubled target for BOTH the bytes and chars pass of a custom-
encoding checkpoint.)

WHICH STEPS TO RUN: use --steps to select a subset, e.g. --steps 1-4 to
only run eval+char-entropies+calibration+patching, or --steps 5-7 to
only regenerate CSV/JS/txt-premium output from already-existing patching
results, or --steps 3,4,7 for some other combination. Default is "1-7"
(everything). Steps you skip are assumed to have already produced
whatever output later steps need -- this is NOT verified; a skipped
step's missing output will surface as whatever error the first step that
actually needs it raises.

ASSUMED LAYOUT: this launcher itself stays at repo root; run_eval.py,
add_char_entropies.py, calibrate_thresholds.py, run_patching.py, and
results_paths.py live under model_eval/; results_to_CSV.py,
results_to_JS.py, and results_to_txt_premiums.py live under results/.
Must be run with repo root as the current working directory -- all the
relative paths involved (floresplus_MASTER_CSV.csv, results/own_models/,
etc.) resolve against cwd, not any script's own file location.

OUTPUT NAMING for steps 3/5/6/7: all saved using a stem derived from
--entropy_repo (via results_paths.derive_filename_stem), e.g.
"entropy_10M_20lang_4gpu_sourcesbalanced_steps6000_ckpt200_lr4.5e-3_step_0000006000":
    calibrated_thresholds/<stem>_thresholds_summary.csv                        (bytes)
    calibrated_thresholds/char_level/<stem>_thresholds_summary.csv             (chars)
    results/results_CSV/<stem>_results.csv                                     (bytes)
    results/results_CSV/char_level/<stem>_results.csv                          (chars)
    results/results_JS/<stem>_results.js                                       (bytes)
    results/results_JS/char_level/<stem>_results.js                            (chars)
    results/txt_premiums/<bound_folder>/<stem>_<case>_premiums_sorted.txt      (bytes)
    results/txt_premiums/char_level/<bound_folder>/<stem>_<case>_premiums_sorted.txt  (chars)
(step 4, run_patching.py, updates the results-dir JSON files in place for
both granularities -- no separate output path.)

Usage:
    python run_eval_and_patch.py --entropy_repo dumps/entropy_10M_20lang_4gpu_sourcesbalanced_steps3000_ckpt200_lr4.5e-3/checkpoints/0000006000/consolidated
    python run_eval_and_patch.py --entropy_repo <repo> --results-dir results/own_models/custom_name --csv-in-path floresplus_MASTER.csv --langs-csv training_setup/langs/langs_chosen.csv
    python run_eval_and_patch.py --entropy_repo <repo> --summary-csv calibrated_thresholds/thresholds_summary.csv   # skip byte-mode recalibration, use a fixed CSV instead

    # Only byte-level granularity (skip step 2 and the whole chars pass):
    python run_eval_and_patch.py --entropy_repo <repo> --granularity bytes

    # Checkpoint trained with a fixed 2-bytes-per-character custom encoding
    # -- double the pps targets to compensate (applies to BOTH granularities):
    python run_eval_and_patch.py --entropy_repo <repo> \\
        --custom-encoding-path training_setup/custom_encoding.json \\
        --target-pps 65.54 --pps-low 72 --pps-high 46

    # Steps 1-4 already done for this checkpoint (both granularities),
    # only want to regenerate CSV/JS/txt-premiums output:
    python run_eval_and_patch.py --entropy_repo <repo> --steps 5-7

    # Only re-run calibration + patching (e.g. after changing --pps-low/--pps-high),
    # reusing step 1/2's already-computed entropy/char scores:
    python run_eval_and_patch.py --entropy_repo <repo> --steps 3,4
"""
import argparse
import os
import subprocess
import sys
from os import path

sys.path.append(path.join(path.dirname(path.abspath(__file__)), "model_eval"))
from results_paths import results_dir_for, derive_filename_stem, OUTPUT_BASE_DIR

CALIBRATION_OUTPUT_DIR = "calibrated_thresholds"
CSV_OUTPUT_DIR = "results/results_CSV"
JS_OUTPUT_DIR = "results/results_JS"
DEFAULT_LANGS_CSV = "training_setup/langs/langs_chosen.csv"
ALL_STEPS = set(range(1, 8))
GRANULARITIES = ("both", "bytes", "chars")
CHAR_LEVEL_SUBDIR = "char_level"
# See module docstring's CHAR-MODE SEARCH BOUNDS section.
CHAR_SEARCH_HIGH_OVERRIDE = 6.0

# --entropy_repo values that don't naturally produce a useful stem via
# derive_filename_stem() (e.g. no "dumps/<run_name>" component to key off
# of) get a hardcoded override here instead of falling back to
# derive_filename_stem()'s sanitized-path fallback. Add entries here as
# new such --entropy_repo values come up.
ENTROPY_REPO_STEM_OVERRIDES = {
    "hf-weights/entropy_model": "base_model",
}


def parse_steps(spec: str) -> set[int]:
    """Parses a comma-separated list of step numbers and/or ranges (e.g.
    "1-3,7" or "5,6,7" or "2") into a set of ints, each required to be
    between 1 and 7 inclusive (matching this launcher's seven steps)."""
    steps = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            steps.update(range(int(start_s), int(end_s) + 1))
        else:
            steps.add(int(part))
    invalid = steps - ALL_STEPS
    if invalid:
        raise ValueError(
            f"Invalid step number(s) in --steps: {sorted(invalid)} -- "
            f"must be between 1 and 7."
        )
    return steps


def char_path(p: str) -> str:
    """<dir>/<name> -> <dir>/char_level/<name> -- used for the parallel
    char-mode output path of steps 3/5/6 (results_to_txt_premiums.py
    nests itself instead, since it already knows --score-source)."""
    d, name = os.path.split(p)
    return os.path.join(d, CHAR_LEVEL_SUBDIR, name) if d else os.path.join(CHAR_LEVEL_SUBDIR, name)


def run_granularity_pass(source: str, args, steps: set[int], results_dir: str, stem: str):
    """Runs steps 3-7 for one score source ("bytes" or "chars")."""
    is_chars = source == "chars"
    tag = f"[{source}]"

    calibrated_csv_path = os.path.join(
        CALIBRATION_OUTPUT_DIR, f"{stem}_thresholds_summary.csv")
    csv_out_path = os.path.join(CSV_OUTPUT_DIR, f"{stem}_results.csv")
    js_out_path = os.path.join(JS_OUTPUT_DIR, f"{stem}_results.js")
    if is_chars:
        calibrated_csv_path = char_path(calibrated_csv_path)
        csv_out_path = char_path(csv_out_path)
        js_out_path = char_path(js_out_path)

    explicit_summary_csv = args.char_summary_csv if is_chars else args.summary_csv

    if 3 in steps and explicit_summary_csv is None:
        print(f"\n[3/7]{tag} Running calibrate_thresholds.py -> {calibrated_csv_path}")
        calib_cmd = [sys.executable, "model_eval/calibrate_thresholds.py",
                     "--results-dir", results_dir, "--out-path", calibrated_csv_path,
                     "--score-source", source]
        if args.target_pps is not None:
            calib_cmd += ["--target-pps", str(args.target_pps)]
        if args.pps_low is not None:
            calib_cmd += ["--pps-low", str(args.pps_low)]
        if args.pps_high is not None:
            calib_cmd += ["--pps-high", str(args.pps_high)]
        if is_chars:
            calib_cmd += ["--search-high-override", str(CHAR_SEARCH_HIGH_OVERRIDE)]
        subprocess.run(calib_cmd, check=True)
        summary_csv = calibrated_csv_path
    elif explicit_summary_csv is not None:
        print(f"\n[3/7]{tag} Skipping recalibration -- using fixed "
              f"--{'char-' if is_chars else ''}summary-csv={explicit_summary_csv}")
        summary_csv = explicit_summary_csv
    else:
        print(f"\n[3/7]{tag} Skipped (not in --steps) -- assuming {calibrated_csv_path} "
              f"already exists")
        summary_csv = calibrated_csv_path

    if 4 in steps:
        print(f"\n[4/7]{tag} Running run_patching.py on {results_dir}")
        subprocess.run(
            [sys.executable, "model_eval/run_patching.py",
             "--results-dir", results_dir, "--summary-csv", summary_csv,
             "--score-source", source],
            check=True,
        )
    else:
        print(f"\n[4/7]{tag} Skipped (not in --steps) -- assuming patching already exists")

    if 5 in steps:
        print(f"\n[5/7]{tag} Running results_to_CSV.py -> {csv_out_path}")
        subprocess.run(
            [sys.executable, "results/results_to_CSV.py",
             "--results-dir", results_dir,
             "--csv-in-path", args.csv_in_path,
             "--csv-out-path", csv_out_path,
             "--score-source", source],
            check=True,
        )
    else:
        print(f"\n[5/7]{tag} Skipped (not in --steps) -- assuming {csv_out_path} already exists")

    if 6 in steps:
        print(f"\n[6/7]{tag} Running results_to_JS.py -> {js_out_path}")
        subprocess.run(
            [sys.executable, "results/results_to_JS.py",
             "--csv-in-path", csv_out_path,
             "--out-path", js_out_path],
            check=True,
        )
    else:
        print(f"\n[6/7]{tag} Skipped (not in --steps)")

    if 7 in steps:
        print(f"\n[7/7]{tag} Running results_to_txt_premiums.py (stem={stem})")
        subprocess.run(
            [sys.executable, "results/results_to_txt_premiums.py",
             "--csv-in-path", csv_out_path,
             "--summary-csv", summary_csv,
             "--langs-csv", args.langs_csv,
             "--filename-prefix", stem,
             "--score-source", source],
            check=True,
        )
    else:
        print(f"\n[7/7]{tag} Skipped (not in --steps)")

    return summary_csv, csv_out_path, js_out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--entropy_repo",
        type=str,
        required=True,
        help="Passed straight through to run_eval.py's --entropy_repo. Also used "
             "to derive the results directory and all output filenames.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override the auto-derived results directory used for steps 1-4. "
             f"If omitted, derived from --entropy_repo under {OUTPUT_BASE_DIR}/ "
             "(same logic run_eval.py itself uses).",
    )
    parser.add_argument(
        "--granularity",
        choices=GRANULARITIES,
        default="both",
        help="Which score source(s) steps 3-7 run over: 'both' (default, runs "
             "the full 3-7 pass once for bytes and once for chars), 'bytes', "
             "or 'chars'. See module docstring's GRANULARITY section. Step 2 "
             "(add_char_entropies.py) is skipped automatically for 'bytes'.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Skip step 3's actual recalibration for the BYTES pass (even if "
             "step 3 is in --steps) and pass this CSV straight to "
             "run_patching.py's --summary-csv instead -- e.g. to reuse the "
             "base model's own fixed calibration. NOTE: premiums computed "
             "this way are NOT comparable to a run that DID recalibrate (see "
             "calibrate_thresholds.py's module docstring), so only do this "
             "deliberately. Also used, unchanged, as the --summary-csv for "
             "step 7 when step 3 is excluded from --steps. Use "
             "--char-summary-csv for the analogous override on the chars pass.",
    )
    parser.add_argument(
        "--char-summary-csv",
        type=str,
        default=None,
        help="Same as --summary-csv, but for the CHARS pass only. Independent "
             "of --summary-csv -- a bytes-mode calibration CSV cannot be "
             "reused for the chars pass (different score sequence).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="1-7",
        help='Which of the seven steps to run, e.g. "1-7" (default, everything), '
             '"5-7" (only CSV/JS/txt-premiums, reusing existing patching '
             'results), "3,4" (only recalibration+patching, reusing existing '
             'entropy/char scores), etc. See module docstring\'s "WHICH STEPS TO '
             'RUN" section.',
    )
    parser.add_argument(
        "--custom-encoding-path",
        type=str,
        default=None,
        help="Path to the SAME custom_encoding.json passed to "
             "launch_training.py's --custom-encoding-path when this checkpoint "
             "was trained. Forwarded only to step 1 (run_eval.py) -- see module "
             "docstring's CUSTOM ENCODING section. Ignored if step 1 is not in "
             "--steps. Omit for a checkpoint trained on plain UTF-8.",
    )
    parser.add_argument(
        "--target-pps",
        type=float,
        default=None,
        help="Passed to calibrate_thresholds.py's --target-pps for BOTH "
             "granularity passes (omit to use its own default, matching the "
             "base model's original calibration). Ignored if the relevant "
             "--summary-csv/--char-summary-csv is given or step 3 is not in "
             "--steps for that pass.",
    )
    parser.add_argument(
        "--pps-low",
        type=float,
        default=None,
        help="Passed to calibrate_thresholds.py's --pps-low for BOTH "
             "granularity passes (omit for its default).",
    )
    parser.add_argument(
        "--pps-high",
        type=float,
        default=None,
        help="Passed to calibrate_thresholds.py's --pps-high for BOTH "
             "granularity passes (omit for its default).",
    )
    parser.add_argument(
        "--csv-in-path",
        type=str,
        default="floresplus_MASTER.csv",
        help="Master CSV passed to results_to_CSV.py's --csv-in-path "
             "(default: floresplus_MASTER.csv). Left unmodified -- the new "
             "columns are written to the derived results_CSV output instead.",
    )
    parser.add_argument(
        "--langs-csv",
        type=str,
        default=DEFAULT_LANGS_CSV,
        help=f"Passed to results_to_txt_premiums.py's --langs-csv -- its "
             f"'language_code' column restricts which languages appear in the "
             f".txt premium lists (default {DEFAULT_LANGS_CSV}).",
    )
    parser.add_argument(
        "--only-trained-langs",
        action="store_true",
        help="Passed to run_eval.py's --only-trained-langs (using the same "
             "--langs-csv) -- only evaluate/patch the 20 trained languages "
             "instead of the full FLORES+ set, much faster when that's all "
             "you need. Ignored if step 1 is not in --steps.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Passed to run_eval.py's --force -- re-run the eval even for "
             "languages whose output already exists, instead of the default "
             "skip-if-present behavior. Ignored if step 1 is not in --steps.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Passed to run_eval.py's --gpu -- explicit physical GPU index, "
             "skipping auto-detection. Ignored if step 1 is not in --steps.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Passed to run_eval.py's --cpu -- force CPU, skipping GPU "
             "auto-detection entirely (much slower). Ignored if step 1 is not "
             "in --steps.",
    )
    args = parser.parse_args()

    steps = parse_steps(args.steps)
    sources = ["bytes", "chars"] if args.granularity == "both" else [args.granularity]

    results_dir = args.results_dir if args.results_dir is not None else results_dir_for(args.entropy_repo)
    stem = ENTROPY_REPO_STEM_OVERRIDES.get(args.entropy_repo, derive_filename_stem(args.entropy_repo))

    print(f"Running steps: {sorted(steps)}  |  granularity: {args.granularity}\n")

    if 1 in steps:
        print(f"[1/7] Running run_eval.py for --entropy_repo={args.entropy_repo}")
        print(f"      (results will land under {results_dir}/)")
        eval_cmd = [sys.executable, "model_eval/run_eval.py", "--entropy_repo", args.entropy_repo,
                    "--results-dir", results_dir]
        if args.only_trained_langs:
            eval_cmd += ["--only-trained-langs", "--langs-csv", args.langs_csv]
        if args.custom_encoding_path is not None:
            eval_cmd += ["--custom-encoding-path", args.custom_encoding_path]
        if args.force:
            eval_cmd += ["--force"]
        if args.gpu is not None:
            eval_cmd += ["--gpu", str(args.gpu)]
        if args.cpu:
            eval_cmd += ["--cpu"]
        subprocess.run(eval_cmd, check=True)
    else:
        print("[1/7] Skipped (not in --steps) -- assuming results already exist "
              f"under {results_dir}/")

    if "chars" in sources:
        if 2 in steps:
            print(f"\n[2/7] Running add_char_entropies.py on {results_dir}")
            subprocess.run(
                [sys.executable, "model_eval/add_char_entropies.py",
                 "--results-dir", results_dir],
                check=True,
            )
        else:
            print(f"\n[2/7] Skipped (not in --steps) -- assuming chars_entropies "
                  f"already exists under {results_dir}/")
    else:
        print("\n[2/7] Skipped (--granularity=bytes -- chars_entropies not needed)")

    outputs = {}
    for source in sources:
        summary_csv, csv_out_path, js_out_path = run_granularity_pass(
            source, args, steps, results_dir, stem)
        outputs[source] = (summary_csv, csv_out_path, js_out_path)

    print(f"\nDone: steps {sorted(steps)} complete for granularity={args.granularity}.")
    for source, (summary_csv, csv_out_path, js_out_path) in outputs.items():
        bound_dir = "results/txt_premiums" + ("/char_level" if source == "chars" else "")
        print(f"  [{source}]")
        print(f"    Calibration:  {summary_csv}")
        print(f"    CSV:          {csv_out_path}")
        print(f"    JS:           {js_out_path}")
        print(f"    txt premiums: {bound_dir}/<bound_folder>/{stem}_<case>_premiums_sorted.txt")


if __name__ == "__main__":
    main()