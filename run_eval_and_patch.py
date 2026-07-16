"""
run_eval_and_patch.py

Convenience launcher: runs, in order, for a single checkpoint --
    1. model_eval/run_eval.py             -- entropy scores over FLORES+ per language
    2. model_eval/calibrate_thresholds.py -- recalibrates t_low/t_mid/t_high/t_anchor
                                             for THIS model, so English hits the same
                                             pps/bpp targets the base model was
                                             calibrated against (see step 2 below for
                                             why this matters)
    3. model_eval/run_patching.py         -- patch lengths per case/bound, using the
                                             calibration from step 2
    4. results/results_to_CSV.py          -- aggregates into pps/bpp columns on the master CSV
    5. results/results_to_JS.py           -- converts that CSV to results/results_JS/*.js
    6. results/results_to_txt_premiums.py -- per-bound-type sorted premium .txt files

-- so you don't have to work out the results directory or output
filenames by hand and invoke six scripts manually.

WHY STEP 2 (CALIBRATION) MATTERS: premium = lang_pps / eng_pps only means
the same thing across two models if English is anchored to the same
pps/bpp target each time -- a fixed literal threshold value does NOT
guarantee that, since entropy score distributions can differ in
scale/sharpness between models. Recalibrating per checkpoint (using this
checkpoint's own eng_Latn.json, already produced by step 1) keeps
premiums comparable across different models/checkpoints. Pass
--summary-csv explicitly to skip this and reuse a fixed CSV instead (e.g.
the base model's own calibration) -- doing so means premiums will NOT be
directly comparable to a run that DID recalibrate, so only do this
deliberately.

ASSUMED LAYOUT: this launcher itself stays at repo root; run_eval.py,
calibrate_thresholds.py, run_patching.py, and results_paths.py live under
model_eval/; results_to_CSV.py, results_to_JS.py, and
results_to_txt_premiums.py live under results/. Must be run with repo
root as the current working directory -- all the relative paths involved
(floresplus_MASTER_CSV.csv, results/own_models/, etc.) resolve against
cwd, not any script's own file location.

OUTPUT NAMING for steps 2/4/5/6: all saved using a stem derived from
--entropy_repo (via results_paths.derive_filename_stem), e.g.
"entropy_10M_20lang_4gpu_sourcesbalanced_steps6000_ckpt200_lr4.5e-3_step_0000006000":
    calibrated_thresholds/<stem>_thresholds_summary.csv
    results/results_CSV/<stem>_results.csv
    results/results_JS/<stem>_results.js
    results/txt_premiums/<bound_folder>/<stem>_<case>_premiums_sorted.txt

Usage:
    python run_eval_and_patch.py --entropy_repo dumps/entropy_10M_20lang_4gpu_sourcesbalanced_steps3000_ckpt200_lr4.5e-3/checkpoints/0000006000/consolidated
    python run_eval_and_patch.py --entropy_repo <repo> --results-dir results/own_models/custom_name --csv-in-path floresplus_MASTER.csv --langs-csv training_setup/langs/langs_chosen.csv
    python run_eval_and_patch.py --entropy_repo <repo> --summary-csv calibrated_thresholds/thresholds_summary.csv   # skip recalibration, use a fixed CSV instead

If a checkpoint's earlier-stage results already exist and you only want
to (re-)run a later step, call that script directly instead -- this
launcher always runs all six steps.
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
        "--summary-csv",
        type=str,
        default=None,
        help="Skip step 2 (per-checkpoint calibration) and pass this CSV straight "
             "to run_patching.py's --summary-csv instead -- e.g. to reuse the base "
             "model's own fixed calibration. NOTE: premiums computed this way are "
             "NOT comparable to a run that DID recalibrate (see module docstring), "
             "so only do this deliberately.",
    )
    parser.add_argument(
        "--target-pps",
        type=float,
        default=None,
        help="Passed to calibrate_thresholds.py's --target-pps (omit to use its "
             "own default, matching the base model's original calibration). "
             "Ignored if --summary-csv is given.",
    )
    parser.add_argument(
        "--bpp-low",
        type=float,
        default=None,
        help="Passed to calibrate_thresholds.py's --bpp-low (omit for its default). "
             "Ignored if --summary-csv is given.",
    )
    parser.add_argument(
        "--bpp-high",
        type=float,
        default=None,
        help="Passed to calibrate_thresholds.py's --bpp-high (omit for its default). "
             "Ignored if --summary-csv is given.",
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
             "you need.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Passed to run_eval.py's --force -- re-run the eval even for "
             "languages whose output already exists, instead of the default "
             "skip-if-present behavior.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Passed to run_eval.py's --gpu -- explicit physical GPU index, "
             "skipping auto-detection.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Passed to run_eval.py's --cpu -- force CPU, skipping GPU "
             "auto-detection entirely (much slower).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir is not None else results_dir_for(args.entropy_repo)
    stem = derive_filename_stem(args.entropy_repo)
    calibrated_csv_path = os.path.join(CALIBRATION_OUTPUT_DIR, f"{stem}_thresholds_summary.csv")
    csv_out_path = os.path.join(CSV_OUTPUT_DIR, f"{stem}_results.csv")
    js_out_path = os.path.join(JS_OUTPUT_DIR, f"{stem}_results.js")

    print(f"[1/6] Running run_eval.py for --entropy_repo={args.entropy_repo}")
    print(f"      (results will land under {results_dir}/)")
    eval_cmd = [sys.executable, "model_eval/run_eval.py", "--entropy_repo", args.entropy_repo,
                "--results-dir", results_dir]
    if args.only_trained_langs:
        eval_cmd += ["--only-trained-langs", "--langs-csv", args.langs_csv]
    if args.force:
        eval_cmd += ["--force"]
    if args.gpu is not None:
        eval_cmd += ["--gpu", str(args.gpu)]
    if args.cpu:
        eval_cmd += ["--cpu"]
    subprocess.run(eval_cmd, check=True)

    if args.summary_csv is not None:
        print(f"\n[2/6] Skipping recalibration -- using fixed --summary-csv={args.summary_csv}")
        summary_csv = args.summary_csv
    else:
        print(f"\n[2/6] Running calibrate_thresholds.py -> {calibrated_csv_path}")
        calib_cmd = [sys.executable, "model_eval/calibrate_thresholds.py",
                     "--results-dir", results_dir, "--out-path", calibrated_csv_path]
        if args.target_pps is not None:
            calib_cmd += ["--target-pps", str(args.target_pps)]
        if args.bpp_low is not None:
            calib_cmd += ["--bpp-low", str(args.bpp_low)]
        if args.bpp_high is not None:
            calib_cmd += ["--bpp-high", str(args.bpp_high)]
        subprocess.run(calib_cmd, check=True)
        summary_csv = calibrated_csv_path

    print(f"\n[3/6] Running run_patching.py on {results_dir}")
    subprocess.run(
        [sys.executable, "model_eval/run_patching.py",
         "--results-dir", results_dir, "--summary-csv", summary_csv],
        check=True,
    )

    print(f"\n[4/6] Running results_to_CSV.py -> {csv_out_path}")
    subprocess.run(
        [sys.executable, "results/results_to_CSV.py",
         "--results-dir", results_dir,
         "--csv-in-path", args.csv_in_path,
         "--csv-out-path", csv_out_path],
        check=True,
    )

    print(f"\n[5/6] Running results_to_JS.py -> {js_out_path}")
    subprocess.run(
        [sys.executable, "results/results_to_JS.py",
         "--csv-in-path", csv_out_path,
         "--out-path", js_out_path],
        check=True,
    )

    print(f"\n[6/6] Running results_to_txt_premiums.py (stem={stem})")
    subprocess.run(
        [sys.executable, "results/results_to_txt_premiums.py",
         "--csv-in-path", csv_out_path,
         "--summary-csv", summary_csv,
         "--langs-csv", args.langs_csv,
         "--filename-prefix", stem],
        check=True,
    )

    print(f"\nDone: eval + calibration + patching + CSV + JS + txt premiums complete.")
    print(f"  Calibration:  {summary_csv}")
    print(f"  CSV:          {csv_out_path}")
    print(f"  JS:           {js_out_path}")
    print(f"  txt premiums: results/txt_premiums/<bound_folder>/{stem}_<case>_premiums_sorted.txt")


if __name__ == "__main__":
    main()