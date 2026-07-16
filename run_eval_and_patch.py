"""
run_eval_and_patch.py

Convenience launcher: runs, in order, for a single checkpoint --
    1. model_eval/run_eval.py            -- entropy scores over FLORES+ per language
    2. model_eval/run_patching.py        -- patch lengths per case/threshold
    3. results/results_to_CSV.py         -- aggregates into pps/bpp columns on the master CSV
    4. results/results_to_JS.py          -- converts that CSV to results/results_JS/*.js
    5. results/results_to_txt_premiums.py -- per-mode sorted pps_premium .txt files

-- so you don't have to work out the results directory or output
filenames by hand and invoke five scripts manually.

ASSUMED LAYOUT: this launcher itself stays at repo root; run_eval.py,
run_patching.py, and results_paths.py live under model_eval/;
results_to_CSV.py, results_to_JS.py, and results_to_txt_premiums.py live
under results/. Must be run with repo root as the current working
directory -- all the relative paths involved (floresplus_MASTER.csv,
results/own_models/, etc.) resolve against cwd, not any script's own
file location.

OUTPUT NAMING for steps 3/4/5: all saved using a stem derived from
--entropy_repo (via results_paths.derive_filename_stem), e.g.
"entropy_10M_20lang_4gpu_sourcesbalanced_steps6000_ckpt200_lr4.5e-3_step_0000006000":
    results/results_CSV/<stem>_results.csv
    results/results_JS/<stem>_results.js
    results/txt_premiums/<mode>/<stem>_premiums_sorted.txt  (one per mode)

Usage:
    python run_eval_and_patch.py --entropy_repo dumps/entropy_10M_20lang_4gpu_sourcesbalanced_steps3000_ckpt200_lr4.5e-3/checkpoints/0000006000/consolidated
    python run_eval_and_patch.py --entropy_repo <repo> --results-dir results/own_models/custom_name --summary-csv calibrate_thresholds/thresholds_summary.csv --csv-in-path floresplus_MASTER.csv --langs-csv training_setup/langs/langs_chosen.csv

If a checkpoint's earlier-stage results already exist and you only want
to (re-)run a later step, call that script directly instead -- this
launcher always runs all five steps.
"""
import argparse
import os
import subprocess
import sys
from os import path

sys.path.append(path.join(path.dirname(path.abspath(__file__)), "model_eval"))
from results_paths import results_dir_for, derive_filename_stem, OUTPUT_BASE_DIR

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
             "to derive the results directory and the results_to_CSV.py/"
             "results_to_JS.py output filenames.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override the auto-derived results directory used for steps 1-3. "
             f"If omitted, derived from --entropy_repo under {OUTPUT_BASE_DIR}/ "
             "(same logic run_eval.py itself uses).",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Passed straight through to run_patching.py's --summary-csv "
             "(omit to use its own default).",
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
    csv_out_path = os.path.join(CSV_OUTPUT_DIR, f"{stem}_results.csv")
    js_out_path = os.path.join(JS_OUTPUT_DIR, f"{stem}_results.js")

    print(f"[1/5] Running run_eval.py for --entropy_repo={args.entropy_repo}")
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

    print(f"\n[2/5] Running run_patching.py on {results_dir}")
    patch_cmd = [sys.executable, "model_eval/run_patching.py", "--results-dir", results_dir]
    if args.summary_csv is not None:
        patch_cmd += ["--summary-csv", args.summary_csv]
    subprocess.run(patch_cmd, check=True)

    print(f"\n[3/5] Running results_to_CSV.py -> {csv_out_path}")
    subprocess.run(
        [sys.executable, "results/results_to_CSV.py",
         "--results-dir", results_dir,
         "--csv-in-path", args.csv_in_path,
         "--csv-out-path", csv_out_path],
        check=True,
    )

    print(f"\n[4/5] Running results_to_JS.py -> {js_out_path}")
    subprocess.run(
        [sys.executable, "results/results_to_JS.py",
         "--csv-in-path", csv_out_path,
         "--out-path", js_out_path],
        check=True,
    )

    print(f"\n[5/5] Running results_to_txt_premiums.py (stem={stem})")
    subprocess.run(
        [sys.executable, "results/results_to_txt_premiums.py",
         "--csv-in-path", csv_out_path,
         "--langs-csv", args.langs_csv,
         "--filename-prefix", stem],
        check=True,
    )

    print(f"\nDone: eval + patching + CSV + JS + txt premiums complete.")
    print(f"  CSV:          {csv_out_path}")
    print(f"  JS:           {js_out_path}")
    print(f"  txt premiums: results/txt_premiums/<mode>/{stem}_premiums_sorted.txt")


if __name__ == "__main__":
    main()