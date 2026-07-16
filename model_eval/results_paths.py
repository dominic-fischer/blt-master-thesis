"""
results_paths.py

Shared helper for deriving a results subdirectory from a checkpoint path,
used by both run_eval.py and run_eval_and_patch.py so the logic lives in
exactly one place.
"""
import os

OUTPUT_BASE_DIR = "results/own_models"


def derive_run_subdir(entropy_repo: str) -> str:
    """Uses the path component immediately after "dumps/" in entropy_repo
    as the results subdirectory (e.g.
    "dumps/entropy_10M_..._lr4.5e-3/checkpoints/0000003000/consolidated"
    -> "entropy_10M_..._lr4.5e-3"), so different runs' FLORES+ results
    land in their own folder instead of overwriting each other under one
    flat results/own_models/ directory.

    If a "checkpoints/<step>" component is also present, that step number
    becomes a further subdirectory (e.g. ".../entropy_10M_..._lr4.5e-3/
    step_0000003000/"), so different checkpoints of the SAME run don't
    overwrite each other either -- otherwise evaluating checkpoint 2000
    and then checkpoint 3000 of one run would land in the same folder.

    Falls back to a sanitized version of the whole repo string if "dumps"
    doesn't appear in the path at all -- e.g. a bare HF repo id like
    "hf_weights/entropy_model", which isn't tied to any particular local
    training run or checkpoint."""
    parts = os.path.normpath(entropy_repo).split(os.sep)

    if "dumps" not in parts:
        fallback = entropy_repo.strip(os.sep).replace(os.sep, "_")
        print(f"  NOTE: no 'dumps/<run_name>' component found in "
              f"--entropy_repo={entropy_repo!r} -- using sanitized fallback "
              f"subdirectory name {fallback!r} instead.")
        return fallback

    dumps_idx = parts.index("dumps")
    if dumps_idx + 1 >= len(parts) or not parts[dumps_idx + 1]:
        fallback = entropy_repo.strip(os.sep).replace(os.sep, "_")
        print(f"  NOTE: 'dumps' has no run-name component after it in "
              f"--entropy_repo={entropy_repo!r} -- using sanitized fallback "
              f"subdirectory name {fallback!r} instead.")
        return fallback
    run_name = parts[dumps_idx + 1]

    subdir = run_name
    if "checkpoints" in parts:
        ckpt_idx = parts.index("checkpoints")
        if ckpt_idx + 1 < len(parts) and parts[ckpt_idx + 1]:
            subdir = os.path.join(run_name, f"step_{parts[ckpt_idx + 1]}")

    return subdir


def results_dir_for(entropy_repo: str) -> str:
    """Full results directory (OUTPUT_BASE_DIR/<derived subdir>) for a
    given --entropy_repo path."""
    return os.path.join(OUTPUT_BASE_DIR, derive_run_subdir(entropy_repo))


def derive_filename_stem(entropy_repo: str) -> str:
    """Like derive_run_subdir, but flattens the result into a single
    filename-safe stem (path separators replaced with underscores)
    instead of a nested directory path -- e.g.
    'entropy_10M_..._lr4.5e-3_step_0000006000', for naming one output
    file per run+checkpoint (results_to_CSV.py / results_to_JS.py)
    rather than a results subdirectory."""
    return derive_run_subdir(entropy_repo).replace(os.sep, "_")