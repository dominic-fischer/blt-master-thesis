import argparse
import re
import subprocess
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))  # add launch_training.py to PYTHONPATH
from launch_training import build_parser, compute_plan, save_tuned_batch_size, SIZE_NAME_MAP

OOM_PATTERNS = [
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"CUBLAS_STATUS_ALLOC_FAILED", re.IGNORECASE),
    re.compile(r"cuDNN error.*CUDNN_STATUS_ALLOC_FAILED", re.IGNORECASE),
    re.compile(r"OutOfMemoryError", re.IGNORECASE),
]


def looks_like_oom(output: str) -> bool:
    return any(p.search(output) for p in OOM_PATTERNS)


def build_plan_for_batch_size(sweep_args: argparse.Namespace, batch_size: int):
    """Builds the exact launch_training plan for one candidate batch size,
    reusing its own arg parser so every other default (clip, fsdp_type,
    model_dtype, GPU selection, etc.) stays identical to a real run.
    --log-root/--dump-root are overridden so sweep-run artifacts land
    under their own directory trees instead of mixing in with real
    training runs; --gpu-ids/--free-mem-threshold-mib/
    --free-util-threshold-pct are forwarded as-is (only included if the
    user actually passed them -- otherwise launch_training's own
    auto-detection default applies, same as a real run)."""
    parser = build_parser()
    cli = [
        sweep_args.size,
        "--n-gpus", str(sweep_args.n_gpus),
        "--sources", sweep_args.sources,
        "--lr", str(sweep_args.lr),
        "--batch-size", str(batch_size),
        "--probe-steps", str(sweep_args.probe_steps),
        "--base-config", sweep_args.base_config,
        "--log-root", sweep_args.log_root,
        "--dump-root", sweep_args.dump_root,
    ]
    if sweep_args.gpu_ids is not None:
        cli += ["--gpu-ids", sweep_args.gpu_ids]
    if sweep_args.free_mem_threshold_mib is not None:
        cli += ["--free-mem-threshold-mib", str(sweep_args.free_mem_threshold_mib)]
    if sweep_args.free_util_threshold_pct is not None:
        cli += ["--free-util-threshold-pct", str(sweep_args.free_util_threshold_pct)]

    parsed = parser.parse_args(cli)
    return compute_plan(parsed, parser)


def check_shard_warning(sweep_args: argparse.Namespace) -> None:
    """Shard availability depends only on --sources/--base-config/--size, none
    of which vary across candidate batch sizes within a sweep, so this only
    needs to run once per size instead of once per candidate."""
    plan = build_plan_for_batch_size(sweep_args, sweep_args.start_batch_size)
    if plan["shard_warning"]:
        raise SystemExit(
            f"\n{plan['shard_warning']}\n"
            f"Batch-size smoke-testing needs real data loaded (it exercises "
            f"the actual training path), so this has to be fixed before "
            f"continuing -- a missing-shard failure would otherwise look "
            f"like a false OOM below."
        )


def try_batch_size(sweep_args: argparse.Namespace, batch_size: int) -> str:
    """Returns 'ok' or 'oom'. Any non-zero exit is treated as OOM: this script
    is validated to run cleanly otherwise, and launchers like torchrun often
    swallow the real per-rank traceback by default (no --redirects/--tee), so
    a crash with no OOM string in the captured output is far more likely a
    genuine OOM that died before logging than some other latent bug. The
    distinction between a confirmed OOM-message match and an assumed one
    (crash, no OOM text found) is still printed so you can go dig up the full
    per-rank log for a specific case if you ever want to double check."""
    plan = build_plan_for_batch_size(sweep_args, batch_size)

    print(f"  -> trying batch_size={batch_size} "
          f"(GPUs: {plan['cuda_visible_devices']}, {plan['gpu_source']}) ... ",
          end="", flush=True)
    if sweep_args.dry_run:
        print("(dry run, skipped)")
        print("     " + " ".join(plan["cmd"]))
        return "ok"

    result = subprocess.run(plan["cmd"], env=plan["run_env"], capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")

    if result.returncode == 0:
        print("OK")
        return "ok"

    if looks_like_oom(output):
        print("OOM (confirmed by message match)")
        return "oom"

    print("OOM (assumed -- crash with no OOM text in captured output)")
    tail = "\n".join(output.strip().splitlines()[-15:])
    print(f"     last lines of output:\n{tail}\n")
    return "oom"


def find_max_batch_size(sweep_args: argparse.Namespace) -> int:
    check_shard_warning(sweep_args)

    last_good = None
    first_bad = None
    candidate = sweep_args.start_batch_size

    while True:
        status = try_batch_size(sweep_args, candidate)
        if status == "ok":
            last_good = candidate
            if candidate >= sweep_args.max_batch_size:
                # Explicitly tested the ceiling itself and it passed --
                # nothing left to binary-search for.
                return last_good
            candidate = min(candidate * 2, sweep_args.max_batch_size)
        else:  # oom
            first_bad = candidate
            break

    if last_good is None:
        raise SystemExit(
            f"[{sweep_args.size}] Even the smallest batch size tested "
            f"({sweep_args.start_batch_size}) OOMs. Try a smaller --start-batch-size."
        )

    lo, hi = last_good, first_bad
    while hi - lo > 1:
        mid = (lo + hi) // 2
        status = try_batch_size(sweep_args, mid)
        if status == "ok":
            lo = mid
        else:
            hi = mid
    return lo


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("sizes", nargs="+", choices=list(SIZE_NAME_MAP.keys()))
    parser.add_argument("--n-gpus", type=int, required=True)
    parser.add_argument("--gpu-ids", type=str, default=None,
                         help="Forwarded as-is to launch_training's --gpu-ids, e.g. "
                              "'1,2,3,4'. Must contain exactly --n-gpus values. If "
                              "omitted, launch_training auto-detects idle GPUs itself "
                              "(same behavior as a real training launch).")
    parser.add_argument("--free-mem-threshold-mib", type=int, default=None,
                         help="Forwarded to launch_training's flag of the same name; "
                              "only relevant if --gpu-ids is omitted (auto-detection).")
    parser.add_argument("--free-util-threshold-pct", type=int, default=None,
                         help="Forwarded to launch_training's flag of the same name; "
                              "only relevant if --gpu-ids is omitted (auto-detection).")
    parser.add_argument("--sources", choices=["balanced", "imbalanced"], required=True)
    parser.add_argument("--lr", type=float, default=1e-4,
                         help="Placeholder LR for these probe runs only -- doesn't "
                              "need to be tuned yet, we're only testing whether the "
                              "batch size fits in memory, not convergence.")
    parser.add_argument("--probe-steps", type=int, default=5,
                         help="Steps to run per candidate batch size. Peak memory is "
                              "usually hit within the first 1-2 steps (model weights + "
                              "optimizer state + first activations), so this can stay small.")
    parser.add_argument("--start-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=1024,
                         help="Hard ceiling for the exponential search, in case a size "
                              "never OOMs (unlikely at these model scales on a 2080Ti, "
                              "but avoids an infinite doubling loop).")
    parser.add_argument("--safety-margin", type=float, default=0.9,
                         help="Fraction of the converged max batch size to actually "
                              "save/use, as a cushion against memory-profile differences "
                              "between a short probe and a full training run.")
    parser.add_argument("--base-config", default="config_2080ti_template.yaml")
    parser.add_argument("--log-root", default="logs/batch_sizes_sweep_logs",
                         help="Forwarded as launch_training's --log-root, so batch-size "
                              "sweep logs land in their own directory tree instead of "
                              "mixing into logs/<run_name> alongside real training runs.")
    parser.add_argument("--dump-root", default="dumps/batch_sizes_sweep",
                         help="Forwarded as launch_training's --dump-root, so batch-size "
                              "sweep checkpoints/metrics.jsonl land in their own directory "
                              "tree instead of mixing into dumps/<run_name>.")
    parser.add_argument("--tuned-batch-sizes-file",
                         default="training_setup/batch_size/smoke_tested_batch_sizes.json")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the search plan/commands without launching any "
                              "subprocess -- sanity-check before spending real GPU time. "
                              "NOTE: every candidate 'succeeds' in this mode, so the "
                              "search always climbs to --max-batch-size; the printed "
                              "'converged' value is not a real answer, just a preview "
                              "of the commands that would run.")
    return parser


def main():
    top_args = build_arg_parser().parse_args()

    forwarded_fields = [
        "n_gpus", "gpu_ids", "free_mem_threshold_mib", "free_util_threshold_pct",
        "sources", "lr", "probe_steps", "start_batch_size", "max_batch_size",
        "base_config", "log_root", "dump_root", "dry_run",
    ]

    for size in top_args.sizes:
        print(f"\n=== {size} ===")
        # Per-size namespace carrying only the fields find_max_batch_size /
        # try_batch_size actually need, plus the concrete `size` for this
        # iteration (top_args.sizes is the full list and isn't meaningful here).
        sweep_args = argparse.Namespace(
            size=size,
            **{f: getattr(top_args, f) for f in forwarded_fields},
        )

        converged = find_max_batch_size(sweep_args)
        safe = max(1, int(converged * top_args.safety_margin))
        print(f"[{size}] max working batch_size={converged}, "
              f"saving safety-margined value={safe} (margin={top_args.safety_margin:.0%})")
        if top_args.dry_run:
            print(f"[{size}] (dry run -- nothing saved; '{converged}' above is not a real result)")
        else:
            save_tuned_batch_size(top_args.tuned_batch_sizes_file, size, safe)


if __name__ == "__main__":
    main()