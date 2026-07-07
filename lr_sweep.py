"""
lr_sweep.py

Runs a short, checkpoint-free training probe (see launch_training.py's
--probe-steps) for each candidate learning rate, for a given model size,
then reports each candidate's bpb trend so you can pick a sensible LR
before committing to a full run.

Why this exists: the base yaml's optim.lr=4e-4 is inherited from Meta's
repo-scale (dim=768/12h/14L) debug config, not re-tuned per size -- and
this script's own launch_training.py CLI only overrides entropy_model.*
architecture dims, never optim.lr, so nothing automatically adjusts it as
you change size. Confirmed on Tiny: lr=4e-4 diverges (bpb climbs
monotonically, grad_norm climbs unbounded). A single follow-up trial at
lr=1e-4 did NOT diverge in that one run -- but that only shows 1e-4 is
*a* value that avoids the specific failure mode seen at 4e-4, not that
it's a well-searched or near-optimal choice; nothing nearby (higher or
lower) was actually tried before adopting it. Don't treat any value in
this file's DEFAULT_LRS, or any size's current entry in the tuned-LR
lookup, as validated until it's actually been through a sweep -- including
re-sweeping Tiny itself, since 1e-4 there was never really searched
around, just the first non-diverging value tried. There's no reliable
formula for transferring the right LR across architectures in this
codebase, so this script automates the same manual metrics.jsonl
inspection used to find that out for Tiny, for each size in turn.

How each candidate is judged (heuristic, not a substitute for eyeballing
the actual metrics.jsonl if a result looks borderline):
  - min_bpb / min_bpb_step: the lowest train bpb reached, and when.
  - final_bpb: train bpb at the last logged step.
  - status:
      CONVERGING  - final_bpb is close to (or still improving on) min_bpb
                    -> hasn't turned upward, or only mildly so
      DIVERGING   - final_bpb is substantially above min_bpb -> same
                    signature as the confirmed lr=4e-4 failure on Tiny
      FAILED      - NaN/Inf in bpb or grad_norm, or the run's subprocess
                    itself errored (e.g. crashed/OOM) -- see printed
                    stderr tail for that candidate
  - grad_norm_growth: mean grad_norm in the run's final third vs its
    first third. Values well above 1.0 alongside a DIVERGING bpb status
    corroborate a genuine LR-driven instability, not just noise.

This is a heuristic screen for narrowing down candidates quickly, not a
guarantee of the single best LR -- especially for candidates near the
CONVERGING/DIVERGING boundary, it's worth looking at the actual bpb curve
in metrics.jsonl yourself before committing to a full run.

Usage:
    python lr_sweep.py <size> --n-gpus 4 [--lrs 4e-4,1e-4,3e-5,1e-5]
                        [--probe-steps 300] [--clip 10.0]
                        [--sweep-root dumps/lr_sweep]

Example:
    python lr_sweep.py small --n-gpus 4
    python lr_sweep.py medium --n-gpus 4 --lrs 2e-4,6e-5,2e-5 --probe-steps 400
"""

import argparse
import json
import subprocess
import sys

import launch_training


DEFAULT_LRS = [4e-4, 1.5e-4, 6e-5, 2.5e-5, 1e-5]
DIVERGE_RATIO = 1.15  # final_bpb / min_bpb above this -> DIVERGING
GRAD_NORM_WATCH_RATIO = 1.3  # final-third / first-third grad_norm mean


def parse_lrs(s: str) -> list[float]:
    return [float(tok.strip()) for tok in s.split(",") if tok.strip()]


def read_metrics(metrics_jsonl: str) -> list[dict]:
    rows = []
    try:
        with open(metrics_jsonl) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return rows


def analyze(rows: list[dict]) -> dict:
    if not rows:
        return {"status": "FAILED", "reason": "no metrics.jsonl rows found"}

    bpbs = [r.get("bpb/interval_across_gpus") for r in rows]
    grad_norms = [r.get("optim/grad_norm") for r in rows]
    steps = [r.get("global_step") for r in rows]

    for b in bpbs:
        if b is None or b != b or b in (float("inf"), float("-inf")):  # NaN check
            return {"status": "FAILED", "reason": "NaN/Inf in bpb"}
    for g in grad_norms:
        if g is None or g != g or g in (float("inf"), float("-inf")):
            return {"status": "FAILED", "reason": "NaN/Inf in grad_norm"}

    min_bpb = min(bpbs)
    min_bpb_step = steps[bpbs.index(min_bpb)]
    final_bpb = bpbs[-1]
    final_step = steps[-1]

    n = len(grad_norms)
    third = max(1, n // 3)
    first_third_mean = sum(grad_norms[:third]) / third
    last_third_mean = sum(grad_norms[-third:]) / third
    grad_norm_growth = (last_third_mean / first_third_mean) if first_third_mean > 0 else float("inf")

    diverge_ratio = final_bpb / min_bpb if min_bpb > 0 else float("inf")
    status = "DIVERGING" if diverge_ratio > DIVERGE_RATIO else "CONVERGING"

    return {
        "status": status,
        "min_bpb": min_bpb,
        "min_bpb_step": min_bpb_step,
        "final_bpb": final_bpb,
        "final_step": final_step,
        "diverge_ratio": diverge_ratio,
        "grad_norm_growth": grad_norm_growth,
        "n_points": n,
    }


def run_probe(size: str, n_gpus: int, lr: float, probe_steps: int, clip: float,
              sweep_root: str, extra_args: list[str]) -> dict:
    parser = launch_training.build_parser()
    argv = [
        size, "--n-gpus", str(n_gpus),
        "--probe-steps", str(probe_steps),
        "--lr", str(lr),
        "--clip", str(clip),
        "--dump-root", sweep_root,
        "--log-root", f"{sweep_root}_logs",
    ] + extra_args
    args = parser.parse_args(argv)
    plan = launch_training.compute_plan(args, parser)

    print(f"\n=== lr={lr:g} -> {plan['run_name']} ===")
    result = subprocess.run(
        plan["cmd"], env=plan["run_env"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.strip().splitlines()[-15:])
        print(f"  subprocess FAILED (exit {result.returncode}), last lines:\n{tail}")
        return {"lr": lr, "run_name": plan["run_name"], "status": "FAILED",
                "reason": f"subprocess exited {result.returncode}"}

    rows = read_metrics(plan["metrics_jsonl"])
    analysis = analyze(rows)
    analysis["lr"] = lr
    analysis["run_name"] = plan["run_name"]
    return analysis


def print_report(results: list[dict], save_best: bool, tuned_lrs_file: str,
                  size: str) -> None:
    print("\n" + "=" * 78)
    print(f"{'lr':>10}  {'status':<11} {'min_bpb':>9} {'@step':>7}  "
          f"{'final_bpb':>10} {'grad_growth':>12}")
    print("-" * 78)
    for r in sorted(results, key=lambda r: r["lr"], reverse=True):
        if r["status"] == "FAILED":
            print(f"{r['lr']:>10.1e}  {'FAILED':<11} {r.get('reason', ''):<30}")
            continue
        flag = " <-- grad_norm growing" if r["grad_norm_growth"] > GRAD_NORM_WATCH_RATIO else ""
        print(f"{r['lr']:>10.1e}  {r['status']:<11} {r['min_bpb']:>9.4f} "
              f"{r['min_bpb_step']:>7} {r['final_bpb']:>10.4f} "
              f"{r['grad_norm_growth']:>11.2f}x{flag}")
    print("=" * 78)

    converging = [r for r in results if r["status"] == "CONVERGING"]
    if converging:
        best = min(converging, key=lambda r: r["min_bpb"])
        print(f"\nBest CONVERGING candidate by min_bpb: lr={best['lr']:g} "
              f"(min_bpb={best['min_bpb']:.4f} at step {best['min_bpb_step']}, "
              f"run: {best['run_name']})")
        print("This is a heuristic screen, not a guarantee -- worth eyeballing "
              "this candidate's actual bpb curve in its metrics.jsonl, "
              "especially if multiple candidates are close.")
        if save_best:
            launch_training.save_tuned_lr(tuned_lrs_file, size, best["lr"])
            print(f"\nSaved lr={best['lr']:g} for size={size} to "
                  f"{tuned_lrs_file} -- launch_training.py will now use "
                  f"this as {size}'s default --lr automatically (still "
                  f"overridable by passing --lr explicitly).")
        else:
            print(f"\nNot saved (pass --save-best to write this to "
                  f"{tuned_lrs_file} so launch_training.py picks it up "
                  f"automatically as {size}'s default --lr).")
    else:
        print("\nNo candidate converged cleanly -- all diverged or failed. "
              "Try lower lr values, e.g.: --lrs "
              + ",".join(f"{r['lr']/4:.1e}" for r in results if r["status"] != "FAILED"))
        if save_best:
            print("Nothing saved -- --save-best only writes a CONVERGING result.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("size", choices=list(launch_training.SIZE_NAME_MAP.keys()))
    parser.add_argument("--n-gpus", type=int, required=True)
    parser.add_argument("--lrs", type=str, default=",".join(str(x) for x in DEFAULT_LRS),
                         help=f"Comma-separated candidate LRs, highest to lowest "
                              f"doesn't matter (sorted for the report either way). "
                              f"Default: {DEFAULT_LRS} -- a broad first pass; narrow "
                              f"in around whichever candidate looks best.")
    parser.add_argument("--probe-steps", type=int, default=300,
                         help="Steps per candidate. Should be enough to see a "
                              "real minimum and at least some post-minimum "
                              "trend -- 300 was enough to see this clearly on "
                              "Tiny; larger/slower-converging sizes may need more.")
    parser.add_argument("--clip", type=float, default=10.0,
                         help="optim.clip, held fixed across all candidates in "
                              "this sweep (matches launch_training.py's default).")
    parser.add_argument("--sweep-root", default="dumps/lr_sweep",
                         help="Each candidate's probe writes to "
                              "<sweep-root>/entropy_<size>..._probesteps<N>_lr<lr>/, "
                              "kept separate from real dumps/ runs.")
    parser.add_argument("--save-best", action="store_true",
                         help="If a CONVERGING candidate is found, save its "
                              "lr for this size to --tuned-lrs-file, so "
                              "launch_training.py picks it up automatically "
                              "as this size's default --lr on future runs.")
    parser.add_argument("--tuned-lrs-file",
                         default="training_setup/tuned_lrs.json",
                         help="Where --save-best writes results. Must match "
                              "launch_training.py's --tuned-lrs-file (same "
                              "default) for it to actually be picked up.")
    args, extra = parser.parse_known_args()

    lrs = parse_lrs(args.lrs)
    print(f"Sweeping {len(lrs)} LR candidates for size={args.size}, "
          f"n_gpus={args.n_gpus}, probe_steps={args.probe_steps}: {lrs}")

    if extra and extra[0] == "--":
        extra = extra[1:]

    results = []
    for lr in lrs:
        results.append(run_probe(args.size, args.n_gpus, lr, args.probe_steps,
                                  args.clip, args.sweep_root, extra))

    print_report(results, args.save_best, args.tuned_lrs_file, args.size)


if __name__ == "__main__":
    main()