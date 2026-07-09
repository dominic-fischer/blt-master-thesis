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
import math
import os
import re
import subprocess
import sys

import launch_training


class Tee:
    """Duplicates writes to both the real stdout and a log file, so the
    sweep's full output (candidate headers, comparison table, save
    confirmation) is saved automatically -- mirroring the same tee
    pattern eval_after_training.sh uses, rather than requiring you to
    remember a shell redirect each time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


DEFAULT_LRS = [4e-4, 1.5e-4, 6e-5, 2.5e-5, 1e-5]
DIVERGE_RATIO = 1.15  # final_bpb / min_bpb above this -> DIVERGING
GRAD_NORM_WATCH_RATIO = 1.3  # final-third / first-third grad_norm mean

# Both the per-rank torchrun logs (attempt_0/<rank>/stdout.log, written by
# launch_training.py via --log-root) and this script's own printed report
# (see main()) live under the same directory -- under logs/, not dumps/,
# matching launch_training.py's own dumps-vs-logs split, and kept together
# since both are "logs about this sweep", not persistent training output.
LR_SWEEP_LOGS_DIR = "logs/lr_sweep_logs"


def parse_lrs(s: str) -> list[float]:
    return [float(tok.strip()) for tok in s.split(",") if tok.strip()]


def round_lr(x: float) -> float:
    """Rounds to 1 decimal digit of scientific-notation mantissa (e.g.
    2.884...e-4 -> 2.9e-4), so auto-generated candidates are readable and
    match the style of hand-picked ones, instead of long floating-point
    tails."""
    return float(f"{x:.1e}")


def log_space_midpoints(low: float, high: float, n: int = 2) -> list[float]:
    """n points strictly between low and high, evenly spaced in LOG space
    (matching the geometric spacing of DEFAULT_LRS) -- e.g. for low=6e-5,
    high=1.5e-4, n=2, splits the log-distance into 3 equal segments and
    returns the 2 interior division points, rounded via round_lr()."""
    log_low = math.log(low)
    log_high = math.log(high)
    step = (log_high - log_low) / (n + 1)
    raw = [math.exp(log_low + step * (i + 1)) for i in range(n)]
    return [round_lr(x) for x in raw]


def compute_refinement_candidates(results: list[dict]) -> tuple[list[float], dict]:
    """Given all currently-known results (any mix of NEW/prior), finds the
    best CONVERGING candidate and its immediate neighbors (by lr value,
    one step up and one step down, regardless of THEIR status -- a
    DIVERGING neighbor is still a valid bracket edge), then returns 2 new
    log-evenly-spaced candidates in each of those two gaps (up to 4 new
    lrs total). If the best candidate is at either edge of the tested
    range, that side has no neighbor to bracket against, so only the
    other side gets refined -- the caller should treat this as a signal
    the search range may need to extend outward on that side instead.
    Returns (new_lrs, info) where info has the best/neighbor lrs and any
    edge warnings, for reporting.
    """
    converging = [r for r in results if r["status"] == "CONVERGING"]
    info = {"best_lr": None, "lower_neighbor": None, "upper_neighbor": None,
            "edge_warning": None}
    if not converging:
        return [], info

    by_lr = sorted(results, key=lambda r: r["lr"])
    lrs_sorted = [r["lr"] for r in by_lr]
    best = min(converging, key=lambda r: (r["min_bpb"], r["final_bpb"]))
    info["best_lr"] = best["lr"]
    best_idx = lrs_sorted.index(best["lr"])

    new_lrs = []
    if best_idx + 1 < len(lrs_sorted):
        upper = lrs_sorted[best_idx + 1]
        info["upper_neighbor"] = upper
        new_lrs += log_space_midpoints(best["lr"], upper, n=2)
    else:
        info["edge_warning"] = ((info["edge_warning"] or "") +
            f"best lr={best['lr']:g} is the HIGHEST candidate tested -- "
            f"consider adding candidates above it too, in case the true "
            f"optimum is even higher. ")

    if best_idx - 1 >= 0:
        lower = lrs_sorted[best_idx - 1]
        info["lower_neighbor"] = lower
        new_lrs += log_space_midpoints(lower, best["lr"], n=2)
    else:
        info["edge_warning"] = ((info["edge_warning"] or "") +
            f"best lr={best['lr']:g} is the LOWEST candidate tested -- "
            f"consider adding candidates below it too, in case the true "
            f"optimum is even lower.")

    # Dedupe: rounding (round_lr) can occasionally collapse a computed
    # midpoint onto an already-tested value, or onto another midpoint
    # from this same round -- drop those so we never relaunch a
    # candidate that already has data.
    existing_lrs = {r["lr"] for r in results}
    seen = set()
    deduped = []
    for lr in new_lrs:
        if lr in existing_lrs or lr in seen:
            continue
        seen.add(lr)
        deduped.append(lr)
    new_lrs = deduped

    return new_lrs, info


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


def discover_candidates(sweep_root: str, size: str, n_gpus: int) -> list[tuple[float, str, str]]:
    """Finds every existing run under sweep_root/entropy_<size>/ whose
    run_name matches this size+n_gpus, regardless of which invocation of
    this script created it (different --probe-steps, different --lrs,
    run days apart -- all included). Since launch_training.py's
    compute_plan() now ALWAYS appends "_lr<value>" as the final suffix
    segment (unconditionally, so directory names are unambiguous about
    which numeric lr was actually used), the lr can be parsed straight
    back out of the directory name.

    Returns a list of (lr, run_name, metrics_jsonl_path) tuples.
    """
    size_root = os.path.join(sweep_root, f"entropy_{size}")
    prefix = f"entropy_{size}_20lang_{n_gpus}gpu"
    candidates = []
    if not os.path.isdir(size_root):
        return candidates
    for name in sorted(os.listdir(size_root)):
        if not name.startswith(prefix):
            continue
        match = re.search(r"_lr(.+)$", name)
        if not match:
            continue  # no lr suffix -- not a run this script's naming scheme produced
        try:
            lr = float(match.group(1))
        except ValueError:
            continue
        metrics_path = os.path.join(size_root, name, "metrics.jsonl")
        candidates.append((lr, name, metrics_path))
    return candidates


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
              sweep_root: str, extra_args: list[str], force_rerun: bool = False,
              report_only: bool = False) -> dict:
    """Automatic reuse-or-run: if this candidate already has non-empty
    metrics.jsonl on disk, its existing data is reused (no relaunch) --
    if not, it's actually launched. --force-rerun always relaunches even
    when data exists; --report-only (see main()) never launches, so a
    missing candidate is simply reported as FAILED instead."""
    parser = launch_training.build_parser()
    argv = [
        size, "--n-gpus", str(n_gpus),
        "--probe-steps", str(probe_steps),
        "--lr", str(lr),
        "--clip", str(clip),
        "--dump-root", os.path.join(sweep_root, f"entropy_{size}"),
        "--log-root", os.path.join(LR_SWEEP_LOGS_DIR, f"entropy_{size}"),
    ] + extra_args
    args = parser.parse_args(argv)
    plan = launch_training.compute_plan(args, parser)

    existing_rows = read_metrics(plan["metrics_jsonl"])
    if existing_rows and not force_rerun:
        print(f"\n=== lr={lr:g} -> {plan['run_name']} (already have data, "
              f"reusing -- pass --force-rerun to relaunch anyway) ===")
        analysis = analyze(existing_rows)
        analysis["lr"] = lr
        analysis["run_name"] = plan["run_name"]
        return analysis

    if report_only:
        print(f"\n=== lr={lr:g} -> {plan['run_name']} (no existing data, "
              f"--report-only prevents launching it) ===")
        return {"lr": lr, "run_name": plan["run_name"], "status": "FAILED",
                "reason": "no existing metrics.jsonl and --report-only set"}

    print(f"\n=== lr={lr:g} -> {plan['run_name']} "
          f"({'rerunning (--force-rerun)' if existing_rows else 'no existing data, running'}) ===")
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
    print("\n" + "=" * 88)
    print(f"{'lr':>10}  {'status':<11} {'min_bpb':>9} {'@step':>7}  "
          f"{'final_bpb':>10} {'grad_growth':>12}  {'source':<6}")
    print("-" * 88)
    for r in sorted(results, key=lambda r: r["lr"], reverse=True):
        source = "NEW" if r.get("from_this_run") else "prior"
        if r["status"] == "FAILED":
            print(f"{r['lr']:>10.1e}  {'FAILED':<11} {r.get('reason', ''):<30}  {source:<6}")
            continue
        flag = " <-- grad_norm growing" if r["grad_norm_growth"] > GRAD_NORM_WATCH_RATIO else ""
        print(f"{r['lr']:>10.1e}  {r['status']:<11} {r['min_bpb']:>9.4f} "
              f"{r['min_bpb_step']:>7} {r['final_bpb']:>10.4f} "
              f"{r['grad_norm_growth']:>11.2f}x  {source:<6}{flag}")
    print("=" * 88)

    converging = [r for r in results if r["status"] == "CONVERGING"]
    if converging:
        # Tie-break on final_bpb (less post-minimum drift = more stable)
        # when min_bpb is equal or very close -- otherwise ties resolve
        # via whatever order discover_candidates()'s directory listing
        # happened to produce, which is alphabetical and meaningless here.
        best = min(converging, key=lambda r: (r["min_bpb"], r["final_bpb"]))
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
    parser.add_argument("--lrs", type=str, default=None,
                         help=f"Comma-separated candidate LRs to run this "
                              f"invocation, highest to lowest doesn't matter "
                              f"(sorted for the report either way). If "
                              f"omitted: with --report-only, runs nothing "
                              f"new and just aggregates/reports on "
                              f"whatever's already logged for this "
                              f"size+n_gpus; otherwise falls back to "
                              f"{DEFAULT_LRS} (each reused automatically if "
                              f"it already has data, launched if not).")
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
    parser.add_argument("--force-rerun", action="store_true",
                         help="By default, each candidate is reused from "
                              "existing metrics.jsonl if present, and only "
                              "actually launched if it's missing -- this is "
                              "now fully automatic, no flag needed for the "
                              "common case. Pass --force-rerun to relaunch "
                              "every requested candidate even if data "
                              "already exists for it.")
    parser.add_argument("--report-only", action="store_true",
                         help="Never launch anything, even for candidates "
                              "missing data (they're reported as FAILED "
                              "instead). Useful to just view/re-aggregate "
                              "whatever's already logged without "
                              "accidentally kicking off new GPU runs.")
    parser.add_argument("--auto-refine", action="store_true",
                         help="After the initial candidates (run this "
                              "invocation + discovered from prior sweeps) "
                              "are analyzed, automatically find the best "
                              "CONVERGING candidate and its two immediate "
                              "neighbors (one lr step up, one down), insert "
                              "2 new log-evenly-spaced candidates in each "
                              "of those two gaps, run those, and repeat for "
                              "--refine-rounds. Each round narrows in "
                              "around wherever the best candidate currently "
                              "sits. If the best candidate is at either "
                              "edge of the tested range, that side can't "
                              "be bracketed and refined -- a warning is "
                              "printed suggesting the range be widened "
                              "there instead.")
    parser.add_argument("--refine-rounds", type=int, default=1,
                         help="Number of --auto-refine rounds to run. Each "
                              "round adds up to 4 new candidates (2 per "
                              "gap around the current best) and re-picks "
                              "the best before the next round.")
    args, extra = parser.parse_known_args()

    log_dir = LR_SWEEP_LOGS_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"entropy_{args.size}_summary.log")
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    print(f"Logging this sweep to: {log_path}")

    if args.lrs is not None:
        lrs = parse_lrs(args.lrs)
    elif args.report_only:
        # No --lrs given, and --report-only means don't launch anything
        # new anyway -- just aggregate/report on whatever's already
        # logged, no need to name any candidates up front.
        lrs = []
    else:
        lrs = DEFAULT_LRS

    if lrs:
        print(f"Candidates this invocation for size={args.size}, "
              f"n_gpus={args.n_gpus}, probe_steps={args.probe_steps}: {lrs} "
              f"(existing data reused automatically; only missing ones "
              f"actually launch)")
    else:
        print(f"No candidates requested (--report-only with no --lrs) -- "
              f"just aggregating whatever's already logged for "
              f"size={args.size}, n_gpus={args.n_gpus}.")

    if extra and extra[0] == "--":
        extra = extra[1:]

    def run_and_discover(lrs_to_run: list[float], force_rerun: bool = False) -> list[dict]:
        for lr in lrs_to_run:
            run_probe(args.size, args.n_gpus, lr, args.probe_steps,
                      args.clip, args.sweep_root, extra,
                      force_rerun=force_rerun, report_only=args.report_only)
        discovered = discover_candidates(args.sweep_root, args.size, args.n_gpus)
        base_parser = launch_training.build_parser()
        just_ran_names = {
            launch_training.compute_plan(
                base_parser.parse_args(
                    [args.size, "--n-gpus", str(args.n_gpus), "--probe-steps",
                     str(args.probe_steps), "--lr", str(lr), "--clip", str(args.clip),
                     "--dump-root", args.sweep_root] + extra),
                base_parser,
            )["run_name"]
            for lr in lrs_to_run
        }
        results = []
        for lr, run_name, metrics_path in discovered:
            rows = read_metrics(metrics_path)
            analysis = analyze(rows)
            analysis["lr"] = lr
            analysis["run_name"] = run_name
            analysis["from_this_run"] = run_name in just_ran_names
            results.append(analysis)
        return results

    # Report and --save-best consider EVERY candidate ever swept for this
    # size+n_gpus under --sweep-root, not just the ones requested this
    # invocation -- so re-running tiny with a single new lr still picks
    # the overall best against every previously-logged candidate,
    # including ones from earlier sweeps (different --probe-steps,
    # different --lrs, run on a different day). run_and_discover's
    # discovery step includes the candidates just run too, since their
    # metrics.jsonl is already on disk by the time it scans.
    results = run_and_discover(lrs, force_rerun=args.force_rerun)
    n_new = sum(1 for r in results if r.get("from_this_run"))
    print(f"\nFound {len(results)} total logged candidate(s) for "
          f"size={args.size}, n_gpus={args.n_gpus} under {args.sweep_root} "
          f"({n_new} from this invocation, {len(results) - n_new} from prior sweeps).")

    if args.auto_refine:
        for round_num in range(1, args.refine_rounds + 1):
            new_lrs, info = compute_refinement_candidates(results)
            if info["edge_warning"]:
                print(f"\n[refine round {round_num}] {info['edge_warning']}")
            if not new_lrs:
                print(f"\n[refine round {round_num}] No CONVERGING candidates "
                      f"(or no room to bracket the best one) -- stopping "
                      f"auto-refine here.")
                break
            neighbor_str = ""
            if info["lower_neighbor"] is not None:
                neighbor_str += f", lower neighbor={info['lower_neighbor']:g}"
            if info["upper_neighbor"] is not None:
                neighbor_str += f", upper neighbor={info['upper_neighbor']:g}"
            print(f"\n[refine round {round_num}] Best so far: "
                  f"lr={info['best_lr']:g}{neighbor_str}")
            print(f"[refine round {round_num}] New candidates: "
                  f"{sorted(new_lrs, reverse=True)}")
            # Refinement candidates are always freshly generated (never
            # already on disk), so the automatic reuse-or-run check in
            # run_probe will naturally launch them regardless of
            # --report-only's effect on the base --lrs from this invocation.
            results = run_and_discover(new_lrs, force_rerun=False)

    print_report(results, args.save_best, args.tuned_lrs_file, args.size)


if __name__ == "__main__":
    main()