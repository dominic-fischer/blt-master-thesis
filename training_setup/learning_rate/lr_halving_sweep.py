"""
lr_halving_sweep.py

Finds a good learning rate for each entropy-model size using a simple,
cheap technique instead of a fixed grid:

  1. HALVING DESCENT (per size): start at an upper-bound LR, run a short
     probe, then repeatedly halve the LR and probe again. Keep going as
     long as each step is BETTER (lower bpb) than the step before it.
     The moment a step is WORSE than the previous one, stop -- we've
     stepped past the optimum.

  2. BRACKET REFINEMENT: once descent stops, take the best point found
     and its two neighbors (one step up, one step down, in LR-space) and
     probe 1 new log-spaced (bisecting) point inside each of those two
     gaps. Repeat for --refine-rounds, re-centering on the updated best
     each round. This narrows in on the optimum without ever running a
     wasteful full grid.

  3. TIE-BREAKING: after refinement, if the best candidate is tied (to
     --tie-break-decimals digits) with an LR-adjacent neighbor, that's a
     sign two grid points landed on essentially the same performance --
     which one "wins" the tie is just an artifact of Python's min()
     picking the first-seen candidate, not evidence either LR is
     actually better. Probe the geometric midpoint of the tied pair to
     see if a sharper optimum lies between them. This runs irrespective
     of --refine-rounds, since resolving a tie is a data-quality
     question, not a matter of how many bisection rounds were budgeted.

  4. CHAINING ACROSS SIZES: sizes are tuned smallest to largest, in the
     order given by --sizes. The FIRST (smallest) size starts its
     descent from --start-lr (default 4e-4, BLT's reported value).
     EVERY SUBSEQUENT size starts its descent from the next multiple of
     --start-lr above the previous size's tuned best LR (e.g. on the
     default 4e-4 grid: a previous best of 3e-4 chains to 4e-4, and a
     previous best of 5.5e-4 chains to 8e-4) -- since bigger models need
     LR <= smaller model's LR in practice, this gives the descent a
     real, round-number upper bound to start from rather than
     re-testing values we already know are too high.

This assumes bigger sizes need a smaller-or-equal LR than smaller ones.
That's the common pattern but isn't guaranteed -- check the per-size
report if a size's best ends up landing right at its upper bound (the
report will flag this), since that's a sign the true optimum for that
size might be ABOVE the previous size's winner, and the chaining
assumption doesn't hold here.

Usage:
    python lr_halving_sweep.py --sizes 10M,50M,100M --n-gpus 4
    python lr_halving_sweep.py --sizes 10M,50M,100M --n-gpus 4 \\
        --start-lr 4e-4 --probe-steps 300 --refine-rounds 2 --save-best

Requires launch_training.py in the same directory, providing:
    build_parser(), compute_plan(args, parser), save_tuned_lr(path, size, lr),
    SIZE_NAME_MAP
(same interface the old lr_sweep.py relied on).
"""

import argparse
import json
import math
import os
import subprocess
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))  # for launch_training import
import launch_training


# --- tunables ---
DIVERGE_RATIO = 1.15          # final_bpb / min_bpb above this -> DIVERGING
GRAD_NORM_WATCH_RATIO = 1.3   # final-third / first-third grad_norm mean, just a flag
DEFAULT_START_LR = 4e-4       # BLT's own reported LR; only used for the SMALLEST size
DEFAULT_MIN_LR_FLOOR = 1e-6   # safety stop for the halving loop
MAX_UPWARD_EXTENSIONS = 6     # safety cap on how many times we'll double past the top edge
MAX_TIE_BREAK_ROUNDS = 3      # safety cap on how many tie-break midpoints we'll chase
TIE_BREAK_DECIMALS = 4        # min_bpb agreement (decimal digits) to call something a "tie"
LR_SWEEP_LOGS_DIR = "logs/lr_halving_sweep"
DEFAULT_SWEEP_ROOT = "dumps/lr_halving_sweep"


class Tee:
    """Duplicates writes to real stdout and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def round_lr(x: float) -> float:
    """Round to 1 decimal digit of scientific-notation mantissa, e.g.
    2.884e-4 -> 2.9e-4, so candidate LRs stay readable."""
    return float(f"{x:.1e}")


def next_grid_multiple(value: float, quantum: float) -> float:
    """Smallest multiple of `quantum` that is STRICTLY GREATER than `value`,
    e.g. next_grid_multiple(3e-4, 4e-4) -> 4e-4, and
    next_grid_multiple(5.5e-4, 4e-4) -> 8e-4 (since 5.5e-4 sits between
    the 4e-4 and 8e-4 grid points, the next one up is 8e-4). Even if
    `value` already lands exactly on a grid point, this always steps up
    one more multiple -- we want a real upper bound for the next size's
    descent to start from, not the previous winner itself. The small
    epsilon guards against value/quantum landing at e.g. 1.9999999999998
    instead of 2.0 due to float representation."""
    k = math.floor(value / quantum + 1e-9) + 1
    return round_lr(k * quantum)


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
        if b is None or b != b or b in (float("inf"), float("-inf")):
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


def candidate_metric(result: dict) -> float:
    """Lower is better. CONVERGING candidates are ranked by min_bpb;
    DIVERGING/FAILED candidates are treated as maximally bad so they
    never get picked as 'best' and always count as 'worse'."""
    if result["status"] == "CONVERGING":
        return result["min_bpb"]
    return float("inf")


def parse_step_map(s: str) -> dict:
    """Parses 'size=steps,size=steps,...' into {size: steps}. Empty string
    -> {}."""
    result = {}
    if not s:
        return result
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Bad entry {pair!r} in step map (expected 'size=steps')")
        k, v = pair.split("=", 1)
        result[k.strip()] = int(v.strip())
    return result


def load_tuned_lrs(tuned_lrs_file: str) -> dict:
    """Reads the flat {size: lr} json written by launch_training.save_tuned_lr,
    e.g. {"10M": 0.0001, "50M": 0.00005}. Returns {} if the file doesn't
    exist yet or isn't valid JSON. Lets you seed a size's start bound from
    a PREVIOUS invocation's results (e.g. tuning 10M and 50M today, then
    100M tomorrow) instead of requiring the whole chain in one run."""
    try:
        with open(tuned_lrs_file) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {k: float(v) for k, v in data.items()}


def run_probe(size: str, n_gpus: int, lr: float, probe_steps: int, clip: float,
              sources: str, sweep_root: str, extra_args: list[str], force_rerun: bool) -> dict:
    """Reuses existing metrics.jsonl for this (size, n_gpus, lr, probe_steps,
    sources) combo if present; otherwise actually launches the probe."""
    parser = launch_training.build_parser()
    argv = [
        size, "--n-gpus", str(n_gpus),
        "--sources", sources,
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
        print(f"  lr={lr:.3g} -> {plan['run_name']}  (reusing existing data)")
        result = analyze(existing_rows)
    else:
        print(f"  lr={lr:.3g} -> {plan['run_name']}  "
              f"({'rerunning' if existing_rows else 'running'})")
        proc = subprocess.run(
            plan["cmd"], env=plan["run_env"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if proc.returncode != 0:
            tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
            print(f"    subprocess FAILED (exit {proc.returncode}), last lines:\n{tail}")
            result = {"status": "FAILED", "reason": f"subprocess exited {proc.returncode}"}
        else:
            result = analyze(read_metrics(plan["metrics_jsonl"]))

    result["lr"] = lr
    result["run_name"] = plan["run_name"]
    _print_candidate_line(result)
    return result


def _print_candidate_line(r: dict) -> None:
    if r["status"] == "FAILED":
        print(f"    -> FAILED ({r.get('reason', '')})")
        return
    flag = "  <-- grad_norm growing" if r["grad_norm_growth"] > GRAD_NORM_WATCH_RATIO else ""
    print(f"    -> {r['status']:<11} min_bpb={r['min_bpb']:.4f} "
          f"(@step {r['min_bpb_step']})  final_bpb={r['final_bpb']:.4f}{flag}")


def halving_descent(size: str, n_gpus: int, start_ub: float, probe_steps: int,
                     clip: float, sources: str, sweep_root: str, extra_args: list[str],
                     min_lr_floor: float, force_rerun: bool, patience: int = 1) -> tuple[list[dict], str]:
    """Probe start_ub, then start_ub/2, /4, /8, ... tracking the best
    (lowest min_bpb) step seen so far. Tolerates up to `patience`
    consecutive steps that come in worse than the running best -- a
    single noisy probe won't end the descent early, since the next
    (lower) step gets a chance to beat the best and reset the streak.
    Only stops once MORE than `patience` consecutive steps in a row have
    failed to beat the running best, or once min_lr_floor is hit.
    The actual best point is picked later from the full results list
    (by print_size_report / compute_refinement_candidates), so it's
    unaffected by any extra steps probed past it during a tolerated
    streak."""
    print(f"\n--- halving descent for size={size}, starting from lr={start_ub:.3g}, "
          f"patience={patience} ---")
    results = []
    lr = start_ub
    best_metric = None
    bad_streak = 0
    stop_reason = "hit min_lr_floor without getting worse -- consider lowering --min-lr-floor"
    while lr >= min_lr_floor:
        res = run_probe(size, n_gpus, lr, probe_steps, clip, sources, sweep_root, extra_args, force_rerun)
        results.append(res)
        cur_metric = candidate_metric(res)
        if best_metric is None or cur_metric <= best_metric:
            best_metric = cur_metric
            bad_streak = 0
        else:
            bad_streak += 1
            print(f"  (worse than best so far: {bad_streak} consecutive worse step(s), "
                  f"tolerating up to {patience})")
            if bad_streak > patience:
                stop_reason = (f"lr={lr:.3g} made {bad_streak} consecutive steps worse than "
                                f"the running best (best_metric={best_metric:.4f}) -- stopping descent")
                break
        lr = round_lr(lr / 2)
    print(f"  {stop_reason}")
    return results, stop_reason


def log_space_midpoints(low: float, high: float, n: int) -> list[float]:
    """n points evenly spaced in log-space strictly between low and high."""
    log_low, log_high = math.log(low), math.log(high)
    step = (log_high - log_low) / (n + 1)
    return [round_lr(math.exp(log_low + step * (i + 1))) for i in range(n)]


def compute_refinement_candidates(results: list[dict], n_midpoints: int = 1) -> tuple[list[float], dict]:
    """Finds the best CONVERGING candidate among ALL results so far, plus
    its immediate LR-neighbors (up and down), and proposes n_midpoints
    new log-spaced candidates inside each of those two gaps. If the best
    sits at either edge of the tested range, that side can't be
    bracketed -- flagged via info['edge_warning'] instead.

    Default n_midpoints=1 just bisects each gap (each gap is only one
    halving-descent step, i.e. a factor of 2, so a single midpoint at
    sqrt(2)x already narrows the window a lot); run multiple
    --refine-rounds to keep bisecting around the updated best instead of
    spending two probes per gap in a single round."""
    info = {"best_lr": None, "lower_neighbor": None, "upper_neighbor": None, "edge_warning": None}
    converging = [r for r in results if r["status"] == "CONVERGING"]
    if not converging:
        return [], info

    by_lr = sorted(results, key=lambda r: r["lr"])
    lrs_sorted = [r["lr"] for r in by_lr]
    best = min(converging, key=lambda r: r["min_bpb"])
    info["best_lr"] = best["lr"]
    idx = lrs_sorted.index(best["lr"])

    new_lrs = []
    if idx + 1 < len(lrs_sorted):
        upper = lrs_sorted[idx + 1]
        info["upper_neighbor"] = upper
        new_lrs += log_space_midpoints(best["lr"], upper, n_midpoints)
    else:
        info["edge_warning"] = (
            f"best lr={best['lr']:.3g} is the HIGHEST candidate tested for this size -- "
            f"the true optimum may be even higher; the chaining assumption (next size's "
            f"ceiling = this size's best) may be too aggressive."
        )

    if idx - 1 >= 0:
        lower = lrs_sorted[idx - 1]
        info["lower_neighbor"] = lower
        new_lrs += log_space_midpoints(lower, best["lr"], n_midpoints)
    else:
        msg = (f"best lr={best['lr']:.3g} is the LOWEST candidate tested -- "
               f"consider lowering --min-lr-floor to check further down.")
        info["edge_warning"] = (info["edge_warning"] + " " + msg) if info["edge_warning"] else msg

    existing = {r["lr"] for r in results}
    seen = set()
    deduped = []
    for lr in new_lrs:
        if lr in existing or lr in seen:
            continue
        seen.add(lr)
        deduped.append(lr)
    return deduped, info


def extend_upper_edge(size: str, n_gpus: int, results: list[dict], probe_steps: int, clip: float,
                       sources: str, sweep_root: str, extra_args: list[str], force_rerun: bool,
                       max_extensions: int = MAX_UPWARD_EXTENSIONS) -> list[dict]:
    """If the current best candidate sits at the TOP of the tested range
    (no upper neighbor -- i.e. compute_refinement_candidates would raise
    the 'HIGHEST candidate tested' edge_warning), we don't yet know
    whether the halving descent simply started too low. So before
    bracketing, keep doubling the best LR and probing it: as long as the
    doubled point keeps winning, double again. The first time a doubled
    point comes in worse (or fails/diverges), it becomes the best's real
    upper neighbor and we stop -- normal bracket refinement can now run
    with a proper gap on both sides.

    e.g. best=4e-4 with nothing tested above it -> probe 8e-4. If 8e-4
    wins, probe 1.6e-3. If that loses, we now bracket between 4e-4/8e-4
    and 8e-4/1.6e-3 (4e-4 being 8e-4's lower neighbor) as usual.

    Capped at `max_extensions` doublings as a safety valve in case LR
    keeps improving indefinitely (shouldn't happen in practice, but we
    don't want an unbounded loop of real training runs)."""
    for _ in range(max_extensions):
        _, info = compute_refinement_candidates(results)
        if info["best_lr"] is None or info["upper_neighbor"] is not None:
            # Nothing converged yet, or the best already has something
            # tested above it -- normal bracketing can proceed.
            break
        candidate_lr = round_lr(info["best_lr"] * 2)
        print(f"\n--- best lr={info['best_lr']:.3g} is the top of the tested range for "
              f"size={size} -- probing lr={candidate_lr:.3g} before bracketing ---")
        results.append(run_probe(size, n_gpus, candidate_lr, probe_steps, clip, sources,
                                  sweep_root, extra_args, force_rerun))
    else:
        print(f"  WARNING: hit --max-upward-extensions ({max_extensions}) while still "
              f"climbing for size={size}; the LR search may not have found a true peak. "
              f"Consider raising the cap or double-checking this size manually.")
    return results


def refine(size: str, n_gpus: int, results: list[dict], rounds: int, probe_steps: int,
           clip: float, sources: str, sweep_root: str, extra_args: list[str], force_rerun: bool,
           max_upward_extensions: int = MAX_UPWARD_EXTENSIONS) -> list[dict]:
    # Before bracketing gaps, make sure the best point isn't sitting at the
    # top of the tested range with nothing above it to bracket against.
    results = extend_upper_edge(size, n_gpus, results, probe_steps, clip, sources,
                                 sweep_root, extra_args, force_rerun, max_upward_extensions)

    for round_num in range(1, rounds + 1):
        new_lrs, info = compute_refinement_candidates(results)
        if info["edge_warning"]:
            print(f"  [refine round {round_num}] NOTE: {info['edge_warning']}")
        if not new_lrs:
            print(f"  [refine round {round_num}] nothing left to refine, stopping.")
            break
        print(f"\n--- refine round {round_num} for size={size}: "
              f"best so far lr={info['best_lr']:.3g}, probing {sorted(new_lrs, reverse=True)} ---")
        for lr in new_lrs:
            results.append(run_probe(size, n_gpus, lr, probe_steps, clip, sources, sweep_root,
                                      extra_args, force_rerun))
    return results


def find_adjacent_tie(results: list[dict], decimals: int) -> tuple[dict, dict] | None:
    """Looks at the best CONVERGING candidate and its immediate LR-neighbors
    (same notion of 'neighbor' as compute_refinement_candidates uses). If
    either neighbor's min_bpb agrees with the best's to `decimals` digits,
    returns (best, tied_neighbor) -- since which one is "best" in that case
    is just an artifact of Python's min() picking whichever was appended
    first, not evidence that LR is actually better. Returns None if there's
    no tie to resolve."""
    converging = [r for r in results if r["status"] == "CONVERGING"]
    if not converging:
        return None

    by_lr = sorted(results, key=lambda r: r["lr"])
    lrs_sorted = [r["lr"] for r in by_lr]
    best = min(converging, key=lambda r: r["min_bpb"])
    idx = lrs_sorted.index(best["lr"])
    best_rounded = round(best["min_bpb"], decimals)

    for neighbor_idx in (idx - 1, idx + 1):
        if 0 <= neighbor_idx < len(by_lr):
            neighbor = by_lr[neighbor_idx]
            if (neighbor["status"] == "CONVERGING"
                    and round(neighbor["min_bpb"], decimals) == best_rounded):
                return best, neighbor
    return None


def resolve_ties(size: str, n_gpus: int, results: list[dict], probe_steps: int, clip: float,
                  sources: str, sweep_root: str, extra_args: list[str], force_rerun: bool,
                  decimals: int = TIE_BREAK_DECIMALS,
                  max_rounds: int = MAX_TIE_BREAK_ROUNDS) -> list[dict]:
    """After refinement, if the best candidate is tied (to `decimals`
    digits) with an LR-adjacent neighbor, probe their geometric midpoint --
    a tie between two grid points is a sign the true optimum may lie
    between them, not a settled result. Runs regardless of --refine-rounds,
    since this is about data quality (an unresolved tie), not about how
    many bisection rounds were budgeted. Repeats up to `max_rounds` times,
    since resolving one tie can reveal a new one against the freshly probed
    midpoint; stops as soon as no tie remains, or if the midpoint of a tied
    pair has already been tested (nothing left to narrow)."""
    for _ in range(max_rounds):
        tie = find_adjacent_tie(results, decimals)
        if tie is None:
            break
        best, neighbor = tie
        low, high = sorted([best["lr"], neighbor["lr"]])
        midpoint = round_lr(math.sqrt(low * high))

        existing = {r["lr"] for r in results}
        if midpoint in existing:
            print(f"  [tie-break] lr={best['lr']:.3g} and lr={neighbor['lr']:.3g} are tied "
                  f"(both min_bpb={best['min_bpb']:.4f} to {decimals} decimals), but their "
                  f"geometric midpoint {midpoint:.3g} was already tested -- stopping "
                  f"tie-break for this size.")
            break

        print(f"\n--- tie-break for size={size}: lr={best['lr']:.3g} and lr={neighbor['lr']:.3g} "
              f"both landed at min_bpb={best['min_bpb']:.4f} -- probing geometric midpoint "
              f"{midpoint:.3g} ---")
        results.append(run_probe(size, n_gpus, midpoint, probe_steps, clip, sources, sweep_root,
                                  extra_args, force_rerun))
    else:
        print(f"  WARNING: hit --max-tie-break-rounds ({max_rounds}) while still finding ties "
              f"for size={size}; consider inspecting manually.")
    return results


def print_size_report(size: str, results: list[dict]) -> dict | None:
    print(f"\n{'=' * 78}\nsummary for size={size}\n{'=' * 78}")
    print(f"{'lr':>10}  {'status':<11} {'min_bpb':>9} {'@step':>7}  {'final_bpb':>10}")
    for r in sorted(results, key=lambda r: r["lr"], reverse=True):
        if r["status"] == "FAILED":
            print(f"{r['lr']:>10.2e}  {'FAILED':<11} {r.get('reason', ''):<20}")
        else:
            print(f"{r['lr']:>10.2e}  {r['status']:<11} {r['min_bpb']:>9.4f} "
                  f"{r['min_bpb_step']:>7} {r['final_bpb']:>10.4f}")
    converging = [r for r in results if r["status"] == "CONVERGING"]
    if not converging:
        print("No candidate converged for this size -- nothing to hand off to the next size.")
        return None
    best = min(converging, key=lambda r: r["min_bpb"])
    print(f"-> best: lr={best['lr']:.3g}  (min_bpb={best['min_bpb']:.4f})")
    return best


def tune_size(size: str, n_gpus: int, start_ub: float, probe_steps: int, clip: float,
              sources: str, sweep_root: str, extra_args: list[str], min_lr_floor: float,
              refine_rounds: int, force_rerun: bool, patience: int,
              max_upward_extensions: int = MAX_UPWARD_EXTENSIONS,
              tie_break_decimals: int = TIE_BREAK_DECIMALS,
              max_tie_break_rounds: int = MAX_TIE_BREAK_ROUNDS) -> dict | None:
    results, _ = halving_descent(size, n_gpus, start_ub, probe_steps, clip, sources,
                                  sweep_root, extra_args, min_lr_floor, force_rerun, patience)
    results = refine(size, n_gpus, results, refine_rounds, probe_steps, clip, sources,
                      sweep_root, extra_args, force_rerun, max_upward_extensions)
    results = resolve_ties(size, n_gpus, results, probe_steps, clip, sources, sweep_root,
                            extra_args, force_rerun, tie_break_decimals, max_tie_break_rounds)
    return print_size_report(size, results)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=str, required=True,
                   help="Comma-separated sizes, SMALLEST FIRST, e.g. '10M,50M,100M'. "
                        "Each must be a key in launch_training.SIZE_NAME_MAP.")
    p.add_argument("--n-gpus", type=int, required=True)
    p.add_argument("--sources", choices=["balanced", "imbalanced"], required=True,
                   help="Passed straight through to launch_training.py, which requires "
                        "this with no default. Applies to every probe for every size in "
                        "this invocation.")
    p.add_argument("--start-lr", type=float, default=DEFAULT_START_LR,
                   help=f"Upper-bound LR for the SMALLEST size, used only if --prev-size "
                        f"isn't given or isn't found in --tuned-lrs-file (default "
                        f"{DEFAULT_START_LR:.0e}, BLT's reported value). Also doubles as the "
                        f"grid unit for chaining: every subsequent size starts its descent "
                        f"from the next multiple of --start-lr above the previous size's "
                        f"tuned best (e.g. with the default grid, a best of 3e-4 chains to "
                        f"4e-4, and a best of 5.5e-4 chains to 8e-4).")
    p.add_argument("--prev-size", type=str, default=None,
                   help="Name of the size one step smaller than the first entry in --sizes "
                        "(e.g. if --sizes=100M and 50M was already tuned in a previous "
                        "invocation, pass --prev-size 50M). If found in --tuned-lrs-file, "
                        "the next multiple of --start-lr above its tuned lr is used as the "
                        "starting upper bound instead of --start-lr itself. Lets you tune "
                        "sizes across separate runs.")
    p.add_argument("--probe-steps", type=int, default=300,
                   help="Fallback probe length for any size not covered by "
                        "--probe-steps-per-size.")
    p.add_argument("--probe-steps-per-size", type=str, default="10M=300,50M=400,100M=500",
                   help="Comma-separated 'size=steps' overrides (default "
                        "'10M=300,50M=400,100M=500'). Bigger/deeper sizes get a few more "
                        "steps since instability can surface later in training, and since "
                        "at >=500 steps the 10%% warmup fraction is no longer floor-bound "
                        "at 50 steps. A size not listed here falls back to --probe-steps.")
    p.add_argument("--clip", type=float, default=10.0)
    p.add_argument("--min-lr-floor", type=float, default=DEFAULT_MIN_LR_FLOOR)
    p.add_argument("--refine-rounds", type=int, default=2,
                   help="Bracket-refinement rounds after the halving descent stops. Each "
                        "round bisects the gap on either side of the current best (1 new "
                        "point per gap); with 2 rounds (default) that's two successive "
                        "sqrt(2)x-ish narrowings re-centered on the updated best each time.")
    p.add_argument("--patience", type=int, default=1,
                   help="Consecutive worse-than-best steps tolerated during the halving "
                        "descent before actually stopping (default 1: tolerate a single "
                        "blip, then stop if the step after that is also worse).")
    p.add_argument("--max-upward-extensions", type=int, default=MAX_UPWARD_EXTENSIONS,
                   help="Safety cap on how many times refine() will double the best LR "
                        "when it sits at the top of the tested range (default "
                        f"{MAX_UPWARD_EXTENSIONS}).")
    p.add_argument("--tie-break-decimals", type=int, default=TIE_BREAK_DECIMALS,
                   help="Number of decimal digits of min_bpb agreement required to treat "
                        f"two LR-adjacent candidates as tied (default {TIE_BREAK_DECIMALS}, "
                        "matching the precision shown in the summary table). When the best "
                        "candidate is tied with a neighbor, their geometric midpoint is "
                        "probed automatically, irrespective of --refine-rounds.")
    p.add_argument("--max-tie-break-rounds", type=int, default=MAX_TIE_BREAK_ROUNDS,
                   help="Safety cap on how many tie-break midpoints will be chased in a row "
                        f"for a single size (default {MAX_TIE_BREAK_ROUNDS}).")
    p.add_argument("--sweep-root", default=DEFAULT_SWEEP_ROOT)
    p.add_argument("--save-best", action="store_true",
                   help="Write each size's tuned best LR to --tuned-lrs-file as it's found.")
    p.add_argument("--tuned-lrs-file", default="training_setup/learning_rate/tuned_lrs.json",
                   help="Must match launch_training.py's own --tuned-lrs-file default "
                        "so values saved here are picked up automatically later.")
    p.add_argument("--force-rerun", action="store_true",
                   help="Relaunch every candidate even if metrics.jsonl already exists for it.")
    args, extra = p.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    os.makedirs(LR_SWEEP_LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LR_SWEEP_LOGS_DIR, "chain_summary.log")
    sys.stdout = Tee(sys.__stdout__, open(log_path, "w"))
    print(f"Logging this run to: {log_path}")

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    if len(sizes) < 1:
        print("No sizes given via --sizes.")
        sys.exit(1)

    tuned_lookup = load_tuned_lrs(args.tuned_lrs_file)
    probe_steps_map = parse_step_map(args.probe_steps_per_size)

    if args.prev_size:
        if args.prev_size in tuned_lookup:
            chain_upper_bound = next_grid_multiple(tuned_lookup[args.prev_size], args.start_lr)
            print(f"Seeding start bound from {args.tuned_lrs_file}: {args.prev_size}'s tuned "
                  f"lr={tuned_lookup[args.prev_size]:.3g} -> next multiple of "
                  f"start_lr={args.start_lr:.3g} -> {chain_upper_bound:.3g}")
        else:
            print(f"WARNING: --prev-size={args.prev_size} not found in {args.tuned_lrs_file} "
                  f"-- falling back to --start-lr={args.start_lr:.3g}")
            chain_upper_bound = args.start_lr
    else:
        chain_upper_bound = args.start_lr

    final_results = {}
    for i, size in enumerate(sizes):
        probe_steps_this_size = probe_steps_map.get(size, args.probe_steps)
        if i == 0:
            print(f"\n### size={size}: starting from lr={chain_upper_bound:.3g}, "
                  f"probe_steps={probe_steps_this_size} ###")
        else:
            print(f"\n### size={size}: starting from next multiple of "
                  f"start_lr={args.start_lr:.3g} above previous size's tuned best "
                  f"= {chain_upper_bound:.3g}, probe_steps={probe_steps_this_size} ###")

        best = tune_size(size, args.n_gpus, chain_upper_bound, probe_steps_this_size, args.clip,
                          args.sources, args.sweep_root, extra, args.min_lr_floor, args.refine_rounds,
                          args.force_rerun, args.patience, args.max_upward_extensions,
                          args.tie_break_decimals, args.max_tie_break_rounds)
        final_results[size] = best

        if best is None:
            if size in tuned_lookup:
                chain_upper_bound = next_grid_multiple(tuned_lookup[size], args.start_lr)
                print(f"\nWARNING: no candidate converged for size={size} this run, but a "
                      f"previously tuned lr={tuned_lookup[size]:.3g} exists in "
                      f"{args.tuned_lrs_file} -- using the next multiple of start_lr above "
                      f"that for the next size instead of a blind fallback.")
            else:
                chain_upper_bound = round_lr(chain_upper_bound / 2)
                print(f"\nWARNING: no candidate converged for size={size}, and no previously "
                      f"tuned value exists for it either -- halving the current upper bound "
                      f"as a fallback starting point instead.")
            continue

        if args.save_best:
            launch_training.save_tuned_lr(args.tuned_lrs_file, size, best["lr"])
            print(f"Saved lr={best['lr']:.3g} for size={size} to {args.tuned_lrs_file}")

        chain_upper_bound = next_grid_multiple(best["lr"], args.start_lr)

    print(f"\n{'=' * 78}\nfinal chained results\n{'=' * 78}")
    for size in sizes:
        best = final_results[size]
        if best is None:
            print(f"{size:>8}: no converging candidate found")
        else:
            print(f"{size:>8}: lr={best['lr']:.3g}  (min_bpb={best['min_bpb']:.4f})")


if __name__ == "__main__":
    main()