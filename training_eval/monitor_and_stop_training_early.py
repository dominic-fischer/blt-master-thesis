"""
monitor_and_stop_training_early.py

Paired with eval_after_training.sh:
  monitor_and_stop_training_early.py -> run in a separate tmux pane WHILE
                              training is actively running; polls for new
                              checkpoints one at a time as they appear,
                              evaluates held-out bpb on each, and stops
                              training once bpb has stopped improving
                              (early stopping).
  eval_after_training.sh   -> run ONCE, after training has finished;
                              processes every existing checkpoint in a
                              single pass. No early-stopping logic.

DESIGN: this deliberately does NOT touch bytelatent.train. It's an
external monitor, same philosophy as batch_size_sweep.py / lr_halving_sweep.py:
training keeps checkpointing on its normal schedule, and this script
decides from the outside whether to keep letting it run.

EARLY STOPPING RULE: patience + min_delta on HELD-OUT bpb (via
eval_entropy_bpb.py's OVERALL, an equal-weight macro-average across
languages), not training bpb -- training bpb can keep slowly improving
from memorization long after held-out performance has plateaued, so it's
not a safe stopping signal on its own.
  - burn-in-evals: the first N eval events are never counted toward
    patience (early bpb is noisy while the LR is still ramping/near peak).
  - min-delta: an improvement must beat the running best by more than
    this to reset the patience counter -- otherwise noise alone could
    reset patience indefinitely on a genuinely flat curve.
  - patience: number of EVAL EVENTS (not steps) tolerated with no
    real improvement before stopping. Units are eval events because
    that's the only cadence this script can observe -- how many wall-
    clock steps that corresponds to is entirely a function of
    checkpoint.eval.every (see launch_training.py's checkpoint_every).

GPU USAGE: never touches the --n-gpus already running training. Reuses
launch_training.get_free_gpu_ids() to auto-detect ONE currently-idle GPU
for each eval pass (the training GPUs will naturally fail its
memory/utilization thresholds and get excluded automatically -- no
special-casing needed). If no GPU is free at a given poll, logs a warning
and retries at the next poll rather than blocking or giving up on that
checkpoint.

STOPPING TRAINING: sends Ctrl-C to the tmux pane running training
(--tmux-target, default 'training') rather than killing by PID -- keeps
this script decoupled from how exactly training was launched, and lets
torchrun/Python handle SIGINT somewhat gracefully rather than a hard
SIGKILL. Pass --no-stop to only monitor/log without ever touching the
training pane.

RESUMABILITY: progress (best bpb/step, patience streak, which checkpoints
have already been evaluated) is persisted to --state-file after every
checkpoint, so if this monitor itself crashes or is restarted, it picks
up where it left off instead of re-evaluating or losing the patience
count.

ASSUMPTIONS WORTH CHECKING:
  - Checkpoint dirs are named as zero-padded step numbers directly under
    <dump_dir>/checkpoints/ (same layout eval_after_training.sh assumes).
  - launch_training.py is importable via the same sys.path convention
    batch_size_sweep.py / lr_halving_sweep.py use (three directories above
    this file) -- adjust the sys.path.append below if this script doesn't
    actually live at that same depth.

Usage:
    python monitor_and_stop_training_early.py <dump_dir> <lang_shards_root> \\
        --patience 3 --min-delta 0.002 --burn-in-evals 3

    # Calibration run: log what WOULD trigger, never actually stop training:
    python monitor_and_stop_training_early.py <dump_dir> <lang_shards_root> \\
        --patience 3 --min-delta 0.002 --burn-in-evals 3 --simulate
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from os import path

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))  # for launch_training import
from launch_training import (
    get_free_gpu_ids,
    DEFAULT_FREE_MEM_THRESHOLD_MIB,
    DEFAULT_FREE_UTIL_THRESHOLD_PCT,
)

EVAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_entropy_bpb.py")
DEFAULT_POLL_INTERVAL_SEC = 60
DEFAULT_PATIENCE = 3
DEFAULT_MIN_DELTA = 0.002
DEFAULT_BURN_IN_EVALS = 3
MAX_CONSOLIDATE_FAILURES_PER_CHECKPOINT = 1
MAX_EVAL_FAILURES_PER_CHECKPOINT = 3


class Tee:
    """Duplicates writes to real stdout and a log file (same helper as
    lr_halving_sweep.py's Tee)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def list_checkpoint_steps(checkpoints_dir: str) -> list[tuple[str, int]]:
    """Returns [(dirname, step_int), ...] sorted ascending by step, for
    every zero-padded-integer-named directory directly under
    checkpoints_dir. Non-integer-named entries are silently skipped
    (defensive -- e.g. a stray temp dir)."""
    if not os.path.isdir(checkpoints_dir):
        return []
    out = []
    for name in os.listdir(checkpoints_dir):
        full = os.path.join(checkpoints_dir, name)
        if not os.path.isdir(full):
            continue
        try:
            step = int(name)
        except ValueError:
            continue
        out.append((name, step))
    return sorted(out, key=lambda t: t[1])


def load_state(state_file: str) -> dict:
    default = {
        "best_bpb": None,
        "best_step": None,
        "bad_evals": 0,
        "eval_count": 0,
        "processed_steps": [],
        "fail_counts": {},
        "stopped": False,
        "trigger_events": [],
    }
    if not os.path.exists(state_file):
        return default
    try:
        with open(state_file) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return default
    default.update(data)
    return default


def save_state(state_file: str, state: dict) -> None:
    os.makedirs(os.path.dirname(state_file) or ".", exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def consolidate_checkpoint(ckpt_dir: str) -> bool:
    consolidated_dir = os.path.join(ckpt_dir, "consolidated")
    if os.path.isdir(consolidated_dir):
        print("  Already consolidated, skipping.")
        return True
    print("  Consolidating...")
    proc = subprocess.run(
        ["python", "-m", "bytelatent.checkpoint", "consolidate", ckpt_dir],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        print(f"  Consolidation FAILED (exit {proc.returncode}):\n{tail}")
        return False
    return True


def stop_training(tmux_target: str) -> None:
    print(f"\n>>> Sending Ctrl-C to tmux target '{tmux_target}' to stop training <<<")
    proc = subprocess.run(
        ["tmux", "send-keys", "-t", tmux_target, "C-c"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"  WARNING: tmux send-keys failed (exit {proc.returncode}): "
              f"{proc.stderr.strip()}\n  You'll need to stop training manually.")
    else:
        print(f"  Sent. Training should stop shortly -- check the '{tmux_target}' pane.")


def run_eval(consolidated_dir: str, lang_shards_root: str, target_bytes_per_lang: int,
             metrics_jsonl: str, json_out_path: str, gpu_id: int) -> dict | None:
    """Runs eval_entropy_bpb.py pinned to gpu_id via CUDA_VISIBLE_DEVICES,
    prints its full output for visibility (so this monitor's own log is a
    complete record, same spirit as eval_after_training.sh's tee), and
    returns the parsed --json-out summary, or None on failure."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable, EVAL_SCRIPT,
        consolidated_dir, lang_shards_root,
        "--target-bytes-per-lang", str(target_bytes_per_lang),
        "--device", "cuda",
        "--metrics-jsonl", metrics_jsonl,
        "--json-out", json_out_path,
    ]
    print(f"  Evaluating held-out bpb on GPU {gpu_id}...")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"  eval_entropy_bpb.py FAILED (exit {proc.returncode}):\n{proc.stderr}")
        return None
    try:
        with open(json_out_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Could not read --json-out result at {json_out_path}: {e}")
        return None


def process_checkpoint(ckpt_name: str, step: int, dump_dir: str, lang_shards_root: str,
                        steps_per_epoch: float, target_bytes_per_lang: int,
                        mem_threshold: int, util_threshold: int, state: dict) -> str:
    """Returns one of: 'done' (evaluated, state updated), 'retry_gpu' (no
    idle GPU right now, try again next poll), 'gave_up' (too many
    consolidate/eval failures for this checkpoint, marked processed anyway
    so the monitor doesn't loop on it forever)."""
    ckpt_dir = os.path.join(dump_dir, "checkpoints", ckpt_name)
    if steps_per_epoch:
        epoch = round(step / steps_per_epoch, 2)
        print(f"\n=== step {step} (epoch ~{epoch}) ===")
    else:
        print(f"\n=== step {step} ===")

    fail_key = str(step)
    fail_counts = state["fail_counts"]

    if not consolidate_checkpoint(ckpt_dir):
        fail_counts[fail_key] = fail_counts.get(fail_key, 0) + 1
        if fail_counts[fail_key] > MAX_CONSOLIDATE_FAILURES_PER_CHECKPOINT:
            print(f"  Giving up on step {step} after repeated consolidate failures.")
            return "gave_up"
        return "retry_gpu"  # reuse the same "try again next poll" path

    try:
        gpu_ids = get_free_gpu_ids(1, mem_threshold, util_threshold)
    except SystemExit as e:
        print(f"  No idle GPU available right now ({e}); will retry at next poll.")
        return "retry_gpu"
    gpu_id = gpu_ids[0]

    metrics_jsonl = os.path.join(dump_dir, "metrics.jsonl")
    json_out_path = os.path.join(ckpt_dir, "eval_result.json")
    result = run_eval(os.path.join(ckpt_dir, "consolidated"), lang_shards_root,
                       target_bytes_per_lang, metrics_jsonl, json_out_path, gpu_id)

    if result is None:
        fail_counts[fail_key] = fail_counts.get(fail_key, 0) + 1
        if fail_counts[fail_key] > MAX_EVAL_FAILURES_PER_CHECKPOINT:
            print(f"  Giving up on step {step} after repeated eval failures.")
            return "gave_up"
        return "retry_gpu"

    overall_bpb = result["overall_bpb"]
    state["eval_count"] += 1
    eval_num = state["eval_count"]

    if eval_num <= state.get("_burn_in_evals", DEFAULT_BURN_IN_EVALS):
        print(f"  held_out_bpb={overall_bpb:.4f}  (eval {eval_num}, still in burn-in -- "
              f"not counted toward patience)")
    else:
        best_bpb = state["best_bpb"]
        min_delta = state.get("_min_delta", DEFAULT_MIN_DELTA)
        if best_bpb is None or overall_bpb < best_bpb - min_delta:
            state["best_bpb"] = overall_bpb
            state["best_step"] = step
            state["bad_evals"] = 0
            print(f"  held_out_bpb={overall_bpb:.4f}  <-- new best (was "
                  f"{best_bpb:.4f})" if best_bpb is not None else
                  f"  held_out_bpb={overall_bpb:.4f}  <-- first counted eval, new best")
        else:
            state["bad_evals"] += 1
            patience = state.get("_patience", DEFAULT_PATIENCE)
            print(f"  held_out_bpb={overall_bpb:.4f}  (no improvement vs best="
                  f"{best_bpb:.4f} @step {state['best_step']}; "
                  f"bad_evals={state['bad_evals']}/{patience})")
            if state["bad_evals"] > patience:
                simulate = state.get("_simulate", False)
                lead = "WOULD HAVE STOPPED" if simulate else "EARLY STOPPING TRIGGERED"
                print(f"\n  {lead} at step {step} "
                      f"(patience={patience}, min_delta={state.get('_min_delta', DEFAULT_MIN_DELTA)}, "
                      f"burn_in_evals={state.get('_burn_in_evals', DEFAULT_BURN_IN_EVALS)}): "
                      f"{state['bad_evals']} consecutive eval(s) without improving "
                      f"on best_bpb={best_bpb:.4f} @step {state['best_step']}.")
                state.setdefault("trigger_events", []).append({
                    "step": step, "bad_evals": state["bad_evals"],
                    "best_bpb": best_bpb, "best_step": state["best_step"],
                    "patience": patience, "min_delta": state.get("_min_delta", DEFAULT_MIN_DELTA),
                    "burn_in_evals": state.get("_burn_in_evals", DEFAULT_BURN_IN_EVALS),
                })
                if not simulate:
                    state["stopped"] = True
                else:
                    print("  --simulate set: logging only, NOT stopping training, "
                          "continuing to monitor.")

    state["processed_steps"].append(step)
    return "done"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump_dir")
    p.add_argument("lang_shards_root")
    p.add_argument("--steps-per-epoch", type=float, default=None,
                   help="Optional, from launch_training.py's printed 'X steps/epoch' "
                        "line -- purely cosmetic (only used for the displayed epoch "
                        "number in this monitor's log). Omit if you're not thinking "
                        "in epochs (e.g. training on a corpus you don't expect to "
                        "finish one epoch of) -- the step number is always shown "
                        "regardless.")
    p.add_argument("--target-bytes-per-lang", type=int, default=150_000)
    p.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL_SEC,
                   help=f"Seconds between checks for new checkpoints (default "
                        f"{DEFAULT_POLL_INTERVAL_SEC}).")
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                   help=f"Consecutive eval EVENTS (not steps) tolerated with no real "
                        f"improvement before stopping training (default {DEFAULT_PATIENCE}).")
    p.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA,
                   help=f"Minimum bpb improvement over the running best to reset the "
                        f"patience counter (default {DEFAULT_MIN_DELTA}); without this, "
                        f"noise alone could reset patience indefinitely on a flat curve.")
    p.add_argument("--burn-in-evals", type=int, default=DEFAULT_BURN_IN_EVALS,
                   help=f"First N eval events are never counted toward patience -- bpb "
                        f"is typically noisy while LR is still near/at peak (default "
                        f"{DEFAULT_BURN_IN_EVALS}).")
    p.add_argument("--free-mem-threshold-mib", type=int, default=DEFAULT_FREE_MEM_THRESHOLD_MIB)
    p.add_argument("--free-util-threshold-pct", type=int, default=DEFAULT_FREE_UTIL_THRESHOLD_PCT)
    p.add_argument("--tmux-target", default="training",
                   help="tmux target (session/window/pane) to send Ctrl-C to on early "
                        "stop (default 'training'). See `tmux send-keys -t <target> C-c`.")
    p.add_argument("--no-stop", action="store_true",
                   help="Never send Ctrl-C to the training pane, even if early "
                        "stopping triggers for real -- the monitor still exits once "
                        "triggered (patience is 'used up'), it just leaves stopping "
                        "training to you. For calibration runs where you don't want "
                        "the monitor to stop reacting at all, use --simulate instead.")
    p.add_argument("--simulate", action="store_true",
                   help="Calibration mode: never sends Ctrl-C AND never exits on "
                        "trigger -- logs 'WOULD HAVE STOPPED' with the parameters "
                        "that triggered it, records it in --state-file's "
                        "trigger_events, and keeps monitoring/logging for the rest "
                        "of the run. Lets you see the full bpb trajectory, including "
                        "what happens after a hypothetical stop point (recovers? "
                        "stays flat? gets worse?), instead of cutting the run short "
                        "the first time patience would have been exceeded.")
    p.add_argument("--state-file", default=None,
                   help="Where to persist best-bpb/patience/processed-checkpoints state "
                        "for crash-resumability. Default: <dump_dir>/early_stopping_state.json")
    args = p.parse_args()

    state_file = args.state_file or os.path.join(args.dump_dir, "early_stopping_state.json")
    state = load_state(state_file)
    # Stash current CLI thresholds into state so process_checkpoint can see them
    # without threading three more params through every call.
    state["_patience"] = args.patience
    state["_min_delta"] = args.min_delta
    state["_burn_in_evals"] = args.burn_in_evals
    state["_simulate"] = args.simulate

    os.makedirs(args.dump_dir, exist_ok=True)
    log_path = os.path.join(args.dump_dir, f"monitor_and_stop_training_early_{time.strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = Tee(sys.__stdout__, open(log_path, "w"))
    print(f"Logging this monitor run to: {log_path}")
    print(f"State file: {state_file}")
    if state["processed_steps"]:
        print(f"Resuming: {len(state['processed_steps'])} checkpoint(s) already evaluated, "
              f"best_bpb={state['best_bpb']} @step {state['best_step']}, "
              f"bad_evals={state['bad_evals']}")

    if state["stopped"]:
        print("State file shows early stopping already triggered in a previous run of "
              "this monitor. Delete the state file (or pass a different --state-file) "
              "to start fresh. Exiting.")
        return

    checkpoints_dir = os.path.join(args.dump_dir, "checkpoints")
    processed = set(state["processed_steps"])

    try:
        while True:
            all_ckpts = list_checkpoint_steps(checkpoints_dir)
            pending = [(name, step) for name, step in all_ckpts if step not in processed]

            if not pending:
                time.sleep(args.poll_interval)
                continue

            for ckpt_name, step in pending:
                outcome = process_checkpoint(
                    ckpt_name, step, args.dump_dir, args.lang_shards_root,
                    args.steps_per_epoch, args.target_bytes_per_lang,
                    args.free_mem_threshold_mib, args.free_util_threshold_pct, state,
                )
                if outcome in ("done", "gave_up"):
                    processed.add(step)
                    save_state(state_file, state)
                if outcome == "retry_gpu":
                    # Stop working through this poll's pending list; retry
                    # everything still pending (in order) on the next pass.
                    break
                if state["stopped"]:
                    break

            if state["stopped"]:
                if not args.no_stop:
                    stop_training(args.tmux_target)
                else:
                    print("  --no-stop set: NOT sending Ctrl-C. Stop training manually "
                          "if desired.")
                save_state(state_file, state)
                print("\nMonitor exiting after early-stop trigger.")
                return

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nMonitor interrupted by user, saving state and exiting.")
        save_state(state_file, state)


if __name__ == "__main__":
    main()