"""
eval_after_training.py

Paired with monitor_and_stop_training_early.py:
  monitor_and_stop_training_early.py -> run in a separate tmux pane WHILE
                              training is actively running; polls for new
                              checkpoints one at a time as they appear.
  eval_after_training.py   -> run ONCE, after training has finished;
                              processes every existing checkpoint in a
                              single pass. Use this to recover a full
                              progression if the live monitor crashed
                              partway through, or to inspect a completed
                              run after the fact.

Retroactively consolidates and evaluates every saved checkpoint, in step
order.

OUTPUT LOCATION: saved (not just printed) to
  <dump_dir>/eval_after_training_<timestamp>.log
alongside the run's own checkpoints/metrics.jsonl, since this is a
permanent artifact about the trained model itself -- not a transient
process log -- and belongs next to the other per-run outputs rather than
under logs/ (which holds live torchrun stdout/stderr per attempt).
Timestamped so re-running eval later (e.g. after fixing a bug in
eval_entropy_bpb.py) doesn't clobber a previous pass.

EPOCH DISPLAY IS OPTIONAL (--steps-per-epoch) -- purely cosmetic, same
convention as monitor_and_stop_training_early.py's --steps-per-epoch, for
anyone still thinking in epochs. Omit it if you're not (e.g. training on
a corpus you don't expect to finish one epoch of); step numbers are
always shown regardless.

FAIL-FAST: a bad checkpoint STOPS this script immediately (unlike the
live monitor, which retries/skips past failures) -- this is a one-shot
retrospective pass over an already-finished run, not a long-lived
watcher, so a failure partway through should be surfaced right away
rather than silently continuing past it and producing a misleadingly
partial report.

SUBPROCESS OUTPUT: each consolidate/eval subprocess's own stdout/stderr
is captured explicitly and printed through this script's own logger,
rather than letting it write directly to the terminal -- reassigning
sys.stdout (as the Tee below does) only affects this process's own
print() calls, not what a child process writes to its inherited file
descriptors, so without this the consolidate/eval output would appear on
screen but be silently missing from the saved log file.

Usage:
    python eval_after_training.py <dump_dir> <lang_shards_root> [--steps-per-epoch 15462.6]
"""
import argparse
import os
import subprocess
import sys
import time


class Tee:
    """Duplicates writes to real stdout and a log file (same helper used
    elsewhere in this pipeline)."""

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


def run_step(cmd: list[str], step_label: str) -> None:
    """Runs cmd, prints its full stdout+stderr through this script's own
    (Tee'd) print so it lands in both the terminal and the saved log
    file, and exits immediately on failure -- matching the original
    bash script's `set -e` fail-fast behavior, but with a clear message
    instead of a bare non-zero exit."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    if proc.returncode != 0:
        print(f"\n{step_label} FAILED (exit {proc.returncode}) -- stopping. "
              f"A bad checkpoint halts this one-shot pass rather than silently "
              f"continuing past it and producing a partial report.")
        sys.exit(proc.returncode)


def consolidate_checkpoint(ckpt_dir: str) -> None:
    consolidated_dir = os.path.join(ckpt_dir, "consolidated")
    if os.path.isdir(consolidated_dir):
        print("Already consolidated, skipping.")
        return
    print("Consolidating...")
    run_step(["python", "-m", "bytelatent.checkpoint", "consolidate", ckpt_dir],
              "Consolidation")


def eval_checkpoint(ckpt_dir: str, lang_shards_root: str) -> None:
    print("Evaluating held-out bpb...")
    run_step(
        ["python", "training_eval/eval_entropy_bpb.py", os.path.join(ckpt_dir, "consolidated"), lang_shards_root],
        "Evaluation",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump_dir")
    parser.add_argument("lang_shards_root")
    parser.add_argument("--steps-per-epoch", type=float, default=None,
                         help="Optional, from launch_training.py's printed 'X steps/epoch' "
                              "line -- purely cosmetic (only used for the displayed epoch "
                              "number). Omit if you're not thinking in epochs; step numbers "
                              "are always shown regardless.")
    args = parser.parse_args()

    log_dir = os.path.join("logs", args.dump_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"eval_after_training_{time.strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = Tee(sys.__stdout__, open(log_path, "w"))
    print(f"Logging this eval run to: {log_path}")

    checkpoints_dir = os.path.join("dumps", args.dump_dir, "checkpoints")
    checkpoints = list_checkpoint_steps(checkpoints_dir)
    if not checkpoints:
        print(f"No checkpoints found under {checkpoints_dir}")
        return

    for ckpt_name, step in checkpoints:
        ckpt_dir = os.path.join(checkpoints_dir, ckpt_name)
        if args.steps_per_epoch:
            epoch = round(step / args.steps_per_epoch, 2)
            print(f"\n=== step {step} (epoch ~{epoch}) ===")
        else:
            print(f"\n=== step {step} ===")
        consolidate_checkpoint(ckpt_dir)
        eval_checkpoint(ckpt_dir, args.lang_shards_root)


if __name__ == "__main__":
    main()