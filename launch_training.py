"""
launch_training.py

Given a model size (tiny/small/medium/paper-scale/repo-scale), computes and
prints the full torchrun launch command with the correct CLI overrides on
top of entropy_model_2080ti.yaml -- architecture dims, data.root_dir,
dump_dir, steps (for a target epoch count), checkpoint cadence, wandb run
name. Avoids hand-editing the yaml or manually recomputing epoch math per
size/GPU-count combination.

Shard directory convention (IMPORTANT -- prepare shards to match this before
running): data/lang_shards_<size>_<n_gpus>gpu/
    e.g. data/lang_shards_tiny_4gpu, data/lang_shards_medium_4gpu

Encoding n_gpus in the directory name is deliberate: it prevents accidentally
reusing shards prepared with a different chunk count for a different GPU
count, which silently triggers bytelatent's chunk-discarding behavior when
n_chunks > world_size (see find_and_sanitize_chunks in bytelatent/args.py).

Reads:
    - blt_configs_computed.csv : architecture (dim, n_heads, n_layers) per size
    - langs_chosen.csv         : per-language byte allocation per size, to
                                  compute total corpus size for steps-per-epoch

Logging: torchrun is launched with --log-dir, so each rank's stdout/stderr
goes to its own file under logs/<run_name>/attempt_0/<rank>/stdout.log
instead of interleaving all ranks into one messy terminal stream. This
script prints a `tail -f .../attempt_0/*/stdout.log` command to watch all
ranks side-by-side (each line prefixed with its filename), which also makes
it easy to spot a rank that's gone silent -- exactly the pattern that
preceded the NCCL-timeout crash this script's other changes are guarding
against.

Evaluation: this script no longer prints an eval_during_training.sh line.
Running eval concurrently with training (a second process loading a
checkpoint onto the same GPUs while training is actively using them) is a
plausible cause of silent rank crashes -- extra GPU/host memory pressure on
a shared machine can get a training rank OOM-killed by the kernel with no
Python traceback, which then shows up as the *other* ranks timing out
waiting on it in NCCL. Use eval_after_training.sh once training has
finished instead; this script prints that command for you, both up front
and again after a successful --run.

Reusable as a module: build_parser() and compute_plan() are exposed
specifically so lr_sweep.py can build run plans (dump_dir, cmd, run_env,
etc.) using the exact same logic/defaults as this script's CLI, without
duplicating and risking drift between the two.

Usage:
    python launch_training.py <size> --n-gpus 4 --epochs 10 [--batch-size 16]
                               [--run]

Example:
    python launch_training.py tiny --n-gpus 4 --epochs 10
    python launch_training.py medium --n-gpus 4 --epochs 10 --run
"""

import argparse
import csv
import json
import os
import subprocess
import sys

SIZE_NAME_MAP = {
    "tiny": "Tiny",
    "small": "Small",
    "medium": "Medium",
    "paper-scale": "Paper-scale",
    "repo-scale": "Repo-scale",
}

# Fallback when a size has no entry in the tuned-LR lookup file yet: the
# yaml's own optim.lr, which is Meta's repo-scale (dim=768/12h/14L) debug
# tuning -- NOT validated for other sizes. See --lr's help text.
FALLBACK_LR = 4e-4


def load_tuned_lr(path: str, size: str) -> float | None:
    """Returns the saved LR for this size from the tuned-LR lookup file
    (see lr_sweep.py's --save-best), or None if the file doesn't exist or
    has no entry for this size yet."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return None
    value = data.get(size)
    return float(value) if value is not None else None


def save_tuned_lr(path: str, size: str, lr: float) -> None:
    """Read-modify-write: updates only this size's entry, preserving
    whatever other sizes are already saved in the file."""
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    data[size] = lr
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def load_architecture(configs_csv: str, size_row_name: str) -> dict:
    with open(configs_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["config_name"] == size_row_name:
                return {
                    "dim": int(row["dim"]),
                    "n_heads": int(row["n_heads"]),
                    "n_layers": int(row["n_layers"]),
                    "total_params": int(row["total_params"]),
                }
    raise ValueError(f"'{size_row_name}' not found in {configs_csv}")


def compute_total_corpus_bytes(langs_csv: str, byte_column: str) -> int:
    total = 0
    with open(langs_csv, newline="") as f:
        for row in csv.DictReader(f):
            total += int(row[byte_column])
    return total


def format_value(x: float) -> str:
    """Compact, filename-safe representation of a hyperparameter value.
    Whole numbers print without a decimal point (10 -> '10'); small
    floats print in short scientific notation without a leading zero in
    the exponent (0.0001 -> '1e-4', matching how people naturally write
    LRs, rather than Python's default '1e-04' or '0.0001')."""
    if x == round(x) and abs(x) >= 1:
        return str(int(round(x)))
    s = f"{x:.1e}" if x != 0 else "0"
    mantissa, exp = s.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exp_sign = exp[0]
    exp_num = str(int(exp[1:]))
    return f"{mantissa}e{exp_sign}{exp_num}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("size", choices=list(SIZE_NAME_MAP.keys()))
    parser.add_argument("--n-gpus", type=int, required=True)
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--probe-steps", type=int, default=None,
                         help="For quick LR probes (see lr_sweep.py): use "
                              "exactly this many steps instead of computing "
                              "steps from --epochs, and disable periodic "
                              "checkpointing/eval entirely (set to fire "
                              "past the end of the run) since a probe only "
                              "needs metrics.jsonl's per-step train bpb/ "
                              "grad_norm, logged every logging.freq steps "
                              "regardless of checkpointing. Overrides "
                              "--epochs when set.")
    parser.add_argument("--batch-size", type=int, default=16,
                         help="Default 16 is validated safe for Medium on "
                              "2080 Ti. Smaller models (Tiny/Small) likely "
                              "have more memory headroom and could use a "
                              "larger batch -- not yet empirically tuned, "
                              "override if you smoke-test a higher value.")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--configs-csv", default="training_setup/model_configs_computed.csv")
    parser.add_argument("--langs-csv", default="training_setup/langs/langs_chosen.csv")
    parser.add_argument("--base-config", default="entropy_model_2080ti.yaml")
    parser.add_argument("--shard-root-base", default="data",
                         help="Shards expected at "
                              "<base>/lang_shards_<size>_<n_gpus>gpu/")
    parser.add_argument("--dump-root", default="dumps")
    parser.add_argument("--log-root", default="logs",
                         help="torchrun writes per-rank stdout/stderr under "
                              "<log-root>/<run_name>/attempt_N/<rank>/ via "
                              "--log-dir, instead of interleaving all ranks "
                              "into one terminal stream.")
    parser.add_argument("--eval-after-script", default="eval_after_training.sh",
                         help="Path to the post-training eval script "
                              "(run once, after training finishes).")
    parser.add_argument("--warmup-fraction", type=float, default=0.1,
                         help="warmup = min(base_warmup, round(warmup_fraction * "
                              "total_steps)). Default base_warmup=500 (the yaml's "
                              "own value) means Medium-scale runs are unaffected; "
                              "short runs (e.g. Tiny at a few hundred/thousand "
                              "total steps) get a proportionally shorter warmup "
                              "instead of inheriting a fixed 500-step warmup that "
                              "can eat most or all of the run. Confirmed: a Tiny "
                              "run with warmup=500 over only 789 total steps spent "
                              "~63%% of training still ramping LR upward, and both "
                              "train AND held-out bpb climbed in lockstep the "
                              "whole time -- not overfitting, just an LR schedule "
                              "mismatched to the run length.")
    parser.add_argument("--base-warmup", type=int, default=500,
                         help="Upper cap for the scaled warmup (the yaml's own "
                              "default warmup value).")
    parser.add_argument("--tuned-lrs-file", default="training_setup/tuned_lrs.json",
                         help="JSON lookup of {size: lr} saved by "
                              "lr_sweep.py --save-best. Used as --lr's "
                              "default when --lr isn't explicitly passed, "
                              "so once a size has been swept you don't "
                              "have to remember/retype its result.")
    parser.add_argument("--lr", type=float, default=None,
                         help="optim.lr override. If omitted, uses the "
                              "value saved for this size in "
                              "--tuned-lrs-file if present, else falls "
                              f"back to the yaml's own {FALLBACK_LR} -- "
                              "which is Meta's original repo-scale "
                              "(dim=768/12h/14L) tuning, NOT validated for "
                              "other sizes, since this script's size "
                              "overrides only touch entropy_model.* dims, "
                              "never optim.lr on their own. Confirmed via "
                              "metrics.jsonl on a Tiny (dim=256/4h/8L) run: "
                              "grad_norm and bpb both climbed steadily for "
                              "~200 steps right after warmup ended at peak "
                              f"lr={FALLBACK_LR}, even though lr was already "
                              "decaying -- classic too-high-lr divergence, "
                              "not a warmup-length or overfitting issue. "
                              "Use lr_sweep.py to find and save a validated "
                              "value per size instead of guessing.")
    parser.add_argument("--clip", type=float, default=10.0,
                         help="optim.clip (grad norm clipping threshold). "
                              "Default 10.0 matches the yaml's own value, "
                              "which is loose relative to the grad_norm "
                              "values actually observed on a Tiny run "
                              "(~0.8-1.2 even while diverging) -- so it was "
                              "providing essentially no protection there. "
                              "Consider tightening (e.g. 1.0) as cheap "
                              "insurance, especially alongside model_dtype: "
                              "fp16 in this yaml, which has no GradScaler / "
                              "no automatic loss-scaling safety net.")
    parser.add_argument("--enable-wandb", action="store_true",
                         help="By default WANDB_MODE=disabled is set so wandb "
                              "is a complete no-op (no network calls, no login "
                              "needed) regardless of the yaml's wandb config. "
                              "Pass this flag to actually enable wandb logging.")
    parser.add_argument("--run", action="store_true",
                         help="Actually execute the command instead of just printing it")
    return parser


def compute_plan(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    """Turns parsed args into everything needed to print or execute a run:
    the torchrun command, env vars, dump/log dirs, and the post-training
    eval command. Kept separate from main() so lr_sweep.py can build and
    launch runs programmatically using identical logic/defaults."""
    size_row_name = SIZE_NAME_MAP[args.size]
    byte_column = f"{size_row_name}_bytes"

    # Resolve --lr: explicit CLI value > saved tuned value for this size >
    # yaml fallback. Tracked separately from a simple parser-default
    # comparison since the "default" here depends on args.size, which
    # argparse itself has no notion of.
    tuned_lr = load_tuned_lr(args.tuned_lrs_file, args.size)
    effective_default_lr = tuned_lr if tuned_lr is not None else FALLBACK_LR
    if args.lr is None:
        args.lr = effective_default_lr
        lr_source = (f"tuned default for {args.size}, from {args.tuned_lrs_file}"
                     if tuned_lr is not None else
                     f"yaml fallback -- not yet tuned for {args.size}, "
                     f"see lr_sweep.py")
    else:
        lr_source = "explicit override"

    arch = load_architecture(args.configs_csv, size_row_name)
    total_corpus_bytes = compute_total_corpus_bytes(args.langs_csv, byte_column)

    bytes_per_step = args.n_gpus * args.batch_size * args.seq_len
    steps_per_epoch = total_corpus_bytes / bytes_per_step

    if args.probe_steps is not None:
        # Quick LR-probe mode: fixed step count, no periodic
        # checkpoint/eval (set to fire past the end of the run so they
        # never trigger) -- a probe only needs metrics.jsonl's per-step
        # train bpb/grad_norm, which log every logging.freq steps
        # regardless of checkpointing, so skipping checkpoint I/O and
        # eval passes makes each probe much faster.
        total_steps = args.probe_steps
        checkpoint_every = total_steps + 1
    else:
        checkpoint_every = round(steps_per_epoch)
        # total_steps is forced to an exact multiple of checkpoint_every
        # (for whole-number epoch requests) as defense-in-depth against a
        # real bytelatent bug: train.py's "saved" flag guarding the final
        # checkpoint is never reset per-iteration, so once any periodic
        # checkpoint fires, the "always save when training ends" safety
        # net silently becomes a no-op -- meaning any steps after the
        # LAST periodic checkpoint are never persisted. Confirmed: a Tiny
        # run (steps=2625, checkpoint.dump.every=263) lost its final 258
        # steps (~1 epoch) this way. Making total_steps an exact multiple
        # of checkpoint_every means the last periodic checkpoint IS the
        # final step, so nothing is lost even if train.py hasn't been
        # patched on a given machine.
        total_steps = round(args.epochs) * checkpoint_every

    warmup = max(1, min(args.base_warmup, round(args.warmup_fraction * total_steps)))

    shard_root = os.path.join(
        args.shard_root_base, f"lang_shards_{args.size}_{args.n_gpus}gpu"
    )

    # Append a suffix for every hyperparameter overridden from its default,
    # so two runs with different lr/epochs/etc. for the same size+n_gpus
    # don't silently collide in the same dump_dir/log_dir and overwrite
    # each other's checkpoints/metrics.jsonl. Compared against the actual
    # parser defaults (not hardcoded numbers here), so this stays correct
    # even if a default above is changed later. "lr" is handled separately
    # (below) since its effective default is size-dependent (tuned-LR
    # lookup), not a single fixed parser default.
    tunable = {
        "epochs": "epochs",
        "probe_steps": "probesteps",
        "clip": "clip",
        "batch_size": "bs",
        "seq_len": "seqlen",
    }
    suffix_parts = []
    for arg_name, label in tunable.items():
        value = getattr(args, arg_name)
        default = parser.get_default(arg_name)
        if value != default:
            suffix_parts.append(f"{label}{format_value(value)}")
    # lr is ALWAYS included in the name, unlike the other tunables above --
    # its effective default is size-dependent (tuned-LR lookup) and can
    # change over time as tuned_lrs.json gets updated, so a run using
    # "the default" today could be using a different actual number than a
    # run using "the default" from before a --save-best update. Comparing
    # against effective_default_lr would make otherwise-identical dump_dir
    # names ambiguous about which numeric lr was actually used at the
    # time -- always printing the number avoids that.
    suffix_parts.append(f"lr{format_value(args.lr)}")
    run_name_suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

    run_name = f"entropy_{args.size}_20lang_{args.n_gpus}gpu{run_name_suffix}"
    dump_dir = os.path.join(args.dump_root, run_name)
    # log_dir intentionally mirrors dump_dir exactly (same run_name, only the
    # root differs: dumps/ vs logs/) -- both derive from the single run_name
    # variable so they can't silently drift apart from each other.
    log_dir = os.path.join(args.log_root, run_name)

    shard_warning = None
    if not os.path.isdir(shard_root):
        shard_warning = (
            f"WARNING: shard directory not found: {shard_root}\n"
            f"  Prepare it first, e.g.:\n"
            f"  python prepare_language_shards.py {args.langs_csv} {byte_column} "
            f"{shard_root} --n-chunks {args.n_gpus}"
        )

    overrides = [
        f"steps={total_steps}",
        f"dump_dir={dump_dir}",
        f"name={run_name}",
        f"entropy_model.dim={arch['dim']}",
        f"entropy_model.n_heads={arch['n_heads']}",
        f"entropy_model.n_layers={arch['n_layers']}",
        f"data.root_dir={shard_root}",
        f"data.batch_size={args.batch_size}",
        f"data.seq_len={args.seq_len}",
        f"data.max_encoder_seq_length={args.seq_len}",
        f"checkpoint.dump.every={checkpoint_every}",
        f"checkpoint.eval.every={checkpoint_every}",
        f"optim.warmup={warmup}",
        f"optim.lr={args.lr}",
        f"optim.clip={args.clip}",
        f"logging.wandb.name={run_name}",
    ]

    cmd = (
        ["torchrun", f"--nproc_per_node={args.n_gpus}", "--standalone",
         f"--log-dir={log_dir}", "--redirects=3",
         "-m", "bytelatent.train", f"config={args.base_config}"]
        + overrides
    )

    cuda_visible_devices = ",".join(str(i) for i in range(args.n_gpus))
    env_prefix = [
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "export PYTHONUNBUFFERED=1",
    ]
    if not args.enable_wandb:
        env_prefix.append("export WANDB_MODE=disabled")
    env_prefix.append(f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}")

    run_env = dict(os.environ)
    run_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    run_env["PYTHONUNBUFFERED"] = "1"
    if not args.enable_wandb:
        run_env["WANDB_MODE"] = "disabled"
    run_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    # Post-training eval only -- do NOT run this while training is still
    # active (see module docstring: concurrent eval is a suspected cause
    # of silent rank crashes on shared/contended GPUs). Not meaningful in
    # probe mode (no checkpoints are saved), so callers should skip this
    # when args.probe_steps is set.
    eval_after_cmd = [
        "bash", args.eval_after_script,
        dump_dir,
        shard_root,
        # checkpoint_every (not the raw steps_per_epoch float) -- this is
        # the actual checkpoint cadence during training, so it's what the
        # epoch numbers in eval_after_training.sh should be computed
        # against for consistency with where checkpoints really land.
        str(checkpoint_every),
    ]

    return {
        "arch": arch,
        "size_row_name": size_row_name,
        "total_corpus_bytes": total_corpus_bytes,
        "steps_per_epoch": steps_per_epoch,
        "checkpoint_every": checkpoint_every,
        "total_steps": total_steps,
        "warmup": warmup,
        "lr_source": lr_source,
        "shard_root": shard_root,
        "shard_warning": shard_warning,
        "run_name": run_name,
        "dump_dir": dump_dir,
        "log_dir": log_dir,
        "metrics_jsonl": os.path.join(dump_dir, "metrics.jsonl"),
        "cmd": cmd,
        "env_prefix": env_prefix,
        "run_env": run_env,
        "cuda_visible_devices": cuda_visible_devices,
        "eval_after_cmd": eval_after_cmd,
    }


def print_plan(args: argparse.Namespace, plan: dict) -> None:
    if plan["shard_warning"]:
        print(plan["shard_warning"])
        print()

    print(f"# {plan['size_row_name']}: {plan['arch']['total_params']:,} params, "
          f"dim={plan['arch']['dim']} n_layers={plan['arch']['n_layers']} "
          f"n_heads={plan['arch']['n_heads']}")
    print(f"# corpus: {plan['total_corpus_bytes']:,} bytes, "
          f"{plan['steps_per_epoch']:.1f} steps/epoch @ {args.n_gpus} GPUs, "
          f"batch_size={args.batch_size}, seq_len={args.seq_len}")
    if args.probe_steps is not None:
        print(f"# PROBE MODE: steps={plan['total_steps']} (fixed, --epochs "
              f"ignored), checkpointing/eval disabled")
    else:
        print(f"# {round(args.epochs)} epochs -> steps={plan['total_steps']} "
              f"(exact multiple of checkpoint_every={plan['checkpoint_every']}, "
              f"guards against the missing-final-checkpoint bug)")
    print(f"# warmup={plan['warmup']} (scaled to {args.warmup_fraction:.0%} of "
          f"total_steps, capped at {args.base_warmup})")
    print(f"# optim.lr={args.lr}  optim.clip={args.clip}")
    print(f"#   lr source: {plan['lr_source']}")
    print()
    print("# --- Training ---")
    for line in plan["env_prefix"][:-1]:
        print(line)
    print(plan["env_prefix"][-1] + " \\")
    print("    " + " \\\n    ".join(plan["cmd"]))
    print()
    print("# --- Terminal 2: watch each rank's log separately (avoids the "
          "interleaved single-stream mess) ---")
    print("# torchrun creates this the moment the run starts, so it's safe "
          "to run right after launching Terminal 1")
    print(f"tail -f {plan['log_dir']}/*/attempt_0/*/stdout.log")
    print("# (if that glob matches nothing, run: "
          f"find {plan['log_dir']} -name stdout.log   -- to see the real "
          f"path torchrun used)")
    print()
    if args.probe_steps is not None:
        print("# (probe mode: no checkpoints saved, so no eval_after_training.sh "
              "step -- inspect metrics.jsonl directly)")
        print(f"#   {plan['metrics_jsonl']}")
    else:
        print("# --- Run this ONLY after training has finished (not concurrently) ---")
        print(" ".join(plan["eval_after_cmd"]))


def main():
    parser = build_parser()
    args = parser.parse_args()
    plan = compute_plan(args, parser)
    print_plan(args, plan)

    if args.run:
        print(f"\nStarting training now.\n")
        subprocess.run(plan["cmd"], check=True, env=plan["run_env"])

        if args.probe_steps is None:
            print("\nTraining finished successfully. To inspect it, run:\n")
            print(" ".join(plan["eval_after_cmd"]))
        else:
            print(f"\nProbe finished. Metrics at: {plan['metrics_jsonl']}")


if __name__ == "__main__":
    main()