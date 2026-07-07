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

Usage:
    python launch_training.py <size> --n-gpus 4 --epochs 10 [--batch-size 16]
                               [--run]

Example:
    python launch_training.py tiny --n-gpus 4 --epochs 10
    python launch_training.py medium --n-gpus 4 --epochs 10 --run
"""

import argparse
import csv
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("size", choices=list(SIZE_NAME_MAP.keys()))
    parser.add_argument("--n-gpus", type=int, required=True)
    parser.add_argument("--epochs", type=float, default=10)
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
    parser.add_argument("--lr", type=float, default=4e-4,
                         help="optim.lr override. Default 4e-4 matches the "
                              "yaml's own value, which is Meta's original "
                              "repo-scale (dim=768/12h/14L) tuning -- it does "
                              "NOT automatically get lighter for smaller "
                              "architectures, since this script's size "
                              "overrides only touch entropy_model.* dims, "
                              "never optim.lr. Confirmed via metrics.jsonl on "
                              "a Tiny (dim=256/4h/8L) run: grad_norm and bpb "
                              "both climbed steadily for ~200 steps right "
                              "after warmup ended at peak lr=4e-4, even "
                              "though lr was already decaying -- classic "
                              "too-high-lr divergence, not a warmup-length or "
                              "overfitting issue. Re-tune per size; don't "
                              "assume this default transfers as you scale up.")
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
    args = parser.parse_args()

    size_row_name = SIZE_NAME_MAP[args.size]
    byte_column = f"{size_row_name}_bytes"

    arch = load_architecture(args.configs_csv, size_row_name)
    total_corpus_bytes = compute_total_corpus_bytes(args.langs_csv, byte_column)

    bytes_per_step = args.n_gpus * args.batch_size * args.seq_len
    steps_per_epoch = total_corpus_bytes / bytes_per_step
    checkpoint_every = round(steps_per_epoch)
    # total_steps is forced to an exact multiple of checkpoint_every (for
    # whole-number epoch requests) as defense-in-depth against a real
    # bytelatent bug: train.py's "saved" flag guarding the final checkpoint
    # is never reset per-iteration, so once any periodic checkpoint fires,
    # the "always save when training ends" safety net silently becomes a
    # no-op -- meaning any steps after the LAST periodic checkpoint are
    # never persisted. Confirmed: a Tiny run (steps=2625,
    # checkpoint.dump.every=263) lost its final 258 steps (~1 epoch) this
    # way. Making total_steps an exact multiple of checkpoint_every means
    # the last periodic checkpoint IS the final step, so nothing is lost
    # even if train.py hasn't been patched on a given machine.
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
    # even if a default above is changed later.
    tunable = {
        "epochs": "epochs",
        "lr": "lr",
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
    run_name_suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

    run_name = f"entropy_{args.size}_20lang_{args.n_gpus}gpu{run_name_suffix}"
    dump_dir = os.path.join(args.dump_root, run_name)
    # log_dir intentionally mirrors dump_dir exactly (same run_name, only the
    # root differs: dumps/ vs logs/) -- both derive from the single run_name
    # variable so they can't silently drift apart from each other.
    log_dir = os.path.join(args.log_root, run_name)

    if not os.path.isdir(shard_root):
        print(f"WARNING: shard directory not found: {shard_root}")
        print(f"  Prepare it first, e.g.:")
        print(f"  python prepare_language_shards.py {args.langs_csv} {byte_column} "
              f"{shard_root} --n-chunks {args.n_gpus}")
        print()

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

    # Post-training eval only -- do NOT run this while training is still
    # active (see module docstring: concurrent eval is a suspected cause
    # of silent rank crashes on shared/contended GPUs).
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

    print(f"# {size_row_name}: {arch['total_params']:,} params, "
          f"dim={arch['dim']} n_layers={arch['n_layers']} n_heads={arch['n_heads']}")
    print(f"# corpus: {total_corpus_bytes:,} bytes, "
          f"{steps_per_epoch:.1f} steps/epoch @ {args.n_gpus} GPUs, "
          f"batch_size={args.batch_size}, seq_len={args.seq_len}")
    print(f"# {round(args.epochs)} epochs -> steps={total_steps} "
          f"(exact multiple of checkpoint_every={checkpoint_every}, "
          f"guards against the missing-final-checkpoint bug)")
    print(f"# warmup={warmup} (scaled to {args.warmup_fraction:.0%} of "
          f"total_steps, capped at {args.base_warmup})")
    print(f"# optim.lr={args.lr}  optim.clip={args.clip}"
          + ("  (yaml defaults -- not yet re-tuned for this size)"
             if (args.lr == 4e-4 and args.clip == 10.0) else ""))
    print()
    print("# --- Training ---")
    for line in env_prefix[:-1]:
        print(line)
    print(env_prefix[-1] + " \\")
    print("    " + " \\\n    ".join(cmd))
    print()
    print("# --- Terminal 2: watch each rank's log separately (avoids the "
          "interleaved single-stream mess) ---")
    print("# torchrun creates this the moment the run starts, so it's safe "
          "to run right after launching Terminal 1")
    print(f"tail -f {log_dir}/*/attempt_0/*/stdout.log")
    print("# (if that glob matches nothing, run: "
          f"find {log_dir} -name stdout.log   -- to see the real path torchrun used)")
    print()
    print("# --- Run this ONLY after training has finished (not concurrently) ---")
    print(" ".join(eval_after_cmd))

    if args.run:
        print(f"\nStarting training now.\n")
        run_env = dict(os.environ)
        run_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        run_env["PYTHONUNBUFFERED"] = "1"
        if not args.enable_wandb:
            run_env["WANDB_MODE"] = "disabled"
        run_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        subprocess.run(cmd, check=True, env=run_env)

        print("\nTraining finished successfully. To inspect it, run:\n")
        print(" ".join(eval_after_cmd))


if __name__ == "__main__":
    main()