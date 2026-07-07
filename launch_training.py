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
    dump_dir = os.path.join(args.dump_root, f"entropy_{args.size}_20lang_{args.n_gpus}gpu")
    run_name = f"entropy_{args.size}_20lang_{args.n_gpus}gpu"

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
        f"logging.wandb.name={run_name}",
    ]

    cmd = (
        ["torchrun", f"--nproc_per_node={args.n_gpus}", "--standalone",
         "-m", "bytelatent.train", f"config={args.base_config}"]
        + overrides
    )

    cuda_visible_devices = ",".join(str(i) for i in range(args.n_gpus))
    env_prefix = [
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}",
    ]

    watch_eval_cmd = [
        "bash", "eval_during_training.sh",
        dump_dir,
        shard_root,
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
    print()
    print("# --- Terminal 1: training ---")
    print(env_prefix[0])
    print(env_prefix[1] + " \\")
    print("    " + " \\\n    ".join(cmd))
    print()
    print("# --- Terminal 2: copy this one line ---")
    print(" ".join(watch_eval_cmd))

    if args.run:
        print(f"\nStarting training now. Once it's progressing, copy the "
              f"Terminal 2 line above into a separate terminal/tmux pane.\n")
        run_env = dict(os.environ)
        run_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        run_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        subprocess.run(cmd, check=True, env=run_env)


if __name__ == "__main__":
    main()