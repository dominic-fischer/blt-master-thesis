"""
launch_training.py

Given a model size (tiny/small/medium/paper-scale/repo-scale), computes and
prints the full torchrun launch command with the correct CLI overrides on
top of entropy_model_2080ti.yaml -- architecture dims, data.root_dir, data
sources/weights, dump_dir, steps (as a generous ceiling, since real
stopping is meant to be plateau-based -- not yet implemented, see the
docstring note below), checkpoint cadence, wandb run name. Avoids
hand-editing the yaml or manually recomputing this math per run.

Design principle (see entropy_model_2080ti.yaml's own header): that yaml is
a TEMPLATE. Every field this script overrides is a placeholder there
("_" or "" -- deliberately chosen so a strict pydantic ValidationError, or
bytelatent's own `assert args.dump_dir`, fires immediately if this script
ever fails to override something it's supposed to). Fields this script
does NOT override (data.sources structure aside -- see below --
distributed.fsdp_type, distributed.model_dtype, optim.clip) are REAL,
deliberately-chosen fallback values in the yaml -- only overridden here if
the corresponding CLI flag is explicitly passed.

NOT YET IMPLEMENTED: plateau-based early stopping. --max-epochs below is a
generous CEILING, not a real training plan -- a safety backstop in case
plateau-detection (external to this script) never fires. Do not treat the
printed step count as "the training plan"; treat it as "the most this run
will ever do."

Shard directory convention (IMPORTANT -- prepare shards to match this
before running): data/lang_shards_<balanced|imbalanced>_<n_gpus>gpu/
    e.g. data/lang_shards_balanced_4gpu, data/lang_shards_imbalanced_4gpu
Keyed by --sources + --n-gpus, NOT by --size -- every model size shares
the exact same shard set for a given --sources choice (see
prepare_language_shards.py and lang_data_ratios_imbalanced.py).

Usage:
    python launch_training.py <size> --n-gpus 4 --sources balanced [--run]

Example:
    python launch_training.py tiny --n-gpus 4 --sources balanced
    python launch_training.py medium --n-gpus 4 --sources imbalanced --run
    python launch_training.py repo-scale --n-gpus 4 --sources balanced --fsdp-type full_shard --run
"""

import argparse
import csv
import json
import os
import subprocess
import sys

import yaml

SIZE_NAME_MAP = {
    "tiny": "Tiny",
    "small": "Small",
    "medium": "Medium",
    "paper-scale": "Paper-scale",
    "repo-scale": "Repo-scale",
}

# Which langs_chosen.csv column to read per --sources choice. Used for BOTH
# (a) each language's training byte target (summed for steps_per_epoch),
# and (b) each language's data.sources.<code> sampling weight -- the SAME
# column for both, deliberately: SamplingIterator renormalizes weights
# before drawing, so a column that's proportional-by-construction to the
# desired weight (as both balanced_allocation_bytes and
# imbalanced_allocation_bytes are) reproduces the exact right normalized
# sampling proportions, with zero risk of the weight column and the
# byte-cap column disagreeing (see prepare_language_shards.py, which uses
# this identical principle).
BYTE_COLUMN_MAP = {
    "balanced": "balanced_allocation_bytes",
    "imbalanced": "imbalanced_allocation_bytes",
}

# Warmup scaling constants: warmup = min(BASE_WARMUP, round(WARMUP_FRACTION *
# total_steps)). No yaml home for these -- they're not real bytelatent
# config fields, just parameters of this launcher's own scaling algorithm,
# so they're plain constants rather than CLI flags with defaults.
# Confirmed necessary: a Tiny run with a flat warmup=500 over only 789
# total steps spent ~63% of training still ramping LR upward, both train
# AND held-out bpb climbing in lockstep the whole time -- not overfitting,
# an LR schedule mismatched to a short run. Scaling avoids that for
# short/small-size runs while leaving long runs (Medium/Repo-scale, where
# 10% of steps still vastly exceeds 500) unaffected.
BASE_WARMUP = 500
WARMUP_FRACTION = 0.1


def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_nested(d: dict, dotted_key: str):
    """Reads a dotted-path key (e.g. 'optim.clip') out of a nested dict."""
    cur = d
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"{dotted_key!r} not found (missing at {key!r})")
        cur = cur[key]
    return cur


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


def load_tuned_batch_size(path: str, size: str) -> int | None:
    """Returns the smoke-tested safe batch size for this size (see the
    companion smoke-testing workflow, saved to
    training_setup/batch_size/smoke_tested_batch_sizes.json), or None if
    the file doesn't exist or has no entry for this size yet."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return None
    value = data.get(size)
    return int(value) if value is not None else None


def save_tuned_batch_size(path: str, size: str, batch_size: int) -> None:
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    data[size] = batch_size
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


def load_language_weights_and_bytes(langs_csv: str, byte_column: str) -> tuple[dict, int]:
    """
    Returns ({language_code: byte_value}, total_bytes) for every row in
    langs_csv that has a value in byte_column. byte_value is used BOTH as
    the data.sources.<code> sampling weight AND summed into total_bytes
    for steps_per_epoch -- see BYTE_COLUMN_MAP's docstring for why the
    same column is correct for both.
    """
    weights = {}
    with open(langs_csv, newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get(byte_column, "n/a")
            if raw in ("n/a", "", None):
                continue
            weights[row["language_code"]] = int(raw)
    if not weights:
        raise ValueError(f"No rows with a value in column {byte_column!r} found in {langs_csv}")
    return weights, sum(weights.values())


def format_value(x: float) -> str:
    """Compact, filename-safe representation of a hyperparameter value.
    Whole numbers print without a decimal point (10 -> '10'); small
    floats print in short scientific notation without a leading zero in
    the exponent (0.0001 -> '1e-4')."""
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
    parser.add_argument("--sources", choices=["balanced", "imbalanced"], required=True,
                         help="Which langs_chosen.csv byte allocation (and, "
                              "by construction, sampling weight) to train "
                              "on -- see lang_data_ratios_imbalanced.py / "
                              "data_to_params_ratio.py for how these two "
                              "columns were derived. No default: this is a "
                              "first-class experimental choice, not "
                              "something to fall back on silently.")
    parser.add_argument("--max-epochs", type=int, default=20,
                         help="A CEILING, not a training plan -- plateau-"
                              "based early stopping is not yet "
                              "implemented (see module docstring), so this "
                              "is just a generous safety backstop on total "
                              "steps. Default 20: empirically, a Tiny run's "
                              "held-out bpb showed strong diminishing "
                              "returns by epoch 20 already (fp32, "
                              "identical setup) -- larger sizes may "
                              "plateau later relative to their own epoch "
                              "count (they sit closer to a sane "
                              "bytes/param ratio per data_to_params_ratio.py), "
                              "so watch their curves and raise this "
                              "manually if needed until real early "
                              "stopping exists.")
    parser.add_argument("--probe-steps", type=int, default=None,
                         help="For quick LR probes (see lr_sweep.py): use "
                              "exactly this many steps instead of computing "
                              "steps from --max-epochs, and disable "
                              "periodic checkpointing/eval entirely (set to "
                              "fire past the end of the run) since a probe "
                              "only needs metrics.jsonl's per-step train "
                              "bpb/grad_norm, logged every logging.freq "
                              "steps regardless of checkpointing. Overrides "
                              "--max-epochs when set.")
    parser.add_argument("--batch-size", type=int, default=None,
                         help="If omitted, looked up from "
                              "--tuned-batch-sizes-file for this exact "
                              "--size. NO FALLBACK: errors out if that "
                              "size has no smoke-tested entry yet and "
                              "--batch-size wasn't passed explicitly -- an "
                              "untuned batch size can OOM immediately and "
                              "the safe value varies a lot by size, so "
                              "guessing isn't safe the way an untuned LR "
                              "merely being suboptimal is.")
    parser.add_argument("--tuned-batch-sizes-file",
                         default="training_setup/batch_size/smoke_tested_batch_sizes.json",
                         help="JSON lookup of {size: batch_size}, populated "
                              "by your smoke-testing workflow (same "
                              "read-modify-write pattern as "
                              "--tuned-lrs-file / lr_sweep.py).")
    parser.add_argument("--configs-csv", default="training_setup/model_configs_computed.csv")
    parser.add_argument("--langs-csv", default="training_setup/langs/langs_chosen.csv")
    parser.add_argument("--base-config", default="entropy_model_2080ti.yaml")
    parser.add_argument("--shard-root-base", default="data",
                         help="Shards expected at "
                              "<base>/lang_shards_<balanced|imbalanced>_<n_gpus>gpu/ "
                              "-- keyed by --sources + --n-gpus, not --size.")
    parser.add_argument("--dump-root", default="dumps")
    parser.add_argument("--log-root", default="logs",
                         help="torchrun writes per-rank stdout/stderr under "
                              "<log-root>/<run_name>/attempt_N/<rank>/ via "
                              "--log-dir, instead of interleaving all ranks "
                              "into one terminal stream.")
    parser.add_argument("--eval-after-script", default="eval_after_training.sh",
                         help="Path to the post-training eval script "
                              "(run once, after training finishes).")
    parser.add_argument("--warmup-fraction", type=float, default=WARMUP_FRACTION,
                         help=f"warmup = min(--base-warmup, round(this * "
                              f"total_steps)). Default {WARMUP_FRACTION}.")
    parser.add_argument("--base-warmup", type=int, default=BASE_WARMUP,
                         help=f"Upper cap for the scaled warmup. Default {BASE_WARMUP}.")
    parser.add_argument("--tuned-lrs-file", default="training_setup/learning_rate/tuned_lrs.json",
                         help="JSON lookup of {size: lr} saved by "
                              "lr_sweep.py --save-best.")
    parser.add_argument("--lr", type=float, default=None,
                         help="optim.lr override. If omitted, uses the "
                              "value saved for this size in "
                              "--tuned-lrs-file. NO FALLBACK: errors out "
                              "if this size has no tuned entry yet and "
                              "--lr wasn't passed explicitly -- run "
                              "lr_sweep.py first, or pass --lr directly.")
    parser.add_argument("--clip", type=float, default=None,
                         help="optim.clip override. If omitted, uses "
                              "the yaml's own real fallback value "
                              "(currently 1.0, bytelatent's own built-in "
                              "OptimArgs default) unchanged.")
    parser.add_argument("--fsdp-type", default=None, choices=["no_shard", "full_shard"],
                         help="Overrides distributed.fsdp_type. If omitted, "
                              "uses the yaml's own real fallback value "
                              "(no_shard) unchanged -- validated safe for "
                              "Medium, confirmed to OOM immediately on "
                              "repo-scale under no_shard. Pass full_shard "
                              "for repo-scale or any size that doesn't fit.")
    parser.add_argument("--model-dtype", default=None, choices=["fp16", "fp32", "bf16"],
                         help="Overrides distributed.model_dtype. If "
                              "omitted, uses the yaml's own real fallback "
                              "value (fp32) unchanged. Confirmed via direct "
                              "fp16-vs-fp32 comparison (identical arch/"
                              "data/schedule): fp16 diverges smoothly and "
                              "monotonically from the first checkpoint "
                              "onward (gradient underflow -- this codebase "
                              "has no GradScaler -- not a spiky blowup "
                              "clipping would catch), fp32 converges "
                              "cleanly. fp32 is the real default here for "
                              "that reason, not just numerical caution in "
                              "the abstract.")
    parser.add_argument("--enable-wandb", action="store_true",
                         help="By default WANDB_MODE=disabled is set so "
                              "wandb is a complete no-op regardless of the "
                              "yaml's wandb config. Pass this flag to "
                              "actually enable wandb logging.")
    parser.add_argument("--run", action="store_true",
                         help="Actually execute the command instead of just printing it")
    return parser


def compute_plan(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    """Turns parsed args into everything needed to print or execute a run:
    the torchrun command, env vars, dump/log dirs, and the post-training
    eval command. Kept separate from main() so lr_sweep.py can build and
    launch runs programmatically using identical logic/defaults."""
    size_row_name = SIZE_NAME_MAP[args.size]
    byte_column = BYTE_COLUMN_MAP[args.sources]

    base_yaml = load_yaml_config(args.base_config)

    # --- LR: no fallback, error if untuned and not explicit ---
    if args.lr is None:
        tuned_lr = load_tuned_lr(args.tuned_lrs_file, args.size)
        if tuned_lr is None:
            raise SystemExit(
                f"No tuned LR for size={args.size!r} in {args.tuned_lrs_file}, "
                f"and --lr wasn't passed explicitly. Run lr_sweep.py for "
                f"this size first (see its --save-best), or pass --lr directly."
            )
        args.lr = tuned_lr
        lr_source = f"tuned value for {args.size}, from {args.tuned_lrs_file}"
    else:
        lr_source = "explicit override"

    # --- batch_size: no fallback, error if untuned and not explicit ---
    if args.batch_size is None:
        tuned_bs = load_tuned_batch_size(args.tuned_batch_sizes_file, args.size)
        if tuned_bs is None:
            raise SystemExit(
                f"No smoke-tested batch size for size={args.size!r} in "
                f"{args.tuned_batch_sizes_file}, and --batch-size wasn't "
                f"passed explicitly. Smoke-test this size first, or pass "
                f"--batch-size directly."
            )
        args.batch_size = tuned_bs
        batch_size_source = f"smoke-tested value for {args.size}, from {args.tuned_batch_sizes_file}"
    else:
        batch_size_source = "explicit override"

    # --- clip: real yaml fallback value, only overridden if explicit ---
    if args.clip is None:
        args.clip = float(get_nested(base_yaml, "optim.clip"))
        clip_source = f"yaml's own real fallback value, from {args.base_config}"
    else:
        clip_source = "explicit override"

    arch = load_architecture(args.configs_csv, size_row_name)
    lang_weights, total_corpus_bytes = load_language_weights_and_bytes(args.langs_csv, byte_column)

    # seq_len is now a fixed, real value in the yaml (8192) -- no longer a
    # CLI-overridable parameter; read it back only to compute
    # bytes_per_step correctly, not to override anything.
    seq_len = int(get_nested(base_yaml, "data.seq_len"))

    bytes_per_step = args.n_gpus * args.batch_size * seq_len
    steps_per_epoch = total_corpus_bytes / bytes_per_step

    if args.probe_steps is not None:
        # Quick LR-probe mode: fixed step count, no periodic
        # checkpoint/eval (set to fire past the end of the run).
        total_steps = args.probe_steps
        checkpoint_every = total_steps + 1
    else:
        checkpoint_every = round(steps_per_epoch)
        # total_steps forced to an exact multiple of checkpoint_every --
        # defense-in-depth against bytelatent's missing-final-checkpoint
        # bug (train.py's "saved" flag guarding the final checkpoint is
        # never reset per-iteration, so once any periodic checkpoint
        # fires, the "always save when training ends" safety net silently
        # becomes a no-op). Confirmed: a Tiny run lost its final ~1 epoch
        # this way. Making total_steps an exact multiple of
        # checkpoint_every means the last periodic checkpoint IS the
        # final step, so nothing is lost.
        total_steps = args.max_epochs * checkpoint_every

    warmup = max(1, min(args.base_warmup, round(args.warmup_fraction * total_steps)))

    shard_root = os.path.join(
        args.shard_root_base, f"lang_shards_{args.sources}_{args.n_gpus}gpu"
    )

    # Suffix for every hyperparameter overridden from its default, so two
    # runs with different lr/clip/etc. for the same size+n_gpus+sources
    # don't silently collide in the same dump_dir/log_dir. --sources is
    # ALWAYS included (like lr) since it's a required, no-default,
    # first-class experimental choice -- never "the default", so always
    # worth having explicit in the name.
    tunable = {
        "max_epochs": "epochs",
        "probe_steps": "probesteps",
        "clip": "clip",
        "batch_size": "bs",
    }
    suffix_parts = [f"sources{args.sources}"]
    for arg_name, label in tunable.items():
        value = getattr(args, arg_name)
        default = parser.get_default(arg_name)
        # clip/batch_size may have been resolved from None -> a looked-up
        # value above; comparing against the raw parser default (None)
        # would always look "non-default" for them once resolved, so we
        # only add these two to the suffix if the ORIGINAL CLI arg was an
        # explicit override, not a lookup.
        if arg_name in ("clip", "batch_size"):
            source = clip_source if arg_name == "clip" else batch_size_source
            if source == "explicit override":
                suffix_parts.append(f"{label}{format_value(value)}")
            continue
        if value != default:
            suffix_parts.append(f"{label}{format_value(value)}")
    if args.fsdp_type is not None:
        suffix_parts.append(f"fsdp{args.fsdp_type}")
    if args.model_dtype is not None:
        suffix_parts.append(f"dtype{args.model_dtype}")
    # lr always included, same reasoning as --sources: its effective value
    # is size-dependent (tuned-LR lookup) and can change over time as
    # tuned_lrs.json gets updated, so always printing the number avoids
    # ambiguity about which numeric lr a given dump_dir actually used.
    suffix_parts.append(f"lr{format_value(args.lr)}")
    run_name_suffix = "_" + "_".join(suffix_parts)

    run_name = f"entropy_{args.size}_20lang_{args.n_gpus}gpu{run_name_suffix}"
    dump_dir = os.path.join(args.dump_root, run_name)
    log_dir = os.path.join(args.log_root, run_name)

    shard_warning = None
    if not os.path.isdir(shard_root):
        shard_warning = (
            f"WARNING: shard directory not found: {shard_root}\n"
            f"  Prepare it first, e.g.:\n"
            f"  python prepare_language_shards.py {args.langs_csv} "
            f"--byte-column {byte_column} {shard_root} --n-chunks {args.n_gpus}"
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
        f"checkpoint.dump.every={checkpoint_every}",
        f"checkpoint.eval.every={checkpoint_every}",
        f"optim.warmup={warmup}",
        f"optim.lr={args.lr}",
        f"optim.clip={args.clip}",
        f"logging.wandb.name={run_name}",
    ]
    # data.sources.<code>=<weight> -- one override per language, always
    # covering every key so nothing is left at its "_" placeholder.
    for code, weight in lang_weights.items():
        overrides.append(f"data.sources.{code}={weight}")

    if args.fsdp_type is not None:
        overrides.append(f"distributed.fsdp_type={args.fsdp_type}")
    if args.model_dtype is not None:
        overrides.append(f"distributed.model_dtype={args.model_dtype}")

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

    eval_after_cmd = [
        "bash", args.eval_after_script,
        dump_dir,
        shard_root,
        str(checkpoint_every),
    ]

    return {
        "arch": arch,
        "size_row_name": size_row_name,
        "byte_column": byte_column,
        "total_corpus_bytes": total_corpus_bytes,
        "steps_per_epoch": steps_per_epoch,
        "checkpoint_every": checkpoint_every,
        "total_steps": total_steps,
        "warmup": warmup,
        "lr_source": lr_source,
        "batch_size_source": batch_size_source,
        "clip_source": clip_source,
        "fsdp_type": args.fsdp_type if args.fsdp_type is not None else "no_shard (yaml default)",
        "model_dtype": args.model_dtype if args.model_dtype is not None else "fp32 (yaml default)",
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
    print(f"# sources: {args.sources} ({plan['byte_column']})")
    print(f"# corpus: {plan['total_corpus_bytes']:,} bytes, "
          f"{plan['steps_per_epoch']:.1f} steps/epoch @ {args.n_gpus} GPUs, "
          f"batch_size={args.batch_size} ({plan['batch_size_source']})")
    if args.probe_steps is not None:
        print(f"# PROBE MODE: steps={plan['total_steps']} (fixed, --max-epochs "
              f"ignored), checkpointing/eval disabled")
    else:
        print(f"# --max-epochs={args.max_epochs} (CEILING, not a training plan -- "
              f"see module docstring) -> steps={plan['total_steps']} "
              f"(exact multiple of checkpoint_every={plan['checkpoint_every']})")
    print(f"# warmup={plan['warmup']} (scaled to {args.warmup_fraction:.0%} of "
          f"total_steps, capped at {args.base_warmup})")
    print(f"# optim.lr={args.lr}  ({plan['lr_source']})")
    print(f"# optim.clip={args.clip}  ({plan['clip_source']})")
    print(f"# distributed.fsdp_type={plan['fsdp_type']}")
    print(f"# distributed.model_dtype={plan['model_dtype']}")
    print()
    print("# --- Training ---")
    for line in plan["env_prefix"][:-1]:
        print(line)
    print(plan["env_prefix"][-1] + " \\")
    print("    " + " \\\n    ".join(plan["cmd"]))
    print()
    print("# --- Terminal 2: watch each rank's log separately ---")
    print(f"tail -f {plan['log_dir']}/*/attempt_0/*/stdout.log")
    print("# (if that glob matches nothing: "
          f"find {plan['log_dir']} -name stdout.log)")
    print()
    if args.probe_steps is not None:
        print("# (probe mode: no checkpoints saved -- inspect metrics.jsonl directly)")
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