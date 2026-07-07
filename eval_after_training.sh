#!/bin/bash
# eval_after_training.sh
#
# Paired with eval_during_training.sh:
#   eval_during_training.sh -> run in a separate tmux pane WHILE training
#                              is actively running; polls for new
#                              checkpoints one at a time as they appear.
#   eval_after_training.sh  -> run ONCE, after training has finished;
#                              processes every existing checkpoint in a
#                              single pass.
#
# Retroactively consolidates and evaluates every saved checkpoint. Prints
# one bpb table per epoch, in order. Use this after training has completed
# (e.g. to recover a full progression if eval_during_training.sh crashed
# partway through, or to inspect a completed run after the fact).
#
# Usage:
#   bash eval_after_training.sh <dump_dir> <lang_shards_root> <steps_per_epoch>
#
# steps_per_epoch: from launch_training.py's printed "X steps/epoch" line --
# MUST match the actual run, or displayed epoch numbers will be wrong.

set -e  # ok here: a bad checkpoint should stop this script since it's
        # a one-shot retrospective run, not a long-lived watcher

DUMP_DIR="$1"
LANG_SHARDS_ROOT="$2"
STEPS_PER_EPOCH="$3"
CHECKPOINTS_DIR="$DUMP_DIR/checkpoints"

for ckpt_dir in "$CHECKPOINTS_DIR"/*/; do
    ckpt_dir="${ckpt_dir%/}"
    ckpt_name=$(basename "$ckpt_dir")
    step=$((10#$ckpt_name))
    epoch_num=$(python3 -c "print(round($step / $STEPS_PER_EPOCH, 2))")

    echo ""
    echo "=== step $step (epoch ~$epoch_num) ==="

    if [ ! -d "$ckpt_dir/consolidated" ]; then
        echo "Consolidating..."
        python -m bytelatent.checkpoint consolidate "$ckpt_dir"
    else
        echo "Already consolidated, skipping."
    fi

    echo "Evaluating held-out bpb..."
    python eval_entropy_bpb.py "$ckpt_dir/consolidated" "$LANG_SHARDS_ROOT"
done