#!/bin/bash
# eval_during_training.sh
#
# Paired with eval_after_training.sh:
#   eval_during_training.sh -> run in a separate tmux pane WHILE training
#                              is actively running; polls for new
#                              checkpoints one at a time as they appear.
#   eval_after_training.sh  -> run ONCE, after training has finished;
#                              processes every existing checkpoint in a
#                              single pass.
#
# Watches the training run's checkpoint directory and, each time a new
# epoch checkpoint appears, consolidates it and prints held-out bpb --
# giving you a per-epoch progress readout without touching train.py.
#
# Run this in a SEPARATE terminal/tmux pane from the training run itself.
# NOTE: --device cpu was tried here previously but is confirmed BROKEN --
# the model's attention bias/mask still gets constructed on cuda regardless
# (xformers-internal, not something bytelatent's own code controls), causing
# a "query.device: cpu, attn_bias: cuda:0" crash even with cpu forced. So
# this now runs eval on cuda too, sharing GPU 0 with training rank 0.
# Acceptable for small models (Tiny showed ~26% memory use, plenty of
# headroom) -- but worth checking memory margins before using this
# unmodified for Medium/larger, where contention could matter more.
#
# Usage:
#   bash eval_during_training.sh <dump_dir> <lang_shards_root> <steps_per_epoch> [poll_seconds]
#
# steps_per_epoch: from launch_training.py's printed "X steps/epoch" line --
# MUST match the actual run, or displayed epoch numbers will be wrong.
#
# Example:
#   bash eval_during_training.sh \
#       /local/scratch/dfische/blt/dumps/entropy_medium_20lang \
#       /local/scratch/dfische/blt/data/lang_shards \
#       1565

set -e

DUMP_DIR="$1"
LANG_SHARDS_ROOT="$2"
STEPS_PER_EPOCH="$3"
POLL_SECONDS="${4:-60}"

CHECKPOINTS_DIR="$DUMP_DIR/checkpoints"
SEEN_FILE=$(mktemp)

echo "Watching $CHECKPOINTS_DIR for new checkpoints (polling every ${POLL_SECONDS}s)..."

while true; do
    for ckpt_dir in "$CHECKPOINTS_DIR"/*/; do
        ckpt_dir="${ckpt_dir%/}"
        ckpt_name=$(basename "$ckpt_dir")

        if grep -qx "$ckpt_name" "$SEEN_FILE" 2>/dev/null; then
            continue
        fi
        if [ ! -f "$ckpt_dir/.metadata" ]; then
            continue  # still being written
        fi

        step=$((10#$ckpt_name))
        epoch_num=$(python3 -c "print(round($step / $STEPS_PER_EPOCH, 2))")

        echo ""
        echo "=== New checkpoint: step $step (epoch ~$epoch_num) ==="

        echo "Consolidating..."
        python -m bytelatent.checkpoint consolidate "$ckpt_dir"

        echo "Evaluating held-out bpb..."
        python eval_entropy_bpb.py "$ckpt_dir/consolidated" "$LANG_SHARDS_ROOT"

        echo "$ckpt_name" >> "$SEEN_FILE"
    done
    sleep "$POLL_SECONDS"
done