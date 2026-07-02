#!/bin/bash
# watch_and_eval.sh
#
# Watches the training run's checkpoint directory and, each time a new
# epoch checkpoint appears, consolidates it and prints held-out bpb --
# giving you a per-epoch progress readout without touching train.py.
#
# Run this in a SEPARATE terminal/tmux pane from the training run itself.
# Uses CPU or a lightly-loaded GPU for eval (doesn't need to contend with
# training's GPUs for anything but a brief forward pass).
#
# Usage:
#   bash watch_and_eval.sh <dump_dir> <lang_shards_root> [poll_seconds]
#
# Example:
#   bash watch_and_eval.sh \
#       /local/scratch/dfische/blt/dumps/entropy_medium_20lang \
#       /local/scratch/dfische/blt/data/lang_shards

set -e

DUMP_DIR="$1"
LANG_SHARDS_ROOT="$2"
POLL_SECONDS="${3:-60}"

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
        epoch_num=$(python3 -c "print(round($step / 1565, 2))")

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