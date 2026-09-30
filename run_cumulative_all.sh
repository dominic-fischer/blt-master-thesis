#!/usr/bin/env bash
set -euo pipefail

# alias | run name | step | extra pps flags
SETTINGS=(
  "Imbalanced|entropy_10M_20lang_4gpu_sourcesimbalanced_steps10000_ckpt200_lr4.5e-3|step_0000002600|"
  "Balanced|entropy_10M_20lang_4gpu_sourcesbalanced_steps10000_ckpt200_lr4.5e-3|step_0000007200|"
  "Balanced-Custom|entropy_10M_20lang_4gpu_sourcesbalanced_steps10000_ckpt200_customenc_lr4.5e-3|step_0000006400|"
)
EXTRA_ARGS="${@}"   # e.g. pass --no-write for a dry run

for entry in "${SETTINGS[@]}"; do
  IFS='|' read -r ALIAS RUN STEP PPS <<< "$entry"
  DIR=results/own_models/$RUN/$STEP
  echo "=== $ALIAS ($DIR)"
  python model_eval/run_cumulative_patching.py --results-dir "$DIR" \
    --score-source both $PPS $EXTRA_ARGS
done