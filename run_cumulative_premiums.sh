#!/usr/bin/env bash
# Results CSVs (all cases, incl. cumulative_*) + premium .txt files for the
# cumulative cases, for all three settings, bytes + chars.
# Run AFTER `bash run_cumulative_all.sh` (the real run, without --no-write).
# Expected warnings: raw_* columns "not found in ... calibrated bounds" --
# their premium files come from the normal pipeline's own calibration.
set -euo pipefail

SETTINGS=(
  "Imbalanced|entropy_10M_20lang_4gpu_sourcesimbalanced_steps10000_ckpt200_lr4.5e-3|step_0000002600"
  "Balanced|entropy_10M_20lang_4gpu_sourcesbalanced_steps10000_ckpt200_lr4.5e-3|step_0000007200"
  "Balanced-Custom|entropy_10M_20lang_4gpu_sourcesbalanced_steps10000_ckpt200_customenc_lr4.5e-3|step_0000006400"
)

for entry in "${SETTINGS[@]}"; do
  IFS='|' read -r ALIAS RUN STEP <<< "$entry"
  DIR=results/own_models/$RUN/$STEP
  STEM=${RUN}_${STEP}
  for SRC in bytes chars; do
    SUB=$([ "$SRC" = chars ] && echo "char_level/" || echo "")
    echo "=== $ALIAS [$SRC]"
    python results/results_to_CSV.py --results-dir "$DIR" --score-source $SRC --cases all \
      --csv-in-path floresplus_MASTER.csv \
      --csv-out-path results/results_CSV/${SUB}${STEM}_results.csv
    python results/results_to_txt_premiums.py --score-source $SRC --filename-prefix "$ALIAS" \
      --csv-in-path results/results_CSV/${SUB}${STEM}_results.csv \
      --summary-csv calibrated_thresholds/cumulative/${SUB}${STEM}_cumulative_thresholds_summary.csv
  done
done