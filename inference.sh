#!/bin/bash

# Check if the folder path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <folder_path>"
  exit 1
fi

FOLDER_PATH=$1
EXP_NAME=$(basename "$FOLDER_PATH")

# Find the best_mIoU checkpoint file
BEST_CHECKPOINT=$(find "$FOLDER_PATH" -maxdepth 1 -type f -name "best_mIoU_iter_*.pth" | head -n 1)

if [ -z "$BEST_CHECKPOINT" ]; then
  echo "No checkpoint file matching pattern 'best_mIoU_iter_*.pth' found in $FOLDER_PATH"
  exit 1
fi

# Define checkpoints
CHECKPOINTS=("$BEST_CHECKPOINT" "${FOLDER_PATH}/latest.pth")

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
  CHECKPOINT_FILE="$CHECKPOINT"
  CONFIG_FILE="${FOLDER_PATH}/${EXP_NAME}.json"
  SHOW_DIR="${FOLDER_PATH}/$(basename "$CHECKPOINT" .pth)_${EXP_NAME}"

  echo "Running inference for checkpoint: ${CHECKPOINT}"
  echo 'Config File:' $CONFIG_FILE
  echo 'Checkpoint File:' $CHECKPOINT_FILE
  echo 'Predictions Output Directory:' $SHOW_DIR

  python -m tools.test "$CONFIG_FILE" "$CHECKPOINT_FILE" --show-dir "$SHOW_DIR" --opacity 1 &
  python tools/loveda_mask_convert.py --mask-dir "$SHOW_DIR" --rgb2mask &
done
wait
echo "Inference completed for all checkpoints."