# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

#!/bin/bash

TEST_ROOT=$1
CHECKPOINT=$2
# CONFIG_FILE="${TEST_ROOT}/*$(echo -n $TEST_ROOT | tail -c 1).json"
EXP_NAME=$(basename "$TEST_ROOT")
CONFIG_FILE="${TEST_ROOT}/${EXP_NAME}.json"
CHECKPOINT_FILE="${TEST_ROOT}/${CHECKPOINT}.pth"
SHOW_DIR="${TEST_ROOT}/tsne_visualization"
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR
python -m tools.tsne_visualization ${CONFIG_FILE} ${CHECKPOINT_FILE} --show-dir ${SHOW_DIR} --num-images 70