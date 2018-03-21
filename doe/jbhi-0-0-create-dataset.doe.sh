#!/bin/bash
# ========== Dataset Idx. [[[:CREATE_INDEX:]]] / Seq. [[[-_S-]]] / N. [[[-CREATE_N-]]] - [[[:CREATE_FACTORS_LEVELS:]]] ==========
set -u

# Prints header
echo -e '\n\n\n========== Dataset Idx. [[[:CREATE_INDEX:]]] / Seq. [[[-_S-]]] / N. [[[-CREATE_N-]]] - [[[:CREATE_FACTORS_LEVELS:]]] ==========\n\n\n'

# Prepares all environment variables
JBHI_DIR="$HOME/jbhi-special-issue"
DATASET_DIR="$JBHI_DIR/data/[[[:CREATE_DATASET_NAME:]]][[[:CREATE_SEG:]]].[[[:CREATE_RESOLUTION:]]].tfr"
OUTPUT_DIR="$DATASET_DIR"
# ...variables expected by jbhi-checks.include.sh and jbhi-footer.include.sh
SOURCES_GIT_DIR="$JBHI_DIR/jbhi-special-issue"
LIST_OF_INPUTS=""
START_PATH="$OUTPUT_DIR/start.txt"
FINISH_PATH="$OUTPUT_DIR/finish.txt"
LOCK_PATH="$OUTPUT_DIR/running.lock"
LAST_OUTPUT=""
# EXPERIMENT_STATUS=1
# STARTED_BEFORE=No
mkdir -p "$DATASET_DIR"

#[[[+$include jbhi-checks+]]]

# If the experiment was started before, do any cleanup necessary
if [[ "$STARTED_BEFORE" == "Yes" ]]; then
    rm "$OUTPUT_DIR/*.tfrecord" -f
    rm "$OUTPUT_DIR/*.pkl" -f
fi

#...creates dataset
if [ "[[[:CREATE_SEGMENTED:]]]" == "Yes" ]; then
    echo "(Adding segmentation masks)"
    python "$SOURCES_GIT_DIR/datasets/convert_skin_lesions.py" \
        [[[:CREATE_DATASET_TRAIN_SOURCE:]]] \
        [[[:CREATE_DATASET_TEST_SOURCE:]]] \
        [[[:CREATE_MASKS_SOURCE:]]] \
        --images_dir "$JBHI_DIR/skinLesions/[[[:CREATE_RESOLUTION:]]]/images" \
        --output_dir "$OUTPUT_DIR"
else
    python "$SOURCES_GIT_DIR/datasets/convert_skin_lesions.py" \
        [[[:CREATE_DATASET_TRAIN_SOURCE:]]] \
        [[[:CREATE_DATASET_TEST_SOURCE:]]] \
        --images_dir "$JBHI_DIR/skinLesions/[[[:CREATE_RESOLUTION:]]]/images" \
        --output_dir "$OUTPUT_DIR"
fi
EXPERIMENT_STATUS="$?"

last_dataset_file=$(ls -1 "$OUTPUT_DIR" | grep '\([0-9]*\)-of-\1.tfrecord$')
if [[ "$last_dataset_file" == "" ]]; then
    EXPERIMENT_STATUS=1
fi

#[[[+$include jbhi-footer+]]]
