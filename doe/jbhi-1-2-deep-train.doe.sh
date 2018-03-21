#!/bin/bash
# ========== Experiment Idx. [[[:D2_INDEX:]]] / Seq. [[[-_S-]]] / N. [[[-D1_N-]]] - [[[:D2_FACTORS_LEVELS:]]] ==========\n\n'
set -u

# Prints header
echo -e '\n\n========== Experiment Idx. [[[:D2_INDEX:]]] / Seq. [[[-_S-]]] / N. [[[-D1_N-]]] - [[[:D2_FACTORS_LEVELS:]]] ==========\n\n'

# Prepares all environment variables
JBHI_DIR="$HOME/jbhi-special-issue"
DATASET_DIR="$JBHI_DIR/data/[[[:D1_DATASET_NAME:]]][[[:D1_SEG:]]].[[[:D1_RESOLUTION:]]].tfr"
MODEL_DIR="$JBHI_DIR/models/deep.[[[-D1_N-]]]"
OUTPUT_DIR="$MODEL_DIR"
# ...variables expected by jbhi-checks.include.sh and jbhi-footer.include.sh
SOURCES_GIT_DIR="$JBHI_DIR/jbhi-special-issue"
LIST_OF_INPUTS="$DATASET_DIR/finish.txt"
START_PATH="$OUTPUT_DIR/start.txt"
FINISH_PATH="$OUTPUT_DIR/finish.txt"
LOCK_PATH="$OUTPUT_DIR/running.lock"
LAST_OUTPUT="$OUTPUT_DIR/model.ckpt-[[[:D1_MAX_NUMBER_OF_STEPS:]]].meta"
# EXPERIMENT_STATUS=1
# STARTED_BEFORE=No
mkdir -p "$MODEL_DIR"

#[[[+$include jbhi-checks+]]]

# If the experiment was started before, do any cleanup necessary
if [[ "$STARTED_BEFORE" == "Yes" ]]; then
    echo -n
fi

#...starts training
python \
    "$SOURCES_GIT_DIR/train_image_classifier.py" \
    --model_name="[[[:D1_MODEL:]]]" \
    --max_number_of_steps="[[[:D1_MAX_NUMBER_OF_STEPS:]]]" \
    --optimizer=rmsprop \
    --dataset_name=skin_lesions \
    --task_name=label \
    --dataset_split_name=train \
    --preprocessing_name=dermatologic \
    --aggressive_augmentation="[[[:D1_AGGRESSIVE_AUGMENTATION:]]]" \
    --add_rotations="[[[:D1_ADD_ROTATIONS:]]]" \
    --minimum_area_to_crop="[[[:D1_MINIMUM_AREA:]]]" \
    --normalize_per_image="[[[:D1_NORMALIZE_PER_IMAGE:]]]" \
    --save_interval_secs=3600 \
    --experiment_file="$MODEL_DIR/experiment.meta" \
    --experiment_tag="Factorial: [[[-D1_N-]]]; [[[:D1_FACTORS:]]]" \
    --dataset_dir="$DATASET_DIR" \
    --train_dir="$MODEL_DIR"
    # Tip: leave last the arguments that make the command fail if they're absent,
    # so if there's a typo or forgotten \ the entire thing fails
EXPERIMENT_STATUS="$?"

#    --checkpoint_path="$JBHI_DIR/running/[[[:D1_START_CHECKPOINT:]]]" \
#    --checkpoint_exclude_scopes="[[[:D1_CHECKPOINT_EXCLUDE_SCOPES:]]]" \
#    --ignore_missing_vars="[[[:D1_IGNORE_MISSING_VARS:]]]" \

#[[[+$include jbhi-footer+]]]
