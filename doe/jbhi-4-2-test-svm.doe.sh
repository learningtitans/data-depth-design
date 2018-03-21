#!/bin/bash
# ========== Experiment Seq. Idx. [[[:D4_2_INDEX:]]] / [[[-_S-]]] / N. [[[-D1_N-]]]/[[[-D3_N-]]]/[[[-D4_N-]]] - [[[:D4_2_FACTORS_LEVELS:]]] ==========
set -u

# Prints header
echo -e '\n\n========== Experiment Seq. Idx. [[[:D4_2_INDEX:]]] / [[[-_S-]]] / N. [[[-D1_N-]]]/[[[-D3_N-]]]/[[[-D4_N-]]] - [[[:D4_2_FACTORS_LEVELS:]]] ==========\n\n'

if [[ "[[[:D3_SVM_LAYER:]]]" == "No" ]]; then
    echo 'FATAL: This treatment did not include an SVM layer.'>&2
    echo '       Something very wrong happened!'>&2
    exit [[[+STATUS_EXECUTION_ERROR+]]]
fi

# Prepares all environment variables
JBHI_DIR="$HOME/jbhi-special-issue"
DATASET_DIR="$JBHI_DIR/data/[[[:D4_DATASET_NAME:]]][[[:D1_SEG:]]].[[[:D1_RESOLUTION:]]].tfr"
MODEL_DIR="$JBHI_DIR/models/deep.[[[-D1_N-]]]"
SVM_DIR="$JBHI_DIR/svm-models"
SVM_PREFIX="$SVM_DIR/deep.[[[-D1_N-]]].layer.[[[-D3_N-]]].svm"
SVM_PATH="$SVM_PREFIX.pkl"
FEATURES_DIR="$JBHI_DIR/features"
TEST_FEATURES_PREFIX="$FEATURES_DIR/deep.[[[-D1_N-]]].layer.[[[-D3_N-]]].test.[[[-D4_N-]]].index.[[[:D4_INDEX:]]].test"
TEST_FEATURES_PATH="$TEST_FEATURES_PREFIX.feats.pkl"
RESULTS_DIR="$JBHI_DIR/results"
RESULTS_PREFIX="$RESULTS_DIR/deep.[[[-D1_N-]]].layer.[[[-D3_N-]]].test.[[[-D4_N-]]].index.[[[:D4_INDEX:]]].svm"
RESULTS_PATH="$RESULTS_PREFIX.results.txt"
# ...variables expected by jbhi-checks.include.sh and jbhi-footer.include.sh
SOURCES_GIT_DIR="$JBHI_DIR/jbhi-special-issue"
LIST_OF_INPUTS="$DATASET_DIR/finish.txt:$MODEL_DIR/finish.txt:$SVM_PREFIX.finish.txt"
START_PATH="$RESULTS_PREFIX.start.txt"
FINISH_PATH="$RESULTS_PREFIX.finish.txt"
LOCK_PATH="$RESULTS_PREFIX.running.lock"
LAST_OUTPUT="$RESULTS_PATH"
# ...creates mid-way checkpoint after the expensive test features extraction
SEMIFINISH_PATH="$TEST_FEATURES_PREFIX.finish.txt"
# EXPERIMENT_STATUS=1
# STARTED_BEFORE=No
mkdir -p "$FEATURES_DIR"
mkdir -p "$RESULTS_DIR"

#[[[+$include jbhi-checks+]]]

#...gets closest checkpoint file
MODEL_CHECKPOINT=$(ls "$MODEL_DIR/"model.ckpt-*.index | \
    sed 's/.*ckpt-\([0-9]*\)\..*/\1/' | \
    sort -n | \
    awk -v c=1 -v t=[[[:D3_CHECKPOINT:]]] \
    'NR==1{d=$c-t;d=d<0?-d:d;v=$c;next}{m=$c-t;m=m<0?-m:m}m<d{d=m;v=$c}END{print v}')
MODEL_PATH="$MODEL_DIR/model.ckpt-$MODEL_CHECKPOINT"
echo "$MODEL_PATH" >> "$START_PATH"

if [[ ! -f "$SEMIFINISH_PATH" ]]; then

    #...performs preliminary feature extraction
    echo Extracting SVM test features with "$MODEL_PATH"
    python \
        "$SOURCES_GIT_DIR/predict_image_classifier.py" \
        --model_name="[[[:D1_MODEL:]]]" \
        --checkpoint_path="$MODEL_PATH" \
        --dataset_name=skin_lesions \
        --task_name=label \
        --dataset_split_name=test \
        --preprocessing_name=dermatologic \
        --aggressive_augmentation="[[[:D1_AGGRESSIVE_AUGMENTATION:]]]" \
        --add_rotations="[[[:D1_ADD_ROTATIONS:]]]" \
        --minimum_area_to_crop="[[[:D1_MINIMUM_AREA:]]]" \
        --normalize_per_image="[[[:D1_NORMALIZE_PER_IMAGE:]]]" \
        --batch_size=1 \
        --id_field_name=id \
        --pool_features=avg \
        --extract_features \
        --output_format=pickle \
        --add_scores_to_features=none \
        --eval_replicas="[[[:D3_TEST_REPLICAS:]]]" \
        --output_file="$TEST_FEATURES_PATH" \
        --dataset_dir="$DATASET_DIR"
        # Tip: leave last the arguments that make the command fail if they're absent,
        # so if there's a typo or forgotten \ the entire thing fails
    EXPERIMENT_STATUS="$?"

    if [[ "$EXPERIMENT_STATUS" != "0" || ! -e "$TEST_FEATURES_PATH" ]]; then
        exit
    fi

    date -u >> "$SEMIFINISH_PATH"
    echo GIT "$GIT_COMMIT" >> "$SEMIFINISH_PATH"

else

    echo Reloading features from "$TEST_FEATURES_PATH"

fi

#...performs prediction with SVM model
python \
    "$SOURCES_GIT_DIR/predict_svm_layer.py" \
    --output_file "$RESULTS_PATH" \
    --input_test "$TEST_FEATURES_PATH" \
    --input_model "$SVM_PATH"
    # Tip: leave last the arguments that make the command fail if they're absent,
    # so if there's a typo or forgotten \ the entire thing fails
EXPERIMENT_STATUS="$?"


#[[[+$include jbhi-footer+]]]

