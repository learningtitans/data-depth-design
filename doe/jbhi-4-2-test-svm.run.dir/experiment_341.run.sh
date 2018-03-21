#!/bin/bash

# ========== Experiment Seq. Idx. 341 / 17.0.2 / N. 13/7/0 - _S=17.0.2 D1_N=13 a=1 b=-1 c=1 d=1 e=-1 f=1 D3_N=7 g=1 h=1 i=1 D4_N=0 j=0 ==========
set -u

# Prints header
echo -e '\n\n========== Experiment Seq. Idx. 341 / 17.0.2 / N. 13/7/0 - _S=17.0.2 D1_N=13 a=1 b=-1 c=1 d=1 e=-1 f=1 D3_N=7 g=1 h=1 i=1 D4_N=0 j=0 ==========\n\n'

if [[ "Yes" == "No" ]]; then
    echo 'FATAL: This treatment did not include an SVM layer.'>&2
    echo '       Something very wrong happened!'>&2
    exit 161
fi

# Prepares all environment variables
JBHI_DIR="$HOME/jbhi-special-issue"
DATASET_DIR="$JBHI_DIR/data/fulltrain-seg.598.tfr"
MODEL_DIR="$JBHI_DIR/models/deep.13"
SVM_DIR="$JBHI_DIR/svm-models"
SVM_PREFIX="$SVM_DIR/deep.13.layer.7.svm"
SVM_PATH="$SVM_PREFIX.pkl"
FEATURES_DIR="$JBHI_DIR/features"
TEST_FEATURES_PREFIX="$FEATURES_DIR/deep.13.layer.7.test.0.index.1010.test"
TEST_FEATURES_PATH="$TEST_FEATURES_PREFIX.feats.pkl"
RESULTS_DIR="$JBHI_DIR/results"
RESULTS_PREFIX="$RESULTS_DIR/deep.13.layer.7.test.0.index.1010.svm"
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

#
# Assumes that the following environment variables where initialized
# SOURCES_GIT_DIR="$JBHI_DIR/jbhi-special-issue"
# LIST_OF_INPUTS="$DATASET_DIR/finish.txt:$MODELS_DIR/finish.txt:"
# START_PATH="$OUTPUT_DIR/start.txt"
# FINISH_PATH="$OUTPUT_DIR/finish.txt"
# LOCK_PATH="$OUTPUT_DIR/running.lock"
# LAST_OUTPUT="$MODEL_DIR/[[[:D1_MAX_NUMBER_OF_STEPS:]]].meta"
EXPERIMENT_STATUS=1
STARTED_BEFORE=No

# Checks if code is stable, otherwise alerts scheduler
pushd "$SOURCES_GIT_DIR" >/dev/null
GIT_STATUS=$(git status --porcelain)
GIT_COMMIT=$(git log | head -n 1)
popd >/dev/null
if [ "$GIT_STATUS" != "" ]; then
    echo 'FATAL: there are uncommitted changes in your git sources file' >&2
    echo '       for reproducibility, experiments only run on committed changes' >&2
    echo >&2
    echo '       Git status returned:'>&2
    echo "$GIT_STATUS" >&2
    exit 162
fi

# The experiment is already finished - exits with special code so scheduler won't retry
if [[ "$FINISH_PATH" != "-" ]]; then
    if [[ -e "$FINISH_PATH" ]]; then
        echo 'INFO: this experiment has already finished' >&2
        exit 163
    fi
fi

# The experiment is not ready to run due to dependencies - alerts scheduler
if [[ "$LIST_OF_INPUTS" != "" ]]; then
    IFS=':' tokens_of_input=( $LIST_OF_INPUTS )
    input_missing=No
    for input_to_check in ${tokens_of_input[*]}; do
        if [[ ! -e "$input_to_check" ]]; then
            echo "ERROR: input $input_to_check missing for this experiment" >&2
            input_missing=Yes
        fi
    done
    if [[ "$input_missing" != No ]]; then
        exit 164
    fi
fi

# Sets trap to return error code if script is interrupted before successful finish
LOCK_SUCCESS=No
FINISH_STATUS=161
function finish_trap {
    if [[ "$LOCK_SUCCESS" == "Yes" ]]; then
        rmdir "$LOCK_PATH" &> /dev/null
    fi
    if [[ "$FINISH_STATUS" == "165" ]]; then
        echo 'WARNING: experiment discontinued because other process holds its lock' >&2
    else
        if [[ "$FINISH_STATUS" == "160" ]]; then
            echo 'INFO: experiment finished successfully' >&2
        else
            [[ "$FINISH_PATH" != "-" ]] && rm -f "$FINISH_PATH"
            echo 'ERROR: an error occurred while executing the experiment' >&2
        fi
    fi
    exit "$FINISH_STATUS"
}
trap finish_trap EXIT


# While running, locks experiment so other parallel threads won't attempt to run it too
if mkdir "$LOCK_PATH" --mode=u=rwx,g=rx,o=rx &>/dev/null; then
    LOCK_SUCCESS=Yes
else
    echo 'WARNING: this experiment is already being executed elsewhere' >&2
    FINISH_STATUS="165"
    exit
fi

# If the experiment was started before, do any cleanup necessary
if [[ "$START_PATH" != "-" ]]; then
    if [[ -e "$START_PATH" ]]; then
        echo 'WARNING: this experiment is being restarted' >&2
        STARTED_BEFORE=Yes
    fi

    #...marks start
    date -u >> "$START_PATH"
    echo GIT "$GIT_COMMIT" >> "$START_PATH"
fi


#...gets closest checkpoint file
MODEL_CHECKPOINT=$(ls "$MODEL_DIR/"model.ckpt-*.index | \
    sed 's/.*ckpt-\([0-9]*\)\..*/\1/' | \
    sort -n | \
    awk -v c=1 -v t=40000 \
    'NR==1{d=$c-t;d=d<0?-d:d;v=$c;next}{m=$c-t;m=m<0?-m:m}m<d{d=m;v=$c}END{print v}')
MODEL_PATH="$MODEL_DIR/model.ckpt-$MODEL_CHECKPOINT"
echo "$MODEL_PATH" >> "$START_PATH"

if [[ ! -f "$SEMIFINISH_PATH" ]]; then

    #...performs preliminary feature extraction
    echo Extracting SVM test features with "$MODEL_PATH"
    python \
        "$SOURCES_GIT_DIR/predict_image_classifier.py" \
        --model_name="inception_v4_seg" \
        --checkpoint_path="$MODEL_PATH" \
        --dataset_name=skin_lesions \
        --task_name=label \
        --dataset_split_name=test \
        --preprocessing_name=dermatologic \
        --aggressive_augmentation="True" \
        --add_rotations="True" \
        --minimum_area_to_crop="0.20" \
        --normalize_per_image="0" \
        --batch_size=1 \
        --id_field_name=id \
        --pool_features=avg \
        --extract_features \
        --output_format=pickle \
        --add_scores_to_features=none \
        --eval_replicas="50" \
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


#
#...starts training
if [[ "$EXPERIMENT_STATUS" == "0" ]]; then
    if [[ "$LAST_OUTPUT" == "" || -e "$LAST_OUTPUT" ]]; then
        if [[ "$FINISH_PATH" != "-" ]]; then
            date -u >> "$FINISH_PATH"
            echo GIT "$GIT_COMMIT" >> "$FINISH_PATH"
        fi
        FINISH_STATUS="160"
    fi
fi



