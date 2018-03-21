#!/bin/bash
# ========== Experiment Seq. Idx. [[[:D3_2_INDEX:]]] / [[[-_S-]]] / N. [[[-D1_N-]]]/[[[-D3_N-]]] - [[[:D3_2_FACTORS_LEVELS:]]] ==========
set -u

# Prints header
echo -e '\n\n========== Experiment Seq. Idx. [[[:D3_2_INDEX:]]] / [[[-_S-]]] / N. [[[-D1_N-]]]/[[[-D3_N-]]] - [[[:D3_2_FACTORS_LEVELS:]]] ==========\n\n'

if [[ "[[[:D3_SVM_LAYER:]]]" == "No" ]]; then
    echo 'FATAL: This treatment did not include an SVM layer.'>&2
    echo '       Something very wrong happened!'>&2
    exit [[[+STATUS_EXECUTION_ERROR+]]]
fi

# Prepares all environment variables
JBHI_DIR="$HOME/jbhi-special-issue"
SVM_DIR="$JBHI_DIR/svm-models"
SVM_PREFIX="$SVM_DIR/deep.[[[-D1_N-]]].layer.[[[-D3_N-]]].svm"
SVM_PATH="$SVM_PREFIX.pkl"
FEATURES_DIR="$JBHI_DIR/features"
TRAIN_FEATURES_PREFIX="$FEATURES_DIR/deep.[[[-D1_N-]]].layer.[[[-D3_N-]]].train"
TRAIN_FEATURES_PATH="$TRAIN_FEATURES_PREFIX.feats.pkl"
# ...variables expected by jbhi-checks.include.sh and jbhi-footer.include.sh
SOURCES_GIT_DIR="$JBHI_DIR/jbhi-special-issue"
LIST_OF_INPUTS="$TRAIN_FEATURES_PREFIX.finish.txt"
START_PATH="$SVM_PREFIX.start.txt"
FINISH_PATH="$SVM_PREFIX.finish.txt"
LOCK_PATH="$SVM_PREFIX.running.lock"
LAST_OUTPUT="$SVM_PATH"
# EXPERIMENT_STATUS=1
# STARTED_BEFORE=No
mkdir -p "$SVM_DIR"

#[[[+$include jbhi-checks+]]]

# If the experiment was started before, do any cleanup necessary
if [[ "$STARTED_BEFORE" == "Yes" ]]; then
    echo -n
fi

#...trains SVM layer
echo Training SVM layer from "$TRAIN_FEATURES_PATH"
python \
    "$SOURCES_GIT_DIR/train_svm_layer.py" \
     --jobs 1 \
     --svm_method LINEAR_PRIMAL \
     --output_model   "$SVM_PATH" \
     --input_training "$TRAIN_FEATURES_PATH"
    # Tip: leave last the arguments that make the command fail if they're absent,
    # so if there's a typo or forgotten \ the entire thing fails
EXPERIMENT_STATUS="$?"

#[[[+$include jbhi-footer+]]]

