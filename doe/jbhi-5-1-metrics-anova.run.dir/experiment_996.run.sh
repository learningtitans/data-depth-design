#!/bin/bash

# ========== Experiment Seq. Idx. 1388 / 24.7.2.0 / N. 0 - _S=24.7.2.0 D1_N=7 a=1 b=1 c=1 d=-1 e=-1 f=1 D3_N=1 g=-1 h=-1 i=1 D4_N=0 j=0 D5_N=0 ==========
set -u
# Prints header
echo -e '\n\n========== Experiment Seq. Idx. 1388 / 24.7.2.0 / N. 0 - _S=24.7.2.0 D1_N=7 a=1 b=1 c=1 d=-1 e=-1 f=1 D3_N=1 g=-1 h=-1 i=1 D4_N=0 j=0 D5_N=0 ==========\n\n'

# Prepares all environment variables
JBHI_DIR="$HOME/jbhi-special-issue"
RESULTS_DIR="$JBHI_DIR/results"
if [[ "No" == "Yes" ]]; then
    SVM_SUFFIX="svm"
    PREDICTIONS_FORMAT="isbi"
else
    SVM_SUFFIX="nosvm"
    PREDICTIONS_FORMAT="titans"
fi
RESULTS_PREFIX="$RESULTS_DIR/deep.7.layer.1.test.0.index.1388.$SVM_SUFFIX"
RESULTS_PATH="$RESULTS_PREFIX.results.txt"
# ...variables expected by jbhi-checks.include.sh and jbhi-footer.include.sh
SOURCES_GIT_DIR="$JBHI_DIR/jbhi-special-issue"
LIST_OF_INPUTS="$RESULTS_PREFIX.finish.txt"
# ...this experiment is a little different --- only one master procedure should run, so there's only a master lock file
METRICS_TEMP_PATH="$RESULTS_DIR/this_results.anova.txt"
METRICS_PATH="$RESULTS_DIR/all_results.anova.txt"
START_PATH="$METRICS_PATH.start.txt"
FINISH_PATH="-"
LOCK_PATH="$METRICS_PATH.running.lock"
LAST_OUTPUT="$METRICS_PATH"
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


if [[ "$STARTED_BEFORE" == "Yes" ]]; then
    # If the experiment was started before, do any cleanup necessary
    echo -n
else
    echo "D1_N;D3_N;D4_N;a;b;c;d;e;f;g;h;i;j;m_ap;m_auc;m_tn;m_fp;m_fn;m_tp;m_tpr;m_fpr;k_ap;k_auc;k_tn;k_fp;k_fn;k_tp;k_tpr;k_fpr;isbi_auc" > "$METRICS_PATH"
fi

python \
    "$SOURCES_GIT_DIR/etc/compute_metrics.py" \
        --metadata_file "$SOURCES_GIT_DIR/data/all-metadata.csv" \
        --predictions_format "$PREDICTIONS_FORMAT" \
        --metrics_file "$METRICS_TEMP_PATH" \
        --predictions_file "$RESULTS_PATH"
EXPERIMENT_STATUS="$?"
echo -n "7;1;0;" >> "$METRICS_PATH"
echo -n "1;1;1;-1;-1;1;-1;-1;1;0;" >> "$METRICS_PATH"
tail "$METRICS_TEMP_PATH" -n 1 >> "$METRICS_PATH"

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


