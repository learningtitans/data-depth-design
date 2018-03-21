#!/bin/bash
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
    exit [[[+STATUS_STALE_CODE+]]]
fi

# The experiment is already finished - exits with special code so scheduler won't retry
if [[ "$FINISH_PATH" != "-" ]]; then
    if [[ -e "$FINISH_PATH" ]]; then
        echo 'INFO: this experiment has already finished' >&2
        exit [[[+STATUS_ALREADY_FINISHED+]]]
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
        exit [[[+STATUS_MISSING_INPUTS+]]]
    fi
fi

# Sets trap to return error code if script is interrupted before successful finish
LOCK_SUCCESS=No
FINISH_STATUS=[[[+STATUS_EXECUTION_ERROR+]]]
function finish_trap {
    if [[ "$LOCK_SUCCESS" == "Yes" ]]; then
        rmdir "$LOCK_PATH" &> /dev/null
    fi
    if [[ "$FINISH_STATUS" == "[[[+STATUS_LOCKED+]]]" ]]; then
        echo 'WARNING: experiment discontinued because other process holds its lock' >&2
    else
        if [[ "$FINISH_STATUS" == "[[[+STATUS_OK+]]]" ]]; then
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
    FINISH_STATUS="[[[+STATUS_LOCKED+]]]"
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
