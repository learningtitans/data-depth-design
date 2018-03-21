#!/bin/bash
#...starts training
if [[ "$EXPERIMENT_STATUS" == "0" ]]; then
    if [[ "$LAST_OUTPUT" == "" || -e "$LAST_OUTPUT" ]]; then
        if [[ "$FINISH_PATH" != "-" ]]; then
            date -u >> "$FINISH_PATH"
            echo GIT "$GIT_COMMIT" >> "$FINISH_PATH"
        fi
        FINISH_STATUS="[[[+STATUS_OK+]]]"
    fi
fi
