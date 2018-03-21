#!/bin/bash
# ========== Experiment Seq. Idx. [[[:D5_INDEX:]]] / [[[-_S-]]] / N. [[[-D5_N-]]] - [[[:D5_FACTORS_LEVELS:]]] ==========
set -u
# Prints header
echo -e '\n\n========== Experiment Seq. Idx. [[[:D5_INDEX:]]] / [[[-_S-]]] / N. [[[-D5_N-]]] - [[[:D5_FACTORS_LEVELS:]]] ==========\n\n'

# Prepares all environment variables
JBHI_DIR="$HOME/jbhi-special-issue"
RESULTS_DIR="$JBHI_DIR/results"
if [[ "[[[:D3_SVM_LAYER:]]]" == "Yes" ]]; then
    SVM_SUFFIX="svm"
    PREDICTIONS_FORMAT="isbi"
else
    SVM_SUFFIX="nosvm"
    PREDICTIONS_FORMAT="titans"
fi
RESULTS_PREFIX="$RESULTS_DIR/deep.[[[-D1_N-]]].layer.[[[-D3_N-]]].test.[[[-D4_N-]]].index.[[[:D4_INDEX:]]].$SVM_SUFFIX"
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

#[[[+$include jbhi-checks+]]]

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
echo -n "[[[-D1_N-]]];[[[-D3_N-]]];[[[-D4_N-]]];" >> "$METRICS_PATH"
echo -n "[[[-a-]]];[[[-b-]]];[[[-c-]]];[[[-d-]]];[[[-e-]]];[[[-f-]]];[[[-g-]]];[[[-h-]]];[[[-i-]]];[[[-j-]]];" >> "$METRICS_PATH"
tail "$METRICS_TEMP_PATH" -n 1 >> "$METRICS_PATH"

#[[[+$include jbhi-footer+]]]
