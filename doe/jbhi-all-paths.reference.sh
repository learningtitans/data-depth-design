PREDICTIONS_FORMAT="isbi"
PREDICTIONS_FORMAT="titans"

SVM_SUFFIX="nosvm"
SVM_SUFFIX="svm"

DATASET_DIR="$JBHI_DIR/data/[[[:CREATE_DATASET_NAME:]]][[[:CREATE_SEG:]]].[[[:CREATE_RESOLUTION:]]].tfr"
DATASET_DIR="$JBHI_DIR/data/[[[:D1_DATASET_NAME:]]][[[:D1_SEG:]]].[[[:D1_RESOLUTION:]]].tfr"
DATASET_DIR="$JBHI_DIR/data/[[[:D1_DATASET_NAME:]]][[[:D1_SEG:]]].[[[:D1_RESOLUTION:]]].tfr"
DATASET_DIR="$JBHI_DIR/data/[[[:D4_DATASET_NAME:]]][[[:D1_SEG:]]].[[[:D1_RESOLUTION:]]].tfr"
DATASET_DIR="$JBHI_DIR/data/[[[:D4_DATASET_NAME:]]][[[:D1_SEG:]]].[[[:D1_RESOLUTION:]]].tfr"

FEATURES_DIR="$JBHI_DIR/features"
FEATURES_DIR="$JBHI_DIR/features"
FEATURES_DIR="$JBHI_DIR/features"

FINISH_PATH="$OUTPUT_DIR/finish.txt"
FINISH_PATH="$OUTPUT_DIR/finish.txt"
FINISH_PATH="$RESULTS_PREFIX.finish.txt"
FINISH_PATH="$RESULTS_PREFIX.finish.txt"
FINISH_PATH="$SVM_PREFIX.finish.txt"
FINISH_PATH="$TRAIN_FEATURES_PREFIX.finish.txt"
FINISH_PATH="-"

if [[ "[[[:D3_SVM_LAYER:]]]" == "Yes" ]]; then

JBHI_DIR="$HOME/jbhi-special-issue"
JBHI_DIR="$HOME/jbhi-special-issue"
JBHI_DIR="$HOME/jbhi-special-issue"
JBHI_DIR="$HOME/jbhi-special-issue"
JBHI_DIR="$HOME/jbhi-special-issue"
JBHI_DIR="$HOME/jbhi-special-issue"
JBHI_DIR="$HOME/jbhi-special-issue"

LAST_OUTPUT=""
LAST_OUTPUT="$METRICS_PATH"
LAST_OUTPUT="$OUTPUT_DIR/model.ckpt-[[[:D1_MAX_NUMBER_OF_STEPS:]]].meta"
LAST_OUTPUT="$RESULTS_PATH"
LAST_OUTPUT="$RESULTS_PATH"
LAST_OUTPUT="$SVM_PATH"
LAST_OUTPUT="$TRAIN_FEATURES_PATH"

LIST_OF_INPUTS=""
LIST_OF_INPUTS="$DATASET_DIR/finish.txt"
LIST_OF_INPUTS="$DATASET_DIR/finish.txt:$MODEL_DIR/finish.txt"
LIST_OF_INPUTS="$DATASET_DIR/finish.txt:$MODEL_DIR/finish.txt"
LIST_OF_INPUTS="$DATASET_DIR/finish.txt:$MODEL_DIR/finish.txt:$SVM_PREFIX.finish.txt"
LIST_OF_INPUTS="$RESULTS_PREFIX.finish.txt"
LIST_OF_INPUTS="$TRAIN_FEATURES_PREFIX.finish.txt"

LOCK_PATH="$METRICS_PATH.running.lock"
LOCK_PATH="$OUTPUT_DIR/running.lock"
LOCK_PATH="$OUTPUT_DIR/running.lock"
LOCK_PATH="$RESULTS_PREFIX.running.lock"
LOCK_PATH="$RESULTS_PREFIX.running.lock"
LOCK_PATH="$SVM_PREFIX.running.lock"
LOCK_PATH="$TRAIN_FEATURES_PREFIX.running.lock"

METRICS_PATH="$RESULTS_DIR/all_results.anova.txt"
METRICS_TEMP_PATH="$RESULTS_DIR/this_results.anova.txt"

mkdir -p "$DATASET_DIR"
mkdir -p "$FEATURES_DIR"
mkdir -p "$FEATURES_DIR"
mkdir -p "$MODEL_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$SVM_DIR"

MODEL_DIR="$JBHI_DIR/models/deep.[[[-D1_N-]]]"
MODEL_DIR="$JBHI_DIR/models/deep.[[[-D1_N-]]]"
MODEL_DIR="$JBHI_DIR/models/deep.[[[-D1_N-]]]"
MODEL_DIR="$JBHI_DIR/models/deep.[[[-D1_N-]]]"

OUTPUT_DIR="$DATASET_DIR"
OUTPUT_DIR="$MODEL_DIR"

RESULTS_DIR="$JBHI_DIR/results"
RESULTS_DIR="$JBHI_DIR/results"
RESULTS_DIR="$JBHI_DIR/results"

RESULTS_PATH="$RESULTS_PREFIX.results.txt"
RESULTS_PATH="$RESULTS_PREFIX.results.txt"
RESULTS_PATH="$RESULTS_PREFIX.results.txt"

RESULTS_PREFIX="$RESULTS_DIR/deep.[[[-D1_N-]]].layer.[[[-D3_N-]]].test.[[[-D4_N-]]].index.[[[:D4_INDEX:]]].$SVM_SUFFIX"
RESULTS_PREFIX="$RESULTS_DIR/deep.[[[-D1_N-]]].layer.[[[-D3_N-]]].test.[[[-D4_N-]]].index.[[[:D4_INDEX:]]].nosvm"
RESULTS_PREFIX="$RESULTS_DIR/deep.[[[-D1_N-]]].layer.[[[-D3_N-]]].test.[[[-D4_N-]]].index.[[[:D4_INDEX:]]].svm"

SEMIFINISH_PATH="$TEST_FEATURES_PREFIX.finish.txt"

SOURCES_GIT_DIR="$JBHI_DIR/jbhi-special-issue"
SOURCES_GIT_DIR="$JBHI_DIR/jbhi-special-issue"
SOURCES_GIT_DIR="$JBHI_DIR/jbhi-special-issue"
SOURCES_GIT_DIR="$JBHI_DIR/jbhi-special-issue"
SOURCES_GIT_DIR="$JBHI_DIR/jbhi-special-issue"
SOURCES_GIT_DIR="$JBHI_DIR/jbhi-special-issue"
SOURCES_GIT_DIR="$JBHI_DIR/jbhi-special-issue"

START_PATH="$METRICS_PATH.start.txt"
START_PATH="$OUTPUT_DIR/start.txt"
START_PATH="$OUTPUT_DIR/start.txt"
START_PATH="$RESULTS_PREFIX.start.txt"
START_PATH="$RESULTS_PREFIX.start.txt"
START_PATH="$SVM_PREFIX.start.txt"
START_PATH="$TRAIN_FEATURES_PREFIX.start.txt"

SVM_DIR="$JBHI_DIR/svm-models"
SVM_DIR="$JBHI_DIR/svm-models"
SVM_PATH="$SVM_PREFIX.pkl"
SVM_PATH="$SVM_PREFIX.pkl"
SVM_PREFIX="$SVM_DIR/deep.[[[-D1_N-]]].layer.[[[-D3_N-]]].svm"
SVM_PREFIX="$SVM_DIR/deep.[[[-D1_N-]]].layer.[[[-D3_N-]]].svm"

TEST_FEATURES_PATH="$TEST_FEATURES_PREFIX.feats.pkl"
TEST_FEATURES_PREFIX="$FEATURES_DIR/deep.[[[-D1_N-]]].layer.[[[-D3_N-]]].test.[[[-D4_N-]]].index.[[[:D4_INDEX:]]].test"

TRAIN_FEATURES_PATH="$TRAIN_FEATURES_PREFIX.feats.pkl"
TRAIN_FEATURES_PATH="$TRAIN_FEATURES_PREFIX.feats.pkl"

TRAIN_FEATURES_PREFIX="$FEATURES_DIR/deep.[[[-D1_N-]]].layer.[[[-D3_N-]]].train"
TRAIN_FEATURES_PREFIX="$FEATURES_DIR/deep.[[[-D1_N-]]].layer.[[[-D3_N-]]].train"
