#!/bin/bash
set -f # disables globbing
EXPERIMENT_MAIN_PATH=$(dirname $0)
set +f

"${EXPERIMENT_MAIN_PATH}"/experiment_0.run.sh
"${EXPERIMENT_MAIN_PATH}"/experiment_2.run.sh
"${EXPERIMENT_MAIN_PATH}"/experiment_6.run.sh
"${EXPERIMENT_MAIN_PATH}"/experiment_8.run.sh
"${EXPERIMENT_MAIN_PATH}"/experiment_9.run.sh
"${EXPERIMENT_MAIN_PATH}"/experiment_10.run.sh
