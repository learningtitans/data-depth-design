#!/bin/bash
set -f # disables globbing
EXPERIMENT_MAIN_PATH=$(dirname $0)
set +f

"${EXPERIMENT_MAIN_PATH}"/experiment_0.run.sh
"${EXPERIMENT_MAIN_PATH}"/experiment_1.run.sh
"${EXPERIMENT_MAIN_PATH}"/experiment_2.run.sh
"${EXPERIMENT_MAIN_PATH}"/experiment_3.run.sh
"${EXPERIMENT_MAIN_PATH}"/experiment_4.run.sh
"${EXPERIMENT_MAIN_PATH}"/experiment_10.run.sh
"${EXPERIMENT_MAIN_PATH}"/experiment_11.run.sh
"${EXPERIMENT_MAIN_PATH}"/experiment_12.run.sh
"${EXPERIMENT_MAIN_PATH}"/experiment_13.run.sh
"${EXPERIMENT_MAIN_PATH}"/experiment_14.run.sh
