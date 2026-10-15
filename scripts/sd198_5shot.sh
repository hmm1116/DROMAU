#!/usr/bin/env bash
set -euo pipefail
export DATASET=SD198
export SHOT=${SHOT:-5}
export QUERY=${QUERY:-2}
export TRAIN_SEEDS=${TRAIN_SEEDS:-"1 42 332 1116 1999"}
export TEST_SEEDS=${TEST_SEEDS:-"1 42 332 1116 1999"}
export C_LIST=${C_LIST:-5e-3}
exec bash "$(dirname "${BASH_SOURCE[0]}")/../run_experiment.sh"
