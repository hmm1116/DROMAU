#!/usr/bin/env bash
set -euo pipefail
export DATASET=RFMiD
export SHOT=${SHOT:-1}
export QUERY=${QUERY:-2}
export C_LIST=${C_LIST:-1e-3}
exec bash "$(dirname "${BASH_SOURCE[0]}")/../run_experiment.sh"
