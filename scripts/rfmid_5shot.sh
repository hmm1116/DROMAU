#!/usr/bin/env bash
set -euo pipefail
export DATASET=RFMiD
export SHOT=${SHOT:-5}
export QUERY=${QUERY:-1}
export C_LIST=${C_LIST:-5e-3}
exec bash "$(dirname "${BASH_SOURCE[0]}")/../run_experiment.sh"
