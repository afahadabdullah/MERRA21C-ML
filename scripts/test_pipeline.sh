#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR="${TMPDIR:-/tmp}/merraflow-matplotlib"
python -m pytest -q
python -m merraflow.cli smoke --workdir "${1:-runs/smoke_$(date +%Y%m%d_%H%M%S)}"
