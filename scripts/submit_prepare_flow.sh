#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs
array_job="$(sbatch --parsable scripts/slurm_prepare_flow.sh)"
array_job="${array_job%%;*}"
final_job="$(sbatch --parsable --dependency="afterok:${array_job}" scripts/slurm_finalize_prepare.sh)"
final_job="${final_job%%;*}"
echo "Monthly preparation array: ${array_job} (12 tasks, at most 10 concurrent)"
echo "Dependent finalization job: ${final_job}"
