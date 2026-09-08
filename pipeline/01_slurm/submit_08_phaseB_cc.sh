#!/usr/bin/env bash
# SLURM submit script for Phase B (Classifier Chains)
# Usage: sbatch submit_08_phaseB_cc.sh

#SBATCH --job-name=phaseB_cc
#SBATCH --output=logs/phaseB_cc.%j.out
#SBATCH --error=logs/phaseB_cc.%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=htc

ENV_PATH="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2"
PY_BIN="$ENV_PATH/bin/python3"

PROJECT_ROOT="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"
SCRIPT="$PROJECT_ROOT/pipeline/00_scripts/phaseB_cc_test.py"

echo "Activating env at $ENV_PATH"
source "$ENV_PATH/bin/activate" 2>/dev/null || true
echo "Running $SCRIPT with $PY_BIN"
cd "$PROJECT_ROOT"
"$PY_BIN" "$SCRIPT"
