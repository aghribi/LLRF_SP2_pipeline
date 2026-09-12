#!/usr/bin/env bash
# SLURM submit script for Phase C (Label Powerset), pre-trigger-only, V7 dataset
# Usage: sbatch submit_09_phaseC_lp_pretrigger_v7.sh

#SBATCH --job-name=phaseC_lp_pretrigger_v7
#SBATCH --output=logs/phaseC_lp_pretrigger_v7.%j.out
#SBATCH --error=logs/phaseC_lp_pretrigger_v7.%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=htc

ENV_PATH="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2"
PY_BIN="$ENV_PATH/bin/python3"

PROJECT_ROOT="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"
SCRIPT="$PROJECT_ROOT/pipeline/00_scripts/phaseC_lp_test.py"

export SPIRAL2_COOKED_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v7"
export SPIRAL2_PRETRIGGER="1"

echo "Activating env at $ENV_PATH"
source "$ENV_PATH/bin/activate" 2>/dev/null || true
echo "Running $SCRIPT with $PY_BIN against $SPIRAL2_COOKED_DIR (pretrigger=1)"
cd "$PROJECT_ROOT"
"$PY_BIN" "$SCRIPT"
