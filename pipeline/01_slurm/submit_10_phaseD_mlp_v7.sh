#!/usr/bin/env bash
# SLURM submit script for Phase D (Deep multi-head training, CNN arch), against V7 dataset
# Usage: sbatch submit_10_phaseD_mlp_v7.sh

#SBATCH --job-name=phaseD_mlp_v7
#SBATCH --output=logs/phaseD_mlp_v7.%j.out
#SBATCH --error=logs/phaseD_mlp_v7.%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

ENV_PATH="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2"
PY_BIN="$ENV_PATH/bin/python3"

PROJECT_ROOT="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"
SCRIPT="$PROJECT_ROOT/pipeline/00_scripts/phaseD_train.py"

export SPIRAL2_COOKED_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v7"

echo "Activating env at $ENV_PATH"
source "$ENV_PATH/bin/activate" 2>/dev/null || true
echo "Running $SCRIPT with $PY_BIN against $SPIRAL2_COOKED_DIR"
cd "$PROJECT_ROOT"
"$PY_BIN" "$SCRIPT" --arch mlp --epochs 50 --batch-size 128 --device cpu
