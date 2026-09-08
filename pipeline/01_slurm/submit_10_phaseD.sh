#!/usr/bin/env bash
# SLURM submit script for Phase D (Deep multi-head training)
# Usage: sbatch submit_10_phaseD.sh

#SBATCH --job-name=phaseD
#SBATCH --output=logs/phaseD.%j.out
#SBATCH --error=logs/phaseD.%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

ENV_PATH="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2"
PY_BIN="$ENV_PATH/bin/python3"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)/.."
SCRIPT="$PROJECT_ROOT/pipeline/00_scripts/phaseD_train.py"

echo "Activating env at $ENV_PATH"
source "$ENV_PATH/bin/activate" 2>/dev/null || true
echo "Running $SCRIPT with $PY_BIN"
"$PY_BIN" "$SCRIPT" --arch cnn --epochs 30 --batch-size 128 --device cuda
