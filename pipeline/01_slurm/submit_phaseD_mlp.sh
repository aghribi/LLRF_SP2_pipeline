#!/usr/bin/env bash
#SBATCH --job-name=phaseD_mlp
#SBATCH --output=logs/phaseD_mlp.%j.out
#SBATCH --error=logs/phaseD_mlp.%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

ENV_PATH="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2"
PROJECT_ROOT="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"
SCRIPT="$PROJECT_ROOT/pipeline/00_scripts/phaseD_train.py"

echo "Activating environment at $ENV_PATH"
source "$ENV_PATH/bin/activate" 2>/dev/null || true

echo "Running Phase D MLP training"
echo "Script: $SCRIPT"
echo "Working directory: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

"$ENV_PATH/bin/python" "$SCRIPT" --arch mlp --epochs 50 --batch-size 128 --device cpu

echo "Phase D MLP training complete"
