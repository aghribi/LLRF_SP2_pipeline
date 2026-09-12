#!/bin/bash
#SBATCH --job-name=step_07_v7
#SBATCH --output=logs/step_07_v7_%j.out
#SBATCH --error=logs/step_07_v7_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=htc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

echo "STEP 07: Phase 2 - Multi-Label | Job ${SLURM_JOB_ID} | $(date)"
# Use custom Python environment with compatible NumPy version
PYTHON_ENV="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin/python"

COOKED_DATA="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v7"

cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
${PYTHON_ENV} pipeline/00_scripts/prepare_07_phase2_multilabel.py \
    --input ${COOKED_DATA} \
    --output ${COOKED_DATA}/step_07_phase2_v7
EXIT_CODE=$?
echo "Exit: ${EXIT_CODE} | $(date)"
exit ${EXIT_CODE}
