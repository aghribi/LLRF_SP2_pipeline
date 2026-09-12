#!/bin/bash
#SBATCH --job-name=phase3_diag_v7
#SBATCH --output=logs/phase3_diag_v7_%j.out
#SBATCH --error=logs/phase3_diag_v7_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=htc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

echo "PHASE 3: Weakness Diagnosis (V7) | Job ${SLURM_JOB_ID} | $(date)"
PYTHON_ENV="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin/python"

export SPIRAL2_COOKED_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v7"

cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
${PYTHON_ENV} pipeline/00_scripts/phase3_weakness_diagnosis.py
EXIT_CODE=$?
echo "Exit: ${EXIT_CODE} | $(date)"
exit ${EXIT_CODE}
