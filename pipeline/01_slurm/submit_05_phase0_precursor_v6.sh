#!/bin/bash
#SBATCH --job-name=step_05_v6
#SBATCH --output=logs/step_05_v6_%j.out
#SBATCH --error=logs/step_05_v6_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=htc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

echo "STEP 05: Phase 0 - Precursor Detection | Job ${SLURM_JOB_ID} | $(date)"
# Use custom Python environment with compatible NumPy version
PYTHON_ENV="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin/python"

COOKED_DATA="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6"

cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
${PYTHON_ENV} pipeline/00_scripts/prepare_05_phase0_precursor.py \
    --input ${COOKED_DATA} \
    --output ${COOKED_DATA}/step_05_v6
EXIT_CODE=$?
echo "Exit: ${EXIT_CODE} | $(date)"
exit ${EXIT_CODE}
