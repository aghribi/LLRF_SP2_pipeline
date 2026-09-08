#!/bin/bash
#SBATCH --job-name=step_09_phase3b
#SBATCH --output=logs/step_09_%j.out
#SBATCH --error=logs/step_09_%j.err
#SBATCH --time=7:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=hpc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

echo "STEP 09: Phase 3B - Clustering | Job ${SLURM_JOB_ID} | $(date)"
# Use custom Python environment with compatible NumPy version
PYTHON_ENV="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin/python"

COOKED_DATA="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data"

cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
${PYTHON_ENV} pipeline/00_scripts/prepare_09_phase3b_clustering.py \
    --input ${COOKED_DATA} \
    --output ${COOKED_DATA}/step_09_phase3b
EXIT_CODE=$?
echo "Exit: ${EXIT_CODE} | $(date)"
exit ${EXIT_CODE}
