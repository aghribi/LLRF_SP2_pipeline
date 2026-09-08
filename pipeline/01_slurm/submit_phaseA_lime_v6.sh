#!/usr/bin/env bash
#SBATCH --job-name=phaseA_lime_v6
#SBATCH --output=logs/phaseA_lime_v6.%j.out
#SBATCH --error=logs/phaseA_lime_v6.%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=htc

PYTHON_ENV="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin/python"
PROJECT_DIR="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"

export SPIRAL2_COOKED_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6"

echo "Running phaseA_lime against V6 | Job ${SLURM_JOB_ID} | $(date)"
cd ${PROJECT_DIR}
${PYTHON_ENV} ${PROJECT_DIR}/pipeline/00_scripts/prepare_07a_phaseA_lime.py
EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE} | $(date)"
[ ${EXIT_CODE} -eq 0 ] && echo "SUCCESS" || echo "FAILED"
exit ${EXIT_CODE}
