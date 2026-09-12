#!/usr/bin/env bash
#SBATCH --job-name=saliency_transformer_v7
#SBATCH --output=logs/saliency_transformer_v7.%j.out
#SBATCH --error=logs/saliency_transformer_v7.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=htc

PYTHON_ENV="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin/python"
PROJECT_DIR="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"

export SPIRAL2_COOKED_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v7"

echo "Running saliency_transformer against V7 | Job ${SLURM_JOB_ID} | $(date)"
cd ${PROJECT_DIR}
${PYTHON_ENV} ${PROJECT_DIR}/pipeline/00_scripts/extract_saliency_phaseD.py --arch transformer
EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE} | $(date)"
[ ${EXIT_CODE} -eq 0 ] && echo "SUCCESS" || echo "FAILED"
exit ${EXIT_CODE}
