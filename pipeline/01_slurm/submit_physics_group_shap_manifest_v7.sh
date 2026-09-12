#!/usr/bin/env bash
# Must run AFTER submit_enhanced_shap_v7.sh completes (reads its CSV/json
# outputs from step_10_enhanced_shap/). No _v6 predecessor submit script
# existed either; modeled on the other SHAP submit scripts for consistency.
# Usage: sbatch --dependency=afterok:<enhanced_shap_v7_JOBID> submit_physics_group_shap_manifest_v7.sh
#SBATCH --job-name=physics_shap_manifest_v7
#SBATCH --output=logs/physics_shap_manifest_v7.%j.out
#SBATCH --error=logs/physics_shap_manifest_v7.%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=htc

PYTHON_ENV="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin/python"
PROJECT_DIR="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"
SCRIPT="${PROJECT_DIR}/pipeline/00_scripts/save_physics_group_shap_manifest.py"

export SPIRAL2_COOKED_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v7"

echo "Running ${SCRIPT} against V7 | Job ${SLURM_JOB_ID} | $(date)"
cd ${PROJECT_DIR}
${PYTHON_ENV} ${SCRIPT}
EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE} | $(date)"
[ ${EXIT_CODE} -eq 0 ] && echo "SUCCESS" || echo "FAILED"
exit ${EXIT_CODE}
