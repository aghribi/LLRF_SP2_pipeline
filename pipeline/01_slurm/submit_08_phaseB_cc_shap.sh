#!/usr/bin/env bash
# SLURM submit script to run Phase B SHAP extraction after CC model training
# Usage: sbatch --dependency=afterok:<JOBID> submit_08_phaseB_cc_shap.sh

#SBATCH --job-name=phaseB_cc_shap
#SBATCH --output=logs/phaseB_cc_shap.%j.out
#SBATCH --error=logs/phaseB_cc_shap.%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

PYTHON_ENV="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin/python"
PROJECT_DIR="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"
SCRIPT="${PROJECT_DIR}/pipeline/00_scripts/extract_shap_phaseB_cc.py"

echo "========================================================================"
echo "Phase B: SHAP Analysis for Classifier Chains"
echo "Job ID: ${SLURM_JOB_ID}, Start: $(date)"
echo "========================================================================"

cd ${PROJECT_DIR}

${PYTHON_ENV} ${SCRIPT}
EXIT_CODE=$?

echo "Exit code: ${EXIT_CODE}, End: $(date)"
[ ${EXIT_CODE} -eq 0 ] && echo "✓ SUCCESS" || echo "✗ FAILED"
exit ${EXIT_CODE}
