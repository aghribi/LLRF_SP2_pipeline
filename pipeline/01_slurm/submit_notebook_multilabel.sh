#!/bin/bash
#SBATCH --job-name=nb_multilabel
#SBATCH --output=logs/nb_multilabel_%j.out
#SBATCH --error=logs/nb_multilabel_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=htc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

echo "========================================================================"
echo "Multilabel Phases Comparison Notebook"
echo "Job ID: ${SLURM_JOB_ID}, Start: $(date)"
echo "========================================================================"

PYTHON_ENV="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin/python"
PROJECT_DIR="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"
SCRIPT="${PROJECT_DIR}/analysis/notebooks/run_multilabel_phases.py"

cd ${PROJECT_DIR}
mkdir -p analysis/outputs/multilabel_comparison

${PYTHON_ENV} ${SCRIPT}
EXIT_CODE=$?

echo "Exit code: ${EXIT_CODE}, End: $(date)"
[ ${EXIT_CODE} -eq 0 ] && echo "✓ SUCCESS" || echo "✗ FAILED"
exit ${EXIT_CODE}
