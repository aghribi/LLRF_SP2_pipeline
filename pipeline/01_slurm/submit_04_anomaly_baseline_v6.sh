#!/bin/bash
#SBATCH --job-name=step_04_v6
#SBATCH --output=logs/step_04_v6_%j.out
#SBATCH --error=logs/step_04_v6_%j.err
#SBATCH --time=00:02:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=htc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

echo "========================================================================"
echo "STEP 04: Anomaly Detection Baseline"
echo "Job ID: ${SLURM_JOB_ID}, Start: $(date)"
echo "========================================================================"

# Use custom Python environment with compatible NumPy version
PYTHON_ENV="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin/python"

PROJECT_DIR="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"
COOKED_DATA="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6"
INPUT_DIR="${COOKED_DATA}"  # features_engineered.pkl is at root level
OUTPUT_DIR="${COOKED_DATA}/step_04_v6"

mkdir -p ${PROJECT_DIR}/logs ${OUTPUT_DIR}
cd ${PROJECT_DIR}

${PYTHON_ENV} pipeline/00_scripts/prepare_04_anomaly_baseline.py --input ${INPUT_DIR} --output ${OUTPUT_DIR}
EXIT_CODE=$?

echo "Exit code: ${EXIT_CODE}, End: $(date)"
[ ${EXIT_CODE} -eq 0 ] && echo "✓ SUCCESS" || echo "✗ FAILED"
exit ${EXIT_CODE}
