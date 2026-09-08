#!/bin/bash
#SBATCH --job-name=step_04_tuned
#SBATCH --output=logs/step_04_tuned_%j.out
#SBATCH --error=logs/step_04_tuned_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=htc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

echo "****************************************************************"
echo "*                      SLURM Batch System                      *"
echo "*           IN2P3 Computing Centre, Villeurbanne FR            *"
echo "****************************************************************"
echo "Date: $(date)"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Job id: ${SLURM_JOB_ID}"
echo "User: ${USER}"
echo "Account: ${SLURM_JOB_ACCOUNT}"
echo "Submit host: ${SLURM_SUBMIT_HOST}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Quality of service: ${SLURM_JOB_QOS}"
echo "Nodelist: ${SLURM_JOB_NODELIST}"
echo "Operating System: $(uname -o)"
echo "Architecture: $(uname -m)"
echo "****************************************************************"

echo "========================================================================"
echo "STEP 04: ANOMALY DETECTION (TUNED VERSION)"
echo "========================================================================"
echo "Improvements:"
echo "  - contamination = actual anomaly rate (45.6% vs default 10%)"
echo "  - DBSCAN eps computed from k-NN distances (adaptive)"
echo "  - LOF n_neighbors increased to 50 for robustness"
echo "========================================================================"
echo ""

# Load environment - use conda env python directly to avoid sklearn version conflicts
ENV_PATH="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2"
PY_BIN="$ENV_PATH/bin/python3"
source "$ENV_PATH/bin/activate" 2>/dev/null || true

echo "Python: $PY_BIN"
$PY_BIN -c "import sklearn; print(f'sklearn version: {sklearn.__version__}')"

# Navigate to project directory
cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration

# Run tuned anomaly detection
$PY_BIN pipeline/00_scripts/prepare_04_anomaly_baseline_tuned.py

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "STEP 04 (TUNED) COMPLETED"
echo "========================================================================"
echo "Exit: ${EXIT_CODE} | $(date)"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ SUCCESS"
    echo ""
    echo "Output: /sps/m4cast/_spiral2_data/_llrf_data/cooked_data/step_04_anomaly_tuned/"
    echo "Next: Compare tuned vs baseline performance in analysis notebook"
else
    echo "✗ FAILED"
    echo "Check logs for errors"
fi

echo "****************************************************************"
echo "*                      SLURM Batch System                      *"
echo "*           IN2P3 Computing Centre, Villeurbanne FR            *"
echo "****************************************************************"
echo "Date: $(date)"
echo "Job informations can be found using these commands:"
echo "Accounting:"
echo "sacct -j ${SLURM_JOB_ID}"
echo "Efficiency:"
echo "seff ${SLURM_JOB_ID}"
echo "****************************************************************"

exit ${EXIT_CODE}
