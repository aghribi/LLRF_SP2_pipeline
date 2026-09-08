#!/bin/bash
# ==============================================================================
# STEP 1 ONLY: Feature Extraction (Memory-Optimized V2)
# ==============================================================================
# Submit this first, then monitor with: squeue -u $(whoami)
# Check logs with: tail -f logs/01_extract_*.out
# After completion, run: ./submit_remaining_steps.sh <JOB1_ID>
# ==============================================================================

set -e

PROJECT_DIR="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"
cd ${PROJECT_DIR}
mkdir -p logs

echo "=============================================================="
echo "STEP 1: Feature Extraction (Memory-Optimized)"
echo "=============================================================="
echo "Start time: $(date)"
echo ""

JOB1=$(sbatch --parsable << 'JOBSCRIPT'
#!/bin/bash
#SBATCH --job-name=01_extract
#SBATCH --output=logs/01_extract_%j.out
#SBATCH --error=logs/01_extract_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=htc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

module load Programming_Languages/anaconda/3.11
source activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2

cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
OUTPUT_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data"

echo "=============================================================="
echo "MEMORY-OPTIMIZED DATA EXTRACTION V2"
echo "=============================================================="
echo "Configuration:"
echo "  - n-jobs: 4 (reduced from 8)"
echo "  - batch-size: 500 (reduced from 2000)"
echo "  - store-signals: NO (lightweight mode)"
echo "=============================================================="

# Backup existing features
if [ -f "${OUTPUT_DIR}/features_engineered.pkl" ]; then
    BACKUP_FILE="${OUTPUT_DIR}/features_engineered_backup_$(date +%Y%m%d_%H%M%S).pkl"
    cp "${OUTPUT_DIR}/features_engineered.pkl" "${BACKUP_FILE}"
    echo "Backed up existing features to: ${BACKUP_FILE}"
fi

# CRITICAL: Clear old batches for consistent feature engineering
echo "Clearing old batch files..."
rm -rf ${OUTPUT_DIR}/batches
rm -rf ${OUTPUT_DIR}/checkpoints
mkdir -p ${OUTPUT_DIR}/batches
mkdir -p ${OUTPUT_DIR}/checkpoints

# Run memory-optimized extraction
python pipeline/00_scripts/prepare_data_cluster.py \
    --data-dir /sps/m4cast/_spiral2_data/_llrf_data/raw_data \
    --output-dir ${OUTPUT_DIR} \
    --n-jobs 4 \
    --batch-size 500 \
    --file-timeout 60

echo "=============================================================="
echo "Feature extraction completed!"
echo "=============================================================="
JOBSCRIPT
)

echo "Submitted Job ID: ${JOB1}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$(whoami)"
echo "  tail -f ${PROJECT_DIR}/logs/01_extract_${JOB1}.out"
echo ""
echo "After completion, check results with:"
echo "  ls -lh /sps/m4cast/_spiral2_data/_llrf_data/cooked_data/*.pkl"
echo ""
echo "Then run remaining steps:"
echo "  ./submit_remaining_steps.sh ${JOB1}"
echo "=============================================================="
