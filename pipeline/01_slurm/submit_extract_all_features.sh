#!/bin/bash
#SBATCH --job-name=extract_all_features
#SBATCH --output=logs/extract_all_features_%j.out
#SBATCH --error=logs/extract_all_features_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=htc_highmem
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

# ==============================================================================
# Extract Features from ALL 29 Batches - High Memory Job
# ==============================================================================
# Description: Processes all 29 batch files one-by-one to extract features
#              Handles very large batches (up to 111GB) with 256GB memory
#
# Input:  /sps/m4cast/_spiral2_data/_llrf_data/cooked_data/batches/batch_*.pkl
# Output: /sps/m4cast/_spiral2_data/_llrf_data/cooked_data/features_engineered.pkl
#
# Compute requirements:
#   - CPUs: 4
#   - Memory: 256 GB (to handle 111GB batch_017)
#   - Time: ~12 hours (29 batches, some very large)
# ==============================================================================

echo "========================================================================"
echo "EXTRACT FEATURES FROM ALL 29 BATCHES"
echo "========================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Running on: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE} MB"
echo "========================================================================"
echo ""

# Load environment
module load Programming_Languages/anaconda/3.11
conda activate anomalies

# Directories
PROJECT_DIR="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"
BATCHES_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data/batches"
OUTPUT_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data"

# Navigate to project directory
cd ${PROJECT_DIR}

# Create logs directory
mkdir -p logs

# Check batch files
BATCH_COUNT=$(find ${BATCHES_DIR} -name "batch_*.pkl" | wc -l)
echo "Found ${BATCH_COUNT} batch files in ${BATCHES_DIR}"
echo ""

# Show batch sizes
echo "Batch file sizes:"
ls -lh ${BATCHES_DIR}/batch_*.pkl | awk '{print $5, $9}' | tail -10
echo ""

# Run extraction
echo "Starting one-by-one feature extraction..."
echo ""

python load_features_onebyone.py

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "EXTRACTION COMPLETED"
echo "========================================================================"
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ SUCCESS"
    echo ""
    echo "Output file:"
    ls -lh ${OUTPUT_DIR}/features_engineered.pkl
    echo ""
    echo "Next step: Re-run analysis pipeline (steps 04-10)"
    echo "  sbatch cluster_slurm/submit_04_anomaly_baseline.sh"
else
    echo "✗ FAILED"
    echo ""
    echo "Check logs for errors:"
    echo "  - SLURM log: logs/extract_all_features_${SLURM_JOB_ID}.err"
fi

echo "========================================================================"

exit ${EXIT_CODE}
