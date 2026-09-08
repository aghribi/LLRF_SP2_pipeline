#!/bin/bash
#SBATCH --job-name=llrf_data_prep
#SBATCH --output=llrf_prep_%j.out
#SBATCH --error=llrf_prep_%j.err
#SBATCH --time=7:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --partition=hpc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

# SLURM Script for LLRF Data Preparation on CC-IN2P3
# ===================================================
#
# Adjust the following parameters based on your dataset size:
#
# For full dataset (14,231 files):
#   --cpus-per-task=32  (or 64 for faster processing)
#   --mem=64G           (increase to 128G if memory errors occur)
#   --time=48:00:00     (2 days should be enough)
#
# For testing (100-1000 files):
#   --cpus-per-task=8
#   --mem=16G
#   --time=2:00:00

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths (already configured for cluster)
DATA_DIR="/sps/m4cast/_spiral2_data/_llrf_data/raw_data"
OUTPUT_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data"
# VENV_DIR not needed - using conda environment

# Processing parameters
N_JOBS=${SLURM_CPUS_PER_TASK}  # Use all allocated CPUs
BATCH_SIZE="500"  # Process in batches to avoid OOM (smaller batches = faster iteration)

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "=========================================="
echo "LLRF Data Preparation - SLURM Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "=========================================="

# Load modules (adjust for CC-IN2P3)
module purge
# module load python/3.11
# module load gcc/11.2.0

# Set Python path to use conda environment directly
PYTHON_BIN="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin/python"
export PATH="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin:$PATH"

# Verify environment
echo "Python: $PYTHON_BIN"
echo "Python version: $($PYTHON_BIN --version)"
echo "Packages installed:"
$PYTHON_BIN -m pip list | grep -E "numpy|pandas|scikit-learn|scipy"

# ============================================================================
# RUN DATA PREPARATION
# ============================================================================

echo ""
echo "Starting data preparation..."
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Number of jobs: ${N_JOBS}"
echo ""

# Change to project directory
cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration

# Build command - using pipeline script with correct Python
CMD="${PYTHON_BIN} pipeline/00_scripts/prepare_data_cluster.py \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --n-jobs ${N_JOBS} \
    --batch-size ${BATCH_SIZE} \
    --file-timeout 30"

# Add resume flag (ENABLED - will skip existing batches)
CMD="$CMD --resume"

# Run the script
eval $CMD
EXIT_CODE=$?

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

# Show output files
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Output files created:"
    ls -lh ${OUTPUT_DIR}/*.pkl
    echo ""
    echo "Summary:"
    cat ${OUTPUT_DIR}/processing_summary.txt
fi

exit $EXIT_CODE
