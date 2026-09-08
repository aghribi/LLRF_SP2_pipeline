#!/bin/bash
#SBATCH --job-name=reextract_corrected
#SBATCH --output=logs/reextract_corrected_%j.out
#SBATCH --error=logs/reextract_corrected_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=htc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

# ==============================================================================
# RE-EXTRACT FEATURES WITH CORRECTED PHYSICS DEFINITIONS
# ==============================================================================
# This script re-runs feature extraction after corrections made on 2026-01-25:
#
# CORRECTIONS APPLIED:
# 1. RF Mismatch Features (formerly "control error"):
#    - BEFORE: amp_error = A Ucr - A Uamp (called "control error")
#    - AFTER: rf_mismatch = A Ucr - A Uamp (correctly named as RF mismatch)
#    - V_ref does NOT exist in postmortem data (was a hallucination)
#
# 2. Modulator Command Features (NEW):
#    - Added IQ, MODP features from I/Q modulateur signals
#    - These represent actual control effort sent to RF amplifier
#
# 3. RF Power Ratio (Prefl/Pfwd) FIX:
#    - BEFORE: reflected_power = courant_pickup² (WRONG!)
#    - AFTER: reflected_power = A_Ucr² (CORRECT per thesis Table 2.2)
#
# Signal definitions per Charly Lassalle thesis Table 2.2:
#   - A Ucr (col 9): Amplitude of REFLECTED signal
#   - A Uamp (col 10): Amplifier voltage output
#   - courant pickup (col 14): Pickup current (NOT reflected power!)
#   - I/Q modulateur (cols 11-12): Modulator command signals
# ==============================================================================

echo "========================================================================"
echo "RE-EXTRACT FEATURES WITH CORRECTED PHYSICS DEFINITIONS"
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
source activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2

# Directories
PROJECT_DIR="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"
DATA_DIR="/sps/m4cast/_spiral2_data/_llrf_data/raw_data"
OUTPUT_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data"

# Navigate to project directory
cd ${PROJECT_DIR}

# Create logs directory
mkdir -p logs

# Backup existing features file
if [ -f "${OUTPUT_DIR}/features_engineered.pkl" ]; then
    BACKUP_NAME="features_engineered_backup_$(date +%Y%m%d_%H%M%S).pkl"
    echo "Backing up existing features to ${BACKUP_NAME}..."
    cp "${OUTPUT_DIR}/features_engineered.pkl" "${OUTPUT_DIR}/${BACKUP_NAME}"
fi

echo ""
echo "Starting feature re-extraction with corrected code..."
echo ""

# Run the corrected pipeline
python pipeline/00_scripts/prepare_data_cluster.py \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --n-jobs ${SLURM_CPUS_PER_TASK} \
    --file-timeout 60

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "EXTRACTION COMPLETED"
echo "========================================================================"
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS - Features re-extracted with corrections"
    echo ""
    echo "Output file:"
    ls -lh ${OUTPUT_DIR}/features_engineered.pkl
    echo ""
    echo "Next steps:"
    echo "  1. Run analysis pipeline: sbatch pipeline/01_slurm/submit_04_anomaly_baseline.sh"
    echo "  2. Run notebooks: sbatch pipeline/01_slurm/submit_notebook_phaseA.sh"
else
    echo "FAILED"
    echo ""
    echo "Check logs for errors:"
    echo "  - SLURM log: logs/reextract_corrected_${SLURM_JOB_ID}.err"
fi

echo "========================================================================"

exit ${EXIT_CODE}
