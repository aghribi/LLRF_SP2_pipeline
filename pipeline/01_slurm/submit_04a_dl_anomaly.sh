#!/bin/bash
#SBATCH --job-name=04a_dl_anomaly
#SBATCH --partition=gpu_v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:02:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/04a_dl_anomaly_%j.out
#SBATCH --error=logs/04a_dl_anomaly_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

# ============================================================================
# Step 04a: Deep Learning Anomaly Detection
# Trains Autoencoder and VAE models for LLRF anomaly detection
# ============================================================================

echo "=========================================================================="
echo "STEP 04a: DEEP LEARNING ANOMALY DETECTION"
echo "=========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader
echo ""

# Environment setup
echo "Activating conda environment..."
module load Programming_Languages/anaconda/3.11
conda activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2

cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration

SCRIPT_PATH="pipeline/00_scripts/prepare_04a_dl_anomaly.py"

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Script: $SCRIPT_PATH"
echo ""

# Run the script
echo "=========================================================================="
echo "EXECUTING TRAINING SCRIPT"
echo "=========================================================================="
python $SCRIPT_PATH

EXIT_CODE=$?

echo ""
echo "=========================================================================="
echo "JOB COMPLETED"
echo "=========================================================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
    echo "Output: /sps/m4cast/_spiral2_data/_llrf_data/cooked_data/step_04a_dl_anomaly/"
else
    echo "Status: FAILED"
fi

exit $EXIT_CODE
