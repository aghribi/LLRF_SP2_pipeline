#!/bin/bash
#SBATCH --job-name=05a_lstm_v7
#SBATCH --partition=gpu_v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/05a_lstm_v7_%j.out
#SBATCH --error=logs/05a_lstm_v7_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

# ============================================================================
# Step 05a: LSTM Event Sequence Classification
# Trains LSTM models for temporal pattern recognition in LLRF sequences
# ============================================================================

echo "=========================================================================="
echo "STEP 05a: LSTM EVENT SEQUENCE CLASSIFICATION"
echo "=========================================================================="
echo "Job ID: $SLURM_JOB_ID (V7 dataset)"
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

SCRIPT_PATH="pipeline/00_scripts/prepare_05a_event_sequence_lstm.py"

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Script: $SCRIPT_PATH"
echo ""

# Run the script
echo "=========================================================================="
echo "EXECUTING TRAINING SCRIPT"
echo "=========================================================================="
export SPIRAL2_COOKED_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v7"
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
    echo "Output: /sps/m4cast/_spiral2_data/_llrf_data/cooked_data/step_05a_event_sequence_lstm/"
else
    echo "Status: FAILED"
fi

exit $EXIT_CODE
