#!/bin/bash
#SBATCH --job-name=step_07A_v7
#SBATCH --output=logs/step_07_phaseA_v7_%j.out
#SBATCH --error=logs/step_07_phaseA_v7_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

echo "STEP 07: Phase A - BR baseline | Job ${SLURM_JOB_ID} | $(date)"
module load Programming_Languages/anaconda/3.11
conda activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2

COOKED_DATA="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v7"

cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration

# Resolve features file (try step_03_features, then root)
if [ -f "${COOKED_DATA}/step_03_features/features_engineered.pkl" ]; then
    FEATURES_FILE="${COOKED_DATA}/step_03_features/features_engineered.pkl"
elif [ -f "${COOKED_DATA}/features_engineered.pkl" ]; then
    FEATURES_FILE="${COOKED_DATA}/features_engineered.pkl"
else
    echo "ERROR: features_engineered.pkl not found in expected locations."
    echo "Checked: ${COOKED_DATA}/step_03_features/features_engineered.pkl and ${COOKED_DATA}/features_engineered.pkl"
    echo "Consider running Step 03 to generate features: sbatch pipeline/01_slurm/submit_03_features.sh"
    exit 2
fi

python pipeline/00_scripts/phaseA_br_test.py \
    --features ${FEATURES_FILE} \
    --output ${COOKED_DATA}/step_07_phaseA_v7_br
EXIT_CODE=$?
echo "Exit: ${EXIT_CODE} | $(date)"
exit ${EXIT_CODE}
