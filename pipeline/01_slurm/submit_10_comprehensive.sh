#!/bin/bash
#SBATCH --job-name=step_10_comprehensive
#SBATCH --output=logs/step_10_%j.out
#SBATCH --error=logs/step_10_%j.err
#SBATCH --time=7:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=hpc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

echo "STEP 10: Comprehensive Analysis | Job ${SLURM_JOB_ID} | $(date)"
module load Programming_Languages/anaconda/3.11
conda activate anomalies
cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
python pipeline/00_scripts/prepare_10_comprehensive.py \
    --base-dir /sps/m4cast/_spiral2_data/_llrf_data/cooked_data \
    --output /sps/m4cast/_spiral2_data/_llrf_data/cooked_data/step_10_comprehensive
EXIT_CODE=$?
echo "Exit: ${EXIT_CODE} | $(date)"
exit ${EXIT_CODE}
