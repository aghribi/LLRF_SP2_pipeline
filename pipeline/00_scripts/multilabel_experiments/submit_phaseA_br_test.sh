#!/bin/bash
#SBATCH --job-name=phaseA_br_test
#SBATCH --output=phaseA_br_test_%j.out
#SBATCH --error=phaseA_br_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=short

# Adjust these paths
FEATURES_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data/step_03_features"
OUTPUT_DIR="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration/pipeline/00_scripts/multilabel_experiments/output_phaseA"

module purge
module load Programming_Languages/anaconda/3.11
conda activate anomalies

cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration

python pipeline/00_scripts/multilabel_experiments/phaseA_br_test.py \
  --features ${FEATURES_DIR}/features_engineered.pkl \
  --output ${OUTPUT_DIR} \
  --max-samples 2000

