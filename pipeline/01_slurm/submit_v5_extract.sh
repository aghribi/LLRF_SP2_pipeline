#!/bin/bash
# ==============================================================================
# V5 FEATURE EXTRACTION -- + AMPT PHYSICAL-ONSET REFINEMENT (RF-regulation only)
# ==============================================================================
# Runs prepare_data_cluster_v5.py against a dedicated output directory,
# cooked_data_v5/, kept fully separate from V2/V3/V4's outputs.
# ==============================================================================

set -e

PROJECT_DIR="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"
cd ${PROJECT_DIR}
mkdir -p logs

echo "=============================================================="
echo "V5 FEATURE EXTRACTION - AMPT ONSET REFINEMENT"
echo "=============================================================="
echo "Start time: $(date)"

JOB1=$(sbatch --parsable << 'JOBSCRIPT'
#!/bin/bash
#SBATCH --job-name=01v5_extract
#SBATCH --output=logs/01v5_extract_%j.out
#SBATCH --error=logs/01v5_extract_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=htc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

module load Programming_Languages/anaconda/3.11
source activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2

cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
OUTPUT_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v5"
mkdir -p ${OUTPUT_DIR}

python pipeline/00_scripts/prepare_data_cluster_v5.py \
    --data-dir /sps/m4cast/_spiral2_data/_llrf_data/raw_data \
    --output-dir ${OUTPUT_DIR} \
    --n-jobs 4 \
    --batch-size 500 \
    --file-timeout 60
JOBSCRIPT
)
echo "  V5 feature extraction: Job ${JOB1}"
echo ""
echo "Monitor with: squeue -j ${JOB1}"
echo "Log: logs/01v5_extract_${JOB1}.out"
