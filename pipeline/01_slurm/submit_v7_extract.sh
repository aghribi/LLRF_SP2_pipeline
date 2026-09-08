#!/bin/bash
# ==============================================================================
# V7 FEATURE EXTRACTION -- FEED-FORWARD FILTER FIX (2026-09-09)
# The ground-truth classifier's dominant quality-filter check (was logged
# "Beam NON", 68% of every rejection) read the header field BEAM as particle
# beam presence; it actually encodes feed-forward state, unrelated to beam,
# and is only meaningful from 2021 onward. Classify_PostMortemFile's
# _check_filters() now scopes the check to year >= 2021 (config.yaml's
# feed_forward_min_year); the ground-truth CSV has been re-run against the
# fix, recovering several hundred legitimate 2019-2020 events. This script
# re-extracts features against that corrected ground truth. See
# prepare_data_cluster_v7.py's module docstring and
# report/sections/03_system_description.tex for the full writeup.
# ==============================================================================
# Runs prepare_data_cluster_v7.py against a dedicated output directory,
# cooked_data_v7/, kept fully separate from V2/V3/V4/V5/V6's outputs.
# ==============================================================================

set -e

PROJECT_DIR="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"
cd ${PROJECT_DIR}
mkdir -p logs

echo "=============================================================="
echo "V7 FEATURE EXTRACTION - FEED-FORWARD FILTER FIX"
echo "=============================================================="
echo "Start time: $(date)"

JOB1=$(sbatch --parsable << 'JOBSCRIPT'
#!/bin/bash
#SBATCH --job-name=01v7_extract
#SBATCH --output=logs/01v7_extract_%j.out
#SBATCH --error=logs/01v7_extract_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=htc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

module load Programming_Languages/anaconda/3.11
source activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2

cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
OUTPUT_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v7"
mkdir -p ${OUTPUT_DIR}

python pipeline/00_scripts/prepare_data_cluster_v7.py \
    --data-dir /sps/m4cast/_spiral2_data/_llrf_data/raw_data \
    --output-dir ${OUTPUT_DIR} \
    --n-jobs 4 \
    --batch-size 500 \
    --file-timeout 60
JOBSCRIPT
)
echo "  V7 feature extraction: Job ${JOB1}"
echo ""
echo "Monitor with: squeue -j ${JOB1}"
echo "Log: logs/01v7_extract_${JOB1}.out"
