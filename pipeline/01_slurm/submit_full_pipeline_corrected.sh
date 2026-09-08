#!/bin/bash
# ==============================================================================
# FULL PIPELINE RE-RUN WITH CORRECTED PHYSICS DEFINITIONS
# ==============================================================================
# Pipeline dependency structure:
#
#   features_engineered.pkl (Step 1: prepare_data_cluster.py)
#            |
#            v
#   +--------+--------+--------+--------+--------+--------+
#   |        |        |        |        |        |        |
#  Step04  Step05  Step06  Step07  Step08  Step09  PhaseA_BR
#   |        |        |        |        |        |        |
#   +--------+--------+--------+--------+--------+--------+
#                              |
#                              v
#                    PhaseA_SHAP (needs br_results.pkl)
#                              |
#                              v
#                    Analysis Notebooks
#
# CORRECTIONS APPLIED (2026-01-25):
# 1. V_ref hallucination fixed (renamed to RF mismatch features)
# 2. Modulator command features added (IQ, MODP)
# 3. Prefl/Pfwd bug fixed (now uses A Ucr instead of courant pickup)
#
# ==============================================================================

set -e

PROJECT_DIR="/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration"
SLURM_DIR="${PROJECT_DIR}/pipeline/01_slurm"
COOKED_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data"
SCRIPTS_DIR="${PROJECT_DIR}/pipeline/00_scripts"

cd ${PROJECT_DIR}
mkdir -p logs

echo "=============================================================="
echo "FULL PIPELINE RE-RUN - CORRECTED PHYSICS FEATURES"
echo "=============================================================="
echo "Start time: $(date)"
echo ""

# ==============================================================================
# STEP 1: Feature Extraction (all other steps depend on this)
# ==============================================================================
# MEMORY-OPTIMIZED V2 (2026-01-25):
# - Reduced n-jobs from 8 to 4 (less parallel memory)
# - Reduced batch-size from 2000 to 500 (smaller batches)
# - Using lightweight mode (no signal storage in batch files)
# - Clear old batches before starting fresh
# ==============================================================================
echo "[STEP 1] Submitting feature extraction..."

JOB1=$(sbatch --parsable << 'JOBSCRIPT'
#!/bin/bash
#SBATCH --job-name=01_extract
#SBATCH --output=logs/01_extract_%j.out
#SBATCH --error=logs/01_extract_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=htc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

module load Programming_Languages/anaconda/3.11
source activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2

cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
OUTPUT_DIR="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data"

# Backup existing features
if [ -f "${OUTPUT_DIR}/features_engineered.pkl" ]; then
    cp "${OUTPUT_DIR}/features_engineered.pkl" "${OUTPUT_DIR}/features_engineered_backup_$(date +%Y%m%d_%H%M%S).pkl"
fi

# CRITICAL: Clear old batches to ensure consistent feature engineering
# Old batches have different feature definitions and cause inconsistencies
echo "Clearing old batch files for fresh start..."
rm -rf ${OUTPUT_DIR}/batches
rm -rf ${OUTPUT_DIR}/checkpoints
mkdir -p ${OUTPUT_DIR}/batches
mkdir -p ${OUTPUT_DIR}/checkpoints

# MEMORY-OPTIMIZED: Use smaller batches, fewer workers, no signal storage
python pipeline/00_scripts/prepare_data_cluster.py \
    --data-dir /sps/m4cast/_spiral2_data/_llrf_data/raw_data \
    --output-dir ${OUTPUT_DIR} \
    --n-jobs 4 \
    --batch-size 500 \
    --file-timeout 60
JOBSCRIPT
)
echo "  Feature extraction: Job ${JOB1}"

# ==============================================================================
# STEPS 2-8: All these run IN PARALLEL after Step 1
# (They all only need features_engineered.pkl as input)
# ==============================================================================
echo ""
echo "[STEPS 2-8] Submitting parallel jobs (all depend on ${JOB1})..."

# Step 04: Anomaly Baseline
JOB_04=$(sbatch --parsable --dependency=afterok:${JOB1} << 'JOBSCRIPT'
#!/bin/bash
#SBATCH --job-name=04_anomaly
#SBATCH --output=logs/04_anomaly_%j.out
#SBATCH --error=logs/04_anomaly_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=htc

module load Programming_Languages/anaconda/3.11
source activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2
cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
COOKED="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data"
python pipeline/00_scripts/prepare_04_anomaly_baseline.py --input ${COOKED} --output ${COOKED}/step_04_anomaly
JOBSCRIPT
)
echo "  04_anomaly: Job ${JOB_04}"

# Step 05: Precursor Detection
JOB_05=$(sbatch --parsable --dependency=afterok:${JOB1} << 'JOBSCRIPT'
#!/bin/bash
#SBATCH --job-name=05_precursor
#SBATCH --output=logs/05_precursor_%j.out
#SBATCH --error=logs/05_precursor_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=htc

module load Programming_Languages/anaconda/3.11
source activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2
cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
COOKED="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data"
python pipeline/00_scripts/prepare_05_phase0_precursor.py --input ${COOKED} --output ${COOKED}/step_05_phase0
JOBSCRIPT
)
echo "  05_precursor: Job ${JOB_05}"

# Step 06: Binary Classification
JOB_06=$(sbatch --parsable --dependency=afterok:${JOB1} << 'JOBSCRIPT'
#!/bin/bash
#SBATCH --job-name=06_binary
#SBATCH --output=logs/06_binary_%j.out
#SBATCH --error=logs/06_binary_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=htc

module load Programming_Languages/anaconda/3.11
source activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2
cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
COOKED="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data"
python pipeline/00_scripts/prepare_06_phase1_binary.py --input ${COOKED} --output ${COOKED}/step_06_phase1
JOBSCRIPT
)
echo "  06_binary: Job ${JOB_06}"

# Step 07: Multi-label Classification
JOB_07=$(sbatch --parsable --dependency=afterok:${JOB1} << 'JOBSCRIPT'
#!/bin/bash
#SBATCH --job-name=07_multilabel
#SBATCH --output=logs/07_multilabel_%j.out
#SBATCH --error=logs/07_multilabel_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=htc

module load Programming_Languages/anaconda/3.11
source activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2
cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
COOKED="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data"
python pipeline/00_scripts/prepare_07_phase2_multilabel.py --input ${COOKED} --output ${COOKED}/step_07_phase2
JOBSCRIPT
)
echo "  07_multilabel: Job ${JOB_07}"

# Step 08: Root Cause Analysis
JOB_08=$(sbatch --parsable --dependency=afterok:${JOB1} << 'JOBSCRIPT'
#!/bin/bash
#SBATCH --job-name=08_rootcause
#SBATCH --output=logs/08_rootcause_%j.out
#SBATCH --error=logs/08_rootcause_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=htc

module load Programming_Languages/anaconda/3.11
source activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2
cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
COOKED="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data"
python pipeline/00_scripts/prepare_08_phase3a_rootcause.py --input ${COOKED} --output ${COOKED}/step_08_phase3a
JOBSCRIPT
)
echo "  08_rootcause: Job ${JOB_08}"

# Step 09: Clustering
JOB_09=$(sbatch --parsable --dependency=afterok:${JOB1} << 'JOBSCRIPT'
#!/bin/bash
#SBATCH --job-name=09_cluster
#SBATCH --output=logs/09_cluster_%j.out
#SBATCH --error=logs/09_cluster_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=htc

module load Programming_Languages/anaconda/3.11
source activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2
cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
COOKED="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data"
python pipeline/00_scripts/prepare_09_phase3b_clustering.py --input ${COOKED} --output ${COOKED}/step_09_phase3b
JOBSCRIPT
)
echo "  09_cluster: Job ${JOB_09}"

# Phase A: Binary Relevance (produces br_results.pkl for SHAP)
JOB_BR=$(sbatch --parsable --dependency=afterok:${JOB1} << 'JOBSCRIPT'
#!/bin/bash
#SBATCH --job-name=phaseA_br
#SBATCH --output=logs/phaseA_br_%j.out
#SBATCH --error=logs/phaseA_br_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=htc

module load Programming_Languages/anaconda/3.11
source activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2
cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
COOKED="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data"
mkdir -p ${COOKED}/phase_a_output
python pipeline/00_scripts/phaseA_br_test.py \
    --features ${COOKED}/features_engineered.pkl \
    --output ${COOKED}/phase_a_output
JOBSCRIPT
)
echo "  phaseA_br: Job ${JOB_BR}"

# ==============================================================================
# STEP 9: SHAP Analysis (depends on Phase A BR results)
# ==============================================================================
echo ""
echo "[STEP 9] Submitting SHAP analysis (depends on ${JOB_BR})..."

JOB_SHAP=$(sbatch --parsable --dependency=afterok:${JOB_BR} << 'JOBSCRIPT'
#!/bin/bash
#SBATCH --job-name=phaseA_shap
#SBATCH --output=logs/phaseA_shap_%j.out
#SBATCH --error=logs/phaseA_shap_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --partition=htc

module load Programming_Languages/anaconda/3.11
source activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2
cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
python pipeline/00_scripts/prepare_07a_phaseA_shap.py
JOBSCRIPT
)
echo "  phaseA_shap: Job ${JOB_SHAP}"

# ==============================================================================
# STEP 10: Analysis Notebooks (depends on ALL previous jobs)
# ==============================================================================
echo ""
echo "[STEP 10] Submitting analysis notebooks..."

ALL_DEPS="${JOB_04}:${JOB_05}:${JOB_06}:${JOB_07}:${JOB_08}:${JOB_09}:${JOB_SHAP}"

JOB_NB=$(sbatch --parsable --dependency=afterok:${ALL_DEPS} << 'JOBSCRIPT'
#!/bin/bash
#SBATCH --job-name=notebooks
#SBATCH --output=logs/notebooks_%j.out
#SBATCH --error=logs/notebooks_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=htc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

module load Programming_Languages/anaconda/3.11
source activate /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2
cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration

jupyter nbconvert --to notebook --execute analysis/notebooks/phaseA_shap_analysis.ipynb \
    --output phaseA_shap_analysis_executed.ipynb || true
jupyter nbconvert --to notebook --execute analysis/notebooks/phaseD_deep_analysis.ipynb \
    --output phaseD_deep_analysis_executed.ipynb || true
JOBSCRIPT
)
echo "  notebooks: Job ${JOB_NB}"

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================================================="
echo "ALL JOBS SUBMITTED"
echo "=============================================================="
echo ""
echo "Pipeline structure:"
echo "  [1] Feature extraction: ${JOB1}"
echo "  └── [Parallel after ${JOB1}]:"
echo "      ├── 04_anomaly: ${JOB_04}"
echo "      ├── 05_precursor: ${JOB_05}"
echo "      ├── 06_binary: ${JOB_06}"
echo "      ├── 07_multilabel: ${JOB_07}"
echo "      ├── 08_rootcause: ${JOB_08}"
echo "      ├── 09_cluster: ${JOB_09}"
echo "      └── phaseA_br: ${JOB_BR}"
echo "          └── phaseA_shap: ${JOB_SHAP}"
echo "              └── notebooks: ${JOB_NB}"
echo ""
echo "Monitor: squeue -u \$(whoami)"
echo "Logs: ${PROJECT_DIR}/logs/"
echo "=============================================================="
