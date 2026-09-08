#!/bin/bash
#SBATCH --job-name=precursor_midwin
#SBATCH --output=logs/precursor_midwin_%j.out
#SBATCH --error=logs/precursor_midwin_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=htc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

echo "PRECURSOR NORMAL MID-WINDOW TEST | Job ${SLURM_JOB_ID} | $(date)"
PYTHON_ENV="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin/python"

cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration
${PYTHON_ENV} pipeline/00_scripts/investigate_precursor_normal_midwindow.py
EXIT_CODE=$?
echo "Exit: ${EXIT_CODE} | $(date)"
exit ${EXIT_CODE}
