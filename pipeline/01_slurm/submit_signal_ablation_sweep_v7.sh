#!/bin/bash
#SBATCH --job-name=signal_ablation_v7
#SBATCH --output=logs/signal_ablation_v7_%j.out
#SBATCH --error=logs/signal_ablation_v7_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=htc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adnan.ghribi@ganil.fr

echo "SIGNAL ABLATION SWEEP | Job ${SLURM_JOB_ID} | $(date)"
PYTHON_ENV="/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin/python"
COOKED="/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v7"
OUT="${COOKED}/step_signal_ablation"
SCRIPT="pipeline/00_scripts/investigate_category_signal_ablation.py"

cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration

run_one() {
  local category="$1"
  local exclude="$2"
  local tag="$3"
  echo "--- ${tag} ---"
  ${PYTHON_ENV} ${SCRIPT} --input ${COOKED} --category "${category}" --exclude-channels "${exclude}" --tag "${tag}" --output ${OUT}
}

# --- 5 categories with a specific defining signal ---
run_one 'Dép seuil de sécurité RF' '' rfsafety_baseline
run_one 'Dép seuil de sécurité RF' 'Ucav,A Ucr' rfsafety_excl_UcavAUcr

run_one 'Seuil de vide' '' vacuum_baseline
run_one 'Seuil de vide' 'vide,courant pickup' vacuum_excl_videPickup

run_one 'Rég signal RF hors tolérance' '' ecavinstab_baseline
run_one 'Rég signal RF hors tolérance' 'Uci,A Ucr,Ucav,IQ' ecavinstab_excl_UciAUcrUcavIQ

run_one 'Seuil pick-up' '' pickupthresh_baseline
run_one 'Seuil pick-up' 'courant pickup' pickupthresh_excl_pickup

# --- 2 categories with no single defining channel: full per-channel sweep ---
CHANNELS=(
  'I Ucav binary' 'Q Ucav binary' 'I Fref binary' 'Q Fref binary' 'I Uci binary' 'Q Uci binary'
  'A Ucr binary' 'A Uamp binary' 'I modulateur binary' 'Q modulateur binary' 'vide binary'
  'courant pickup binary' 'frequence DSTF binary' 'commandes binary' 'A Ucr' 'A Uamp'
  'I modulateur' 'Q modulateur' 'vide' 'courant pickup' 'Ucav' 'Uci' 'PhaseCav' 'PhaseUci'
  'DPCI' 'IQ' 'MODP'
)

run_one 'Coupure externe rapide' '' extcutoff_baseline
i=0
for ch in "${CHANNELS[@]}"; do
  slug=$(echo "$ch" | tr ' ' '_')
  run_one 'Coupure externe rapide' "$ch" "extcutoff_excl_${slug}"
  i=$((i+1))
done

run_one 'Absence autorisation RF' '' rfauth_baseline
for ch in "${CHANNELS[@]}"; do
  slug=$(echo "$ch" | tr ' ' '_')
  run_one 'Absence autorisation RF' "$ch" "rfauth_excl_${slug}"
done

echo "ALL RUNS COMPLETE | $(date)"
