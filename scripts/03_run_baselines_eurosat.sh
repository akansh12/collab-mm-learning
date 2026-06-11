#!/bin/bash
#SBATCH --job-name=mmfl_baselines_eurosat
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/baseline_eurosat_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/baseline_eurosat_%j.err
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu,tmp
#SBATCH --time=12:00:00

# =============================================================================
# EuroSAT baseline sweep — 8 configs × 3 seeds = 24 sequential runs
# Band layout: band0=VV, band1=VH, band2=R, band3=G, band4=B
# Results → experiments/baseline_eurosat/
#
# Submit: sbatch scripts/03_run_baselines_eurosat.sh
# =============================================================================

set -euo pipefail

WORKDIR="/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning"
IMAGE="projects.cispa.saarland:5005#c01akma/multi-modal-federated-learning:conda_v10"

echo "=== Job ${SLURM_JOB_ID} started: $(date) ==="
echo "=== Node: ${SLURMD_NODENAME} ==="

srun \
  --container-image="${IMAGE}" \
  --container-mounts="/home/c01akma/CISPA-az6:/home/c01akma/CISPA-az6" \
  --container-workdir="${WORKDIR}" \
  bash -c '
set -euo pipefail

N=0

run() {
  N=$((N + 1))
  echo "=== [$N/24] $1  seed=$2  $(date +%H:%M:%S) ==="
  python scripts/train.py --config "$1" --seed $2 --exp_dir baseline_eurosat --no_save
}

for seed in 42 123 456; do
  run configs/eurosat_configs/eurosat_baseline_band0.yaml $seed
  run configs/eurosat_configs/eurosat_baseline_band1.yaml $seed
  run configs/eurosat_configs/eurosat_baseline_band2.yaml $seed
  run configs/eurosat_configs/eurosat_baseline_band3.yaml $seed
  run configs/eurosat_configs/eurosat_baseline_band4.yaml $seed
  run configs/eurosat_configs/eurosat_baseline_s1.yaml    $seed
  run configs/eurosat_configs/eurosat_baseline_rgb.yaml   $seed
  run configs/eurosat_configs/eurosat_baseline_all.yaml   $seed
done

echo "=== All 24 runs complete. Results in experiments/baseline_eurosat/ ==="
'

echo "=== Job done: $(date) ==="
