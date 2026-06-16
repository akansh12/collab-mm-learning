#!/bin/bash
#SBATCH --job-name=mmfl_data_eff_eurosat
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/data_eff_eurosat_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/data_eff_eurosat_%j.err
#SBATCH --gres=gpu:A100:2
#SBATCH --partition=gpu,tmp
#SBATCH --time=48:00:00

# =============================================================================
# EuroSAT data efficiency sweep — 4 configs × 6 scales × 3 seeds = 72 runs
#
# Configs:
#   eurosat_baseline_s1.yaml   — unimodal S1 baseline
#   eurosat_baseline_rgb.yaml  — unimodal RGB baseline
#   eurosat_baseline_all.yaml  — early fusion (paired upper bound)
#   eurosat_joint_s1_rgb.yaml  — proposed method (uses both GPUs for DDP)
#
# Scales (per-class): 25 50 100 200 500 1000
# Results → experiments/data_eff_eurosat/
#
# Submit: sbatch scripts/12_run_data_efficiency_eurosat.sh
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

SCALES="25 50 100 200 500 1000"
SEEDS="42 123 456"
TOTAL=$((4 * 6 * 3))
N=0

run() {
  local config=$1 seed=$2 scale=$3
  N=$((N + 1))
  echo "=== [$N/$TOTAL] $(basename $config .yaml)  scale=$scale  seed=$seed  $(date +%H:%M:%S) ==="
  python scripts/train.py \
    --config "$config" \
    --seed $seed \
    --per_class_count $scale \
    --exp_dir data_eff_eurosat \
    --no_save
}

for scale in $SCALES; do
  for seed in $SEEDS; do
    run configs/eurosat_configs/eurosat_baseline_s1.yaml  $seed $scale
    run configs/eurosat_configs/eurosat_baseline_rgb.yaml $seed $scale
    run configs/eurosat_configs/eurosat_baseline_all.yaml $seed $scale
    run configs/eurosat_configs/eurosat_joint_s1_rgb.yaml $seed $scale
  done
done

echo "=== All $TOTAL runs complete. Results in experiments/data_eff_eurosat/ ==="
'

echo "=== Job done: $(date) ==="
