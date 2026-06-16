#!/bin/bash
#SBATCH --job-name=mmfl_data_eff_paired
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/data_eff_paired_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/data_eff_paired_%j.err
#SBATCH --gres=gpu:A100:2
#SBATCH --partition=gpu,tmp
#SBATCH --time=48:00:00

# =============================================================================
# Data efficiency paired-vs-unpaired comparison — 90 runs total
#
# BGE (72 runs): 4 configs × 6 scales × 3 seeds
#   bge_baseline_s1.yaml      — unimodal S1     → experiments/data_eff_bge/
#   bge_baseline_s2.yaml      — unimodal S2     → experiments/data_eff_bge/
#   bge_data_eff_paired.yaml  — fair paired all → experiments/data_eff_bge/
#   bge_joint_s1_s2.yaml      — joint S1+S2     → experiments/data_eff_bge/
#
# EuroSAT (18 runs): fair paired only — existing uni+joint results reused
#   eurosat_data_eff_paired.yaml → experiments/data_eff_eurosat/
#
# Scales: 25 50 100 200 500 1000
# Seeds:  42 123 456
#
# Submit: sbatch scripts/18_run_data_eff_paired.sh
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
BGE_TOTAL=$((4 * 6 * 3))
EUR_TOTAL=$((1 * 6 * 3))
TOTAL=$((BGE_TOTAL + EUR_TOTAL))
N=0

run() {
  local config=$1 seed=$2 scale=$3 exp_dir=$4
  N=$((N + 1))
  echo "=== [$N/$TOTAL] $(basename $config .yaml)  scale=$scale  seed=$seed  $(date +%H:%M:%S) ==="
  python scripts/train.py \
    --config "$config" \
    --seed $seed \
    --per_class_count $scale \
    --exp_dir "$exp_dir" \
    --no_save
}

echo "--- BGE: 4 configs × 6 scales × 3 seeds = $BGE_TOTAL runs ---"
for scale in $SCALES; do
  for seed in $SEEDS; do
    run configs/bge_configs/bge_baseline_s1.yaml      $seed $scale data_eff_bge
    run configs/bge_configs/bge_baseline_s2.yaml      $seed $scale data_eff_bge
    run configs/bge_configs/bge_data_eff_paired.yaml  $seed $scale data_eff_bge
    run configs/bge_configs/bge_joint_s1_s2.yaml      $seed $scale data_eff_bge
  done
done

echo "--- EuroSAT: fair paired only — 1 config × 6 scales × 3 seeds = $EUR_TOTAL runs ---"
for scale in $SCALES; do
  for seed in $SEEDS; do
    run configs/eurosat_configs/eurosat_data_eff_paired.yaml $seed $scale data_eff_eurosat
  done
done

echo "=== All $TOTAL runs complete ==="
echo "=== BGE results   → experiments/data_eff_bge/ ==="
echo "=== EuroSAT added → experiments/data_eff_eurosat/ ==="
'

echo "=== Job done: $(date) ==="
