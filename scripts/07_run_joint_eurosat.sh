#!/bin/bash
#SBATCH --job-name=mmfl_joint_eurosat
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/joint_eurosat_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/joint_eurosat_%j.err
#SBATCH --gres=gpu:A100:2
#SBATCH --partition=gpu,tmp,xe8545
#SBATCH --time=24:00:00

# =============================================================================
# EuroSAT joint — all 5 bands + S1+RGB bi-modal — 3 seeds each = 6 runs total
# Both use backbone_blocks=[2], proj_blocks=[2,2,2], world_size=min(2,n_mod)
#   joint_all:    5 mod → world_size=2, exp_dir=joint_eurosat_all_bands
#   joint_s1_rgb: 2 mod → world_size=2, exp_dir=joint_eurosat_s1_rgb
#
# Submit: sbatch scripts/07_run_joint_eurosat.sh
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
  echo "=== [$N/6] $1  seed=$2  $(date +%H:%M:%S) ==="
  python scripts/train.py --config "$1" --seed $2 --exp_dir "$3"
}

echo "--- EuroSAT joint all bands ---"
for seed in 42 123 456; do
  run configs/eurosat_configs/eurosat_joint_all.yaml $seed joint_eurosat_all_bands
done

echo "--- EuroSAT joint S1+RGB ---"
for seed in 42 123 456; do
  run configs/eurosat_configs/eurosat_joint_s1_rgb.yaml $seed joint_eurosat_s1_rgb
done

echo "=== All 6 runs complete. ==="
echo "  joint_all  → experiments/joint_eurosat_all_bands/"
echo "  joint_s1_rgb → experiments/joint_eurosat_s1_rgb/"
'

echo "=== Job done: $(date) ==="
