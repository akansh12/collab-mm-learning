#!/bin/bash
#SBATCH --job-name=mmfl_joint_sen12ms_all
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/joint_sen12ms_all_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/joint_sen12ms_all_%j.err
#SBATCH --gres=gpu:A100:4
#SBATCH --partition=gpu,tmp,xe8545
#SBATCH --time=24:00:00

# =============================================================================
# SEN12MS joint — all 15 bands (Experiment 1) — 3 seeds sequential
# 15 modalities, backbone_blocks=[2,2,2], proj_blocks=[2]
# world_size = min(4 GPUs, 15 mod) = 4 → ~4 modalities per GPU
# Results → experiments/joint_sen12ms_all_bands/
#
# Submit: sbatch scripts/08_run_joint_sen12ms_all.sh
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
  echo "=== [$N/3] seed=$1  $(date +%H:%M:%S) ==="
  python scripts/train.py \
    --config configs/sen12ms_configs/sen12ms_joint_all.yaml \
    --seed $1 \
    --exp_dir joint_sen12ms_all_bands
}

run 42
run 123
run 456

echo "=== All 3 seeds complete. Results in experiments/joint_sen12ms_all_bands/ ==="
'

echo "=== Job done: $(date) ==="
