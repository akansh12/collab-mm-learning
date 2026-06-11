#!/bin/bash
#SBATCH --job-name=mmfl_joint_all_bands_bge
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/joint_all_bands_bge_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/joint_all_bands_bge_%j.err
#SBATCH --gres=gpu:A100:4
#SBATCH --partition=gpu,tmp,xe8545
#SBATCH --time=24:00:00

# =============================================================================
# BGE joint — all 14 bands (Experiment 1) — 3 seeds sequential
# 14 modalities, world_size = min(4 GPUs, 14 mod) = 4 → 3.5 modalities per GPU
# Results → experiments/joint_bge_all_bands/
#
# Submit: sbatch scripts/05_run_joint_all_bands_bge.sh
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
    --config configs/bge_configs/bge_joint_all_bands.yaml \
    --seed $1 \
    --exp_dir joint_bge_all_bands
}

run 42
run 123
run 456

echo "=== All 3 seeds complete. Results in experiments/joint_bge_all_bands/ ==="
'

echo "=== Job done: $(date) ==="
