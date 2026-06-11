#!/bin/bash
#SBATCH --job-name=mmfl_joint_s1_s2_bge
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/joint_s1_s2_bge_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/joint_s1_s2_bge_%j.err
#SBATCH --gres=gpu:A100:2
#SBATCH --partition=gpu,tmp,xe8545
#SBATCH --time=12:00:00

# =============================================================================
# BGE joint — S1 + S2 bi-modal (Experiment 2) — 3 seeds sequential
# 2 modalities, world_size = min(2 GPUs, 2 mod) = 2 → 1 modality per GPU
# Results → experiments/joint_bge_s1_s2/
#
# Submit: sbatch scripts/06_run_joint_s1_s2_bge.sh
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
    --config configs/bge_configs/bge_joint_s1_s2.yaml \
    --seed $1 \
    --exp_dir joint_bge_s1_s2
}

run 42
run 123
run 456

echo "=== All 3 seeds complete. Results in experiments/joint_bge_s1_s2/ ==="
'

echo "=== Job done: $(date) ==="
