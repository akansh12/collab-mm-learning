#!/bin/bash
#SBATCH --job-name=mmfl_data_size_seq
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/data_size_seq_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/data_size_seq_%j.err
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu,tmp,xe8545
#SBATCH --time=48:00:00

# =============================================================================
# BGE data-size sweep — sequential standard-BN ("w/o BN calibration" green curve)
#
# Trains joint_seq (sequential standard BN, no GroupBN, no calibration) across
# 10 dataset scales × 3 seeds = 30 runs.
# per_class_count: 100, 200, ..., 1000  →  total samples: 600, 1200, ..., 6000
# Results → experiments/data_size_seq/
#
# Submit: sbatch scripts/15_run_data_size_seq_stdbn.sh
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
TOTAL=30  # 10 scales × 3 seeds

run() {
  local pcc=$1
  local seed=$2
  N=$((N + 1))
  echo "=== [$N/$TOTAL] per_class_count=$pcc  seed=$seed  $(date +%H:%M:%S) ==="
  python scripts/train.py \
    --config configs/bge_configs/bge_joint_all_bands_seq_stdbn.yaml \
    --per_class_count $pcc \
    --seed $seed \
    --exp_dir data_size_seq
}

for pcc in 100 200 300 400 500 600 700 800 900 1000; do
  for seed in 42 123 456; do
    run $pcc $seed
  done
done

echo "=== All $TOTAL runs complete. Results in experiments/data_size_seq/ ==="
'

echo "=== Job done: $(date) ==="
