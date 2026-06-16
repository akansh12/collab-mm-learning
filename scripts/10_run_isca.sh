#!/bin/bash
#SBATCH --job-name=mmfl_isca
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/isca_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/isca_%j.err
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu,tmp
#SBATCH --time=24:00:00

# =============================================================================
# ISCA baseline sweep — 3 datasets × 3 seeds = 9 sequential runs
# Results → experiments/isca_results/  (test_results.json per run)
#
# Submit: sbatch scripts/10_run_isca.sh
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
  echo "=== [$N/9] dataset=$1  seed=$2  $(date +%H:%M:%S) ==="
  python scripts/train_isca.py --dataset "$1" --seed $2 --exp_dir isca_results
}

for seed in 42 123 456; do
  run bge     $seed
  run sen12ms $seed
  run eurosat $seed
done

echo "=== All 9 ISCA runs complete. Results in experiments/isca_results/ ==="
'

echo "=== Job done: $(date) ==="
