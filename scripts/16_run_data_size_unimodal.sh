#!/bin/bash
#SBATCH --job-name=mmfl_data_size_uni
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/data_size_unimodal_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/data_size_unimodal_%j.err
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu,tmp,xe8545
#SBATCH --time=72:00:00

# =============================================================================
# BGE data-size sweep — unimodal baselines (blue curve in Figure 2)
#
# Trains each of the 14 individual bands as a unimodal classifier across
# 10 dataset scales × 3 seeds = 420 runs total.
# per_class_count: 100, 200, ..., 1000  →  total samples: 600, 1200, ..., 6000
# The blue curve is the mean test accuracy across all 14 modalities per scale.
# Results → experiments/data_size_unimodal/
#
# Submit: sbatch scripts/16_run_data_size_unimodal.sh
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
TOTAL=420  # 14 bands × 10 scales × 3 seeds

run() {
  local band=$1
  local pcc=$2
  local seed=$3
  N=$((N + 1))
  echo "=== [$N/$TOTAL] band=$band  per_class_count=$pcc  seed=$seed  $(date +%H:%M:%S) ==="
  python scripts/train.py \
    --config configs/bge_configs/bge_baseline_band${band}.yaml \
    --per_class_count $pcc \
    --seed $seed \
    --exp_dir data_size_unimodal \
    --no_save
}

for pcc in 100 200 300 400 500 600 700 800 900 1000; do
  for band in 0 1 2 3 4 5 6 7 8 9 10 11 12 13; do
    for seed in 42 123 456; do
      run $band $pcc $seed
    done
  done
done

echo "=== All $TOTAL runs complete. Results in experiments/data_size_unimodal/ ==="
'

echo "=== Job done: $(date) ==="
