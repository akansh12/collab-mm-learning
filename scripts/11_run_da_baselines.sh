#!/bin/bash
#SBATCH --job-name=mmfl_da_baselines
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/da_baselines_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/da_baselines_%j.err
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu,tmp
#SBATCH --time=48:00:00

# =============================================================================
# DA baseline sweep — 4 methods × 3 datasets × 3 seeds = 36 sequential runs
# Methods: DANN, CDAN, MCC, MDD
# Datasets: BigEarthNet-MM (bge), SEN12MS (sen12ms), EuroSAT S1-RGB (eurosat)
# Results → experiments/da_baselines/
#
# Submit: sbatch scripts/11_run_da_baselines.sh
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
TOTAL=36

run() {
  N=$((N + 1))
  METHOD=$1; DS=$2; SEED=$3
  echo "=== [$N/$TOTAL] method=$METHOD  dataset=$DS  seed=$SEED  $(date +%H:%M:%S) ==="
  python scripts/train_da.py \
    --method "$METHOD" \
    --dataset "$DS" \
    --seed "$SEED" \
    --exp_dir da_baselines \
    --no_save
}

for seed in 42 123 456; do
  for ds in bge sen12ms eurosat; do
    run dann  $ds $seed
    run cdan  $ds $seed
    run mcc   $ds $seed
    run mdd   $ds $seed
  done
done

echo "=== All $TOTAL runs complete. Results in experiments/da_baselines/ ==="
'

echo "=== Job done: $(date) ==="
