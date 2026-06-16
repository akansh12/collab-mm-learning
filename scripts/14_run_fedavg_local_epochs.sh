#!/bin/bash
#SBATCH --job-name=mmfl_fedavg_le
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/fedavg_le_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/fedavg_le_%j.err
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu,tmp,xe8545
#SBATCH --time=24:00:00

# =============================================================================
# FedAvg local-epochs sweep — BGE S1+S2 and BGE all-bands (Experiment 1 & 2)
# Reproduces figures: effect_local_epochs_bge_s1_s2_fedavg.pdf
#                     effect_local_epochs_bigearthnet_iclr_fedavg.pdf
#
# local_epochs ∈ {2, 5, 10, 25, 50} × global_rounds adjusted so
# local_epochs × global_rounds = 200 total effective epochs for all configs.
# 3 seeds each → 5 × 2 experiments × 3 seeds = 30 runs total.
# Single A100 (fedavg is single-process, no DDP).
#
# Results → experiments/fedavg_bge_s1_s2/  (S1+S2 runs)
#          experiments/fedavg_bge_all/      (all-bands runs)
#
# Submit: sbatch scripts/14_run_fedavg_local_epochs.sh
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
TOTAL=30

run() {
  N=$((N + 1))
  local CONFIG=$1 SEED=$2 EXP_DIR=$3
  echo "=== [$N/$TOTAL] $(basename $CONFIG) seed=$SEED  $(date +%H:%M:%S) ==="
  python scripts/train.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --exp_dir "$EXP_DIR"
}

# ── BGE S1+S2: 5 local_epochs × 3 seeds ─────────────────────────────────────
for LE in 2 5 10 25 50; do
  for SEED in 42 123 456; do
    run "configs/bge_configs/bge_fedavg_s1_s2_le${LE}.yaml" $SEED fedavg_bge_s1_s2
  done
done

# ── BGE all 14 bands: 5 local_epochs × 3 seeds ───────────────────────────────
for LE in 2 5 10 25 50; do
  for SEED in 42 123 456; do
    run "configs/bge_configs/bge_fedavg_all_le${LE}.yaml" $SEED fedavg_bge_all
  done
done

echo "=== All 30 runs complete. ==="
echo "    S1+S2  → experiments/fedavg_bge_s1_s2/"
echo "    all-bands → experiments/fedavg_bge_all/"
'

echo "=== Job done: $(date) ==="
