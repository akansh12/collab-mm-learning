#!/bin/bash
#SBATCH --job-name=mmfl_unequal
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/unequal_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/unequal_%j.err
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu,tmp
#SBATCH --time=48:00:00

# =============================================================================
# Unequal modality dataset sizes experiment
#
# Tests: joint training when the weaker modality (S1) has fewer training samples
# than the stronger modality (S2 / RGB), while the test set stays balanced.
#
# Conditions per dataset:
#   joint    S1=50/class,  S2/RGB=100/class   (50% S1)
#   joint    S1=25/class,  S2/RGB=100/class   (25% S1)
#   baseline S1=50/class   (unimodal — fair comparison baseline)
#   baseline S1=25/class   (unimodal — fair comparison baseline)
# × 3 seeds × 3 datasets (BGE, EuroSAT, SEN12MS) = 36 sequential runs
#
# Results → experiments/unequal/
# Submit: sbatch scripts/12_run_unequal_modality.sh
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

run_joint() {
  local config=$1 dataset=$2 s1_count=$3 s2_count=$4 s2_mod=$5 seed=$6
  N=$((N + 1))
  echo "=== [$N/36] joint ${dataset} S1=${s1_count} ${s2_mod}=${s2_count} seed=${seed}  $(date +%H:%M:%S) ==="
  python scripts/train.py \
    --config "$config" \
    --seed $seed \
    --exp_dir "unequal/${dataset}_s1_${s1_count}_${s2_mod}_${s2_count}" \
    --no_save \
    --mod_count_override "s1=${s1_count},${s2_mod}=${s2_count}"
}

run_baseline() {
  local config=$1 dataset=$2 count=$3 seed=$4
  N=$((N + 1))
  echo "=== [$N/36] baseline ${dataset} S1=${count}/class seed=${seed}  $(date +%H:%M:%S) ==="
  python scripts/train.py \
    --config "$config" \
    --seed $seed \
    --exp_dir "unequal/${dataset}_baseline_s1_${count}" \
    --no_save \
    --per_class_count $count
}

for seed in 42 123 456; do

  # ── BGE ──────────────────────────────────────────────────────────────────
  run_joint   configs/bge_configs/bge_joint_s1_s2.yaml       bge      50  100  s2  $seed
  run_joint   configs/bge_configs/bge_joint_s1_s2.yaml       bge      25  100  s2  $seed
  run_baseline configs/bge_configs/bge_baseline_s1.yaml      bge      50       $seed
  run_baseline configs/bge_configs/bge_baseline_s1.yaml      bge      25       $seed

  # ── EuroSAT ──────────────────────────────────────────────────────────────
  run_joint   configs/eurosat_configs/eurosat_joint_s1_rgb.yaml  eurosat  50  100  rgb  $seed
  run_joint   configs/eurosat_configs/eurosat_joint_s1_rgb.yaml  eurosat  25  100  rgb  $seed
  run_baseline configs/eurosat_configs/eurosat_baseline_s1.yaml  eurosat  50       $seed
  run_baseline configs/eurosat_configs/eurosat_baseline_s1.yaml  eurosat  25       $seed

  # ── SEN12MS ──────────────────────────────────────────────────────────────
  run_joint   configs/sen12ms_configs/sen12ms_joint_s1_s2.yaml   sen12ms  50  100  s2  $seed
  run_joint   configs/sen12ms_configs/sen12ms_joint_s1_s2.yaml   sen12ms  25  100  s2  $seed
  run_baseline configs/sen12ms_configs/sen12ms_baseline_s1.yaml  sen12ms  50       $seed
  run_baseline configs/sen12ms_configs/sen12ms_baseline_s1.yaml  sen12ms  25       $seed

done

echo "=== All 36 runs complete. Results in experiments/unequal/ ==="
'

echo "=== Job done: $(date) ==="
