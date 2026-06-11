#!/bin/bash
#SBATCH --job-name=mmfl_baselines_bge
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/baseline_results_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/baseline_results_%j.err
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu,tmp
#SBATCH --time=24:00:00

# =============================================================================
# BGE baseline sweep — 17 configs × 3 seeds = 51 sequential runs
# Results → experiments/baseline_bge/  (config.json + test_results.json kept; .pth deleted after eval)
#
# Submit: sbatch scripts/run_baselines_bge.sh
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
  echo "=== [$N/51] $1  seed=$2  $(date +%H:%M:%S) ==="
  python scripts/train.py --config "$1" --seed $2 --exp_dir baseline_bge --no_save
}

for seed in 42 123 456; do
  run configs/bge_configs/bge_baseline_band0.yaml  $seed
  run configs/bge_configs/bge_baseline_band1.yaml  $seed
  run configs/bge_configs/bge_baseline_band2.yaml  $seed
  run configs/bge_configs/bge_baseline_band3.yaml  $seed
  run configs/bge_configs/bge_baseline_band4.yaml  $seed
  run configs/bge_configs/bge_baseline_band5.yaml  $seed
  run configs/bge_configs/bge_baseline_band6.yaml  $seed
  run configs/bge_configs/bge_baseline_band7.yaml  $seed
  run configs/bge_configs/bge_baseline_band8.yaml  $seed
  run configs/bge_configs/bge_baseline_band9.yaml  $seed
  run configs/bge_configs/bge_baseline_band10.yaml $seed
  run configs/bge_configs/bge_baseline_band11.yaml $seed
  run configs/bge_configs/bge_baseline_band12.yaml $seed
  run configs/bge_configs/bge_baseline_band13.yaml $seed
  run configs/bge_configs/bge_baseline_all.yaml    $seed
  run configs/bge_configs/bge_baseline_s1.yaml     $seed
  run configs/bge_configs/bge_baseline_s2.yaml     $seed
done

echo "=== All 51 runs complete. Results in experiments/baseline_bge/ ==="
'

echo "=== Job done: $(date) ==="
