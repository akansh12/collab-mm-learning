#!/bin/bash
#SBATCH --job-name=mmfl_data_size_gbn
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/data_size_groupbn_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/data_size_groupbn_%j.err
#SBATCH --gres=gpu:A100:4
#SBATCH --partition=gpu,tmp,xe8545
#SBATCH --time=48:00:00

# =============================================================================
# BGE data-size sweep — GroupBN joint training (orange + yellow curves)
#
# Orange: proposed method — GroupBN training, BN calibration with TRAIN split
# Yellow: same checkpoint,  BN calibration with TEST split (leakage check)
#
# Trains bge_joint_all_bands.yaml across 10 scales × 3 seeds = 30 runs.
# Each run produces:
#   test_results.json          ← orange (train-split calibration, default)
#   test_results_bn_test.json  ← yellow (test-split calibration)
#
# 4 GPUs used: world_size = min(4, 14 modalities) = 4 (3–4 bands per GPU).
# Results → experiments/data_size_groupbn/
#
# Submit: sbatch scripts/17_run_data_size_groupbn.sh
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
  echo ""
  echo "=== [$N/$TOTAL] per_class_count=$pcc  seed=$seed  $(date +%H:%M:%S) ==="

  # --- Train + orange eval (BN calibration with train split, default) ---
  python scripts/train.py \
    --config configs/bge_configs/bge_joint_all_bands.yaml \
    --per_class_count $pcc \
    --seed $seed \
    --exp_dir data_size_groupbn
  # Saves: test_results.json (orange — train-split calibration)

  # --- Yellow eval: same checkpoint, BN calibration with test split ---
  # Find the experiment dir just created (most recent in data_size_groupbn)
  latest=""
  latest=$(ls -td experiments/data_size_groupbn/*/ 2>/dev/null | head -1) || true
  if [ -z "$latest" ]; then
    echo "ERROR: could not find experiment dir. Skipping yellow eval."
  else
    echo "  Yellow eval on: $latest"
    python scripts/train.py \
      --eval_only "$latest" \
      --bn_calib_split test \
      --eval_out test_results_bn_test.json
    # Saves: test_results_bn_test.json (yellow — test-split calibration)
    orange=$(python -c "import json; d=json.load(open(\"$latest/test_results.json\")); acc=d[\"average_test_accuracy\"]; print(f\"{acc:.4f}\")")
    yellow=$(python -c "import json; d=json.load(open(\"$latest/test_results_bn_test.json\")); acc=d[\"average_test_accuracy\"]; print(f\"{acc:.4f}\")")
    echo "  Orange (train calib): ${orange}"
    echo "  Yellow (test  calib): ${yellow}"
  fi
}

for pcc in 100 200 300 400 500 600 700 800 900 1000; do
  for seed in 42 123 456; do
    run $pcc $seed
  done
done

echo ""
echo "=== All $TOTAL runs complete. Results in experiments/data_size_groupbn/ ==="
echo "Each dir contains:"
echo "  test_results.json         (orange — proposed, train-split BN calib)"
echo "  test_results_bn_test.json (yellow — same checkpoint, test-split BN calib)"
'

echo "=== Job done: $(date) ==="
