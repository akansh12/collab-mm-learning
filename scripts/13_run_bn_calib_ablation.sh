#!/bin/bash
#SBATCH --job-name=mmfl_bn_ablation
#SBATCH --output=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/bn_ablation_%j.out
#SBATCH --error=/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning/logs/bn_ablation_%j.err
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu,tmp
#SBATCH --time=04:00:00

# =============================================================================
# BN calibration ablation: no-BN vs BN(train) vs BN(test)
#
# Uses bge_bn_ablation_s1_s2.yaml — same as BGE joint S1+S2 but with standard
# BatchNorm instead of GroupBN. Trains from scratch × 3 seeds, then re-evaluates
# each checkpoint under 3 BN conditions without retraining.
#
# Output per seed dir: test_results_bn_{none,train,test}.json
# → build 3-column table for rebuttal (7XmS R2)
#
# Submit: sbatch scripts/13_run_bn_calib_ablation.sh
# TODO after run: parse the 3 JSON files per seed, compute mean±std per condition
# =============================================================================

set -euo pipefail

WORKDIR="/home/c01akma/CISPA-az6/mmfl-2025/collab-mm-learning"
IMAGE="projects.cispa.saarland:5005#c01akma/multi-modal-federated-learning:conda_v10"
CONFIG="configs/bge_configs/bge_bn_ablation_s1_s2.yaml"

echo "=== Job ${SLURM_JOB_ID} started: $(date) ==="
echo "=== Node: ${SLURMD_NODENAME} ==="

srun \
  --container-image="${IMAGE}" \
  --container-mounts="/home/c01akma/CISPA-az6:/home/c01akma/CISPA-az6" \
  --container-workdir="${WORKDIR}" \
  bash -c '
set -euo pipefail

CONFIG="configs/bge_configs/bge_bn_ablation_s1_s2.yaml"
N=0
TOTAL=9  # 3 seeds × 3 eval conditions

# --- Step 1: train × 3 seeds ---
echo "=== PHASE 1: Training (3 seeds) ==="
for seed in 42 123 456; do
  echo "=== Training seed=$seed  $(date +%H:%M:%S) ==="
  python scripts/train.py \
    --config "$CONFIG" \
    --seed $seed \
    --exp_dir bn_calib_ablation
done

# --- Step 2: evaluate each checkpoint × 3 BN conditions ---
echo ""
echo "=== PHASE 2: BN ablation evaluation (3 seeds × 3 conditions) ==="

for exp_dir in experiments/bn_calib_ablation/*/; do
  for cond in none train test; do
    N=$((N + 1))
    echo "=== [$N/$TOTAL] $(basename $exp_dir)  BN=$cond  $(date +%H:%M:%S) ==="
    python -c "
import json, sys, os, torch
sys.path.insert(0, \".\")
from src.training.testing import evaluate_test_models
exp, cond = sys.argv[1], sys.argv[2]
with open(os.path.join(exp, \"config.json\")) as f:
    cfg = json.load(f)
cfg[\"experiment_name\"] = exp
if cond == \"none\":
    cfg[\"recompute_batchnorm\"] = False
else:
    cfg[\"recompute_batchnorm\"] = True
    cfg[\"bn_calib_split\"] = cond
device = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")
results, avg = evaluate_test_models(cfg, device)
results[\"average_test_accuracy\"] = avg
out = os.path.join(exp, f\"test_results_bn_{cond}.json\")
with open(out, \"w\") as f:
    json.dump(results, f, indent=4)
print(f\"  avg={avg:.4f}  saved {out}\")
" "$exp_dir" "$cond"
  done
done

echo ""
echo "=== All done. Results in experiments/bn_calib_ablation/ ==="
echo "Each seed dir contains: test_results_bn_{none,train,test}.json"
echo "TODO: parse these and build the 3-column comparison table."
'

echo "=== Job done: $(date) ==="
