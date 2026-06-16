"""ISCA baseline training script.

Reproduces the Shared-Component-Analysis notebooks from:
  notebooks/65_ISCA_benchmark/Shared-Component-Analysis_*.ipynb

Usage:
    python scripts/train_isca.py --dataset bge   --seed 42
    python scripts/train_isca.py --dataset sen12ms --seed 123 --exp_dir isca_results
    python scripts/train_isca.py --dataset eurosat --seed 456 --no_save

Results are saved to experiments/<exp_dir>/<timestamp>_isca_<dataset>_seed<N>/test_results.json
"""

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import get_loader
from src.models.isca_networks import (
    ForeverDataIterator,
    ISCAClassifier,
    ISCADiscriminator,
    MinimumClassConfusionLoss,
)
from src.models.projections import BasicBlock, CNNProjectionLayer
from src.utils.utils import setup_seed


# ------------------------------------------------------------------ per-dataset configs

DATASET_CONFIGS = {
    "bge": {
        "modality_1": "s1", "in_channels_1": 2,
        "modality_2": "s2", "in_channels_2": 12,
        "num_classes": 6,
        "data_root": "data/bge",
    },
    "sen12ms": {
        "modality_1": "s1", "in_channels_1": 2,
        "modality_2": "s2", "in_channels_2": 13,
        "num_classes": 7,
        "data_root": "data/sen12ms",
    },
    "eurosat": {
        "modality_1": "s1", "in_channels_1": 2,
        "modality_2": "rgb", "in_channels_2": 3,
        "num_classes": 10,
        "data_root": "data/eurosat",
    },
}

# Shared ISCA hyperparameters (identical across all 3 original notebooks)
HPARAMS = dict(
    D=256,
    feature_size=512,
    alpha=2e-4,       # Z optimizer lr
    beta=2e-5,        # discriminator lr
    class_lr=0.02,
    n_epochs=200,
    lr_gamma=0.001,
    lr_decay=0.75,
    normalize=True,
    n_critic=1,
    n_z=1,
    temperature=0.55,
    lambda_dist=1.0,
    lambdaa_classify=0.1,
    lambdaa=1.0,
    lsmooth=1.0,
    batch_size=32,
    num_workers=4,
    per_class_count=100,
    num_blocks=[2, 2, 2, 2],
)


def build_loader_config(dataset_name, data_root):
    return {
        "dataset": dataset_name,
        "data_root": data_root,
        "batch_size": HPARAMS["batch_size"],
        "num_workers": HPARAMS["num_workers"],
        "per_class_count": HPARAMS["per_class_count"],
        "pin_memory": True,
        "prefetch_factor": 2,
    }


def evaluate(Z1, Z2, classifier, loader1, loader2, device):
    Z1.eval(); Z2.eval(); classifier.eval()
    all_preds1, all_targets1 = [], []
    all_preds2, all_targets2 = [], []
    with torch.no_grad():
        for (x1, y1), (x2, y2) in zip(loader1, loader2):
            c1 = Z1(x1.float().to(device))
            c2 = Z2(x2.float().to(device))
            p1 = torch.argmax(classifier(c1), dim=1).cpu()
            p2 = torch.argmax(classifier(c2), dim=1).cpu()
            all_preds1.append(p1); all_targets1.append(y1)
            all_preds2.append(p2); all_targets2.append(y2)

    p1 = torch.cat(all_preds1).numpy()
    t1 = torch.cat(all_targets1).numpy()
    p2 = torch.cat(all_preds2).numpy()
    t2 = torch.cat(all_targets2).numpy()

    acc1 = 100.0 * (p1 == t1).mean()
    acc2 = 100.0 * (p2 == t2).mean()
    f1_1 = 100.0 * f1_score(t1, p1, average="macro", zero_division=0)
    f1_2 = 100.0 * f1_score(t2, p2, average="macro", zero_division=0)
    return acc1, acc2, f1_1, f1_2


def train(args):
    ds_cfg = DATASET_CONFIGS[args.dataset]
    h = HPARAMS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_seed(args.seed)

    loader_cfg = build_loader_config(args.dataset, ds_cfg["data_root"])

    train1 = get_loader(ds_cfg["modality_1"], loader_cfg, "train", shuffle=True, sampler=None)
    train2 = get_loader(ds_cfg["modality_2"], loader_cfg, "train", shuffle=True, sampler=None)
    val1   = get_loader(ds_cfg["modality_1"], loader_cfg, "val",   shuffle=False)
    val2   = get_loader(ds_cfg["modality_2"], loader_cfg, "val",   shuffle=False)
    test1  = get_loader(ds_cfg["modality_1"], loader_cfg, "test",  shuffle=False)
    test2  = get_loader(ds_cfg["modality_2"], loader_cfg, "test",  shuffle=False)

    iters_per_epoch = len(train1.dataset) // h["batch_size"]

    # --- build models ---
    # Initialization order matches the original notebook exactly:
    # proj1 → linear1 → proj2 → linear2 (interleaved, not proj1/proj2 then linear1/linear2)
    num_blocks = h["num_blocks"]
    D, feature_size = h["D"], h["feature_size"]
    num_classes = ds_cfg["num_classes"]

    proj1 = CNNProjectionLayer(ds_cfg["in_channels_1"], BasicBlock, num_blocks, avg_pool=True, flatten=True)
    linear1 = nn.Linear(feature_size, D, bias=False)
    proj2 = CNNProjectionLayer(ds_cfg["in_channels_2"], BasicBlock, num_blocks, avg_pool=True, flatten=True)
    linear2 = nn.Linear(feature_size, D, bias=False)
    Z1 = nn.Sequential(proj1, linear1).to(device)
    Z2 = nn.Sequential(proj2, linear2).to(device)

    f = ISCADiscriminator(D).to(device)
    classifier = ISCAClassifier(D, num_classes, num_layers=1).to(device)

    optimizer_z = torch.optim.Adam(list(Z1.parameters()) + list(Z2.parameters()), lr=h["alpha"])
    optimizer_f = torch.optim.Adam(f.parameters(), lr=h["beta"])
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=h["class_lr"])
    lr_sched = LambdaLR(optimizer_cls, lambda x: h["class_lr"] * (1.0 + h["lr_gamma"] * x) ** (-h["lr_decay"]))

    mcc_loss_fn = MinimumClassConfusionLoss(h["temperature"])
    bce = nn.BCELoss()
    ce  = nn.CrossEntropyLoss()
    ID  = torch.eye(D, device=device)

    src_iter = ForeverDataIterator(train1)
    tgt_iter = ForeverDataIterator(train2)

    # --- output dir ---
    exp_name = f"isca_{args.dataset}_seed{args.seed}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = Path("experiments") / args.exp_dir / f"{timestamp}_{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    best_val_avg = 0.0
    best_state = None  # in-memory best: (Z1_sd, Z2_sd, cls_sd)
    batches_done = 0

    for epoch in range(h["n_epochs"]):
        noise_factor = 1.0 - epoch / h["n_epochs"]
        Z1.train(); Z2.train()

        for _ in range(iters_per_epoch):
            x_s, y_s = next(src_iter)
            x_t, y_t = next(tgt_iter)

            X1 = x_s.float().to(device)
            X2 = x_t.float().to(device)
            if h["normalize"]:
                X1 = X1 - X1.mean(dim=0)
                X2 = X2 - X2.mean(dim=0)
            y_s = y_s.to(device)
            y_t = y_t.to(device)
            bs = X1.shape[0]

            lbl_true  = (1.0 - h["lsmooth"] * torch.rand(bs, 1, device=device) * 0.2 * noise_factor)
            lbl_false = (       h["lsmooth"] * torch.rand(bs, 1, device=device) * 0.2 * noise_factor)

            # --- discriminator update ---
            for _ in range(h["n_critic"]):
                optimizer_f.zero_grad()
                c1, c2 = Z1(X1), Z2(X2)
                loss_f = bce(f(c1), lbl_true) + bce(f(c2), lbl_false)
                loss_f.backward()
                optimizer_f.step()

            # --- Z + classifier update ---
            for _ in range(h["n_z"]):
                optimizer_cls.zero_grad()
                optimizer_z.zero_grad()
                c1, c2 = Z1(X1), Z2(X2)

                loss_adv = bce(f(c1), lbl_false) + bce(f(c2), lbl_true)
                pred_s, pred_t = classifier(c1), classifier(c2)
                loss_cls = ce(pred_s, y_s) + ce(pred_t, y_t)

                or1 = (1 / D) * torch.norm((c1.T @ c1) / bs - ID)
                or2 = (1 / D) * torch.norm((c2.T @ c2) / bs - ID)

                loss = h["lambda_dist"] * loss_adv + h["lambdaa_classify"] * loss_cls + h["lambdaa"] * (or1 + or2)
                loss += 0.1 * (mcc_loss_fn(pred_t) + mcc_loss_fn(pred_s))
                loss.backward()
                optimizer_cls.step()
                optimizer_z.step()

            lr_sched.step()

            # --- periodic val check: always keep best state in memory ---
            # Note: matches notebook behavior — Z1/Z2 are set to eval here and NOT
            # reset to train until the next epoch's Z1.train()/Z2.train() above.
            # This means training batches after the first val check in each epoch
            # run with BN in eval mode (using frozen running stats), which is an
            # intentional reproduction of the original notebook's training protocol.
            if batches_done % 10 == 0:
                acc1, acc2, _, _ = evaluate(Z1, Z2, classifier, val1, val2, device)
                val_avg = (acc1 + acc2) / 2.0
                if val_avg > best_val_avg:
                    best_val_avg = val_avg
                    best_state = (
                        copy.deepcopy(Z1.state_dict()),
                        copy.deepcopy(Z2.state_dict()),
                        copy.deepcopy(classifier.state_dict()),
                    )
                # Do NOT reset to train here — notebook only resets at epoch start

            batches_done += 1

        print(f"Epoch {epoch+1:3d}/{h['n_epochs']}  best_val_avg={best_val_avg:.2f}%")

    # --- restore best weights for test evaluation ---
    if best_state is not None:
        Z1.load_state_dict(best_state[0])
        Z2.load_state_dict(best_state[1])
        classifier.load_state_dict(best_state[2])
        print(f"Restored best in-memory checkpoint (val_avg={best_val_avg:.2f}%)")
    else:
        print("Warning: no checkpoint saved; using final-epoch weights.")

    acc1, acc2, f1_1, f1_2 = evaluate(Z1, Z2, classifier, test1, test2, device)
    mean_acc = (acc1 + acc2) / 2.0
    mean_f1  = (f1_1 + f1_2) / 2.0
    mod1, mod2 = ds_cfg["modality_1"], ds_cfg["modality_2"]

    print(f"\n=== Test results ({args.dataset}, seed={args.seed}) ===")
    print(f"  {mod1}: acc={acc1:.2f}%  f1_macro={f1_1:.2f}%")
    print(f"  {mod2}: acc={acc2:.2f}%  f1_macro={f1_2:.2f}%")
    print(f"  mean:  acc={mean_acc:.2f}%  f1_macro={mean_f1:.2f}%")

    results = {
        mod1: {"test_accuracy": round(acc1, 4), "test_f1_macro": round(f1_1, 4)},
        mod2: {"test_accuracy": round(acc2, 4), "test_f1_macro": round(f1_2, 4)},
        "average_test_accuracy": round(mean_acc, 4),
        "average_test_f1_macro": round(mean_f1, 4),
        "seed": args.seed,
        "dataset": args.dataset,
    }

    results_path = exp_dir / "test_results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=4)
    print(f"Results saved → {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISCA baseline training")
    parser.add_argument("--dataset", choices=["bge", "sen12ms", "eurosat"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_dir", type=str, default="isca_results",
                        help="Subdirectory under experiments/ for results")
    args = parser.parse_args()
    train(args)
