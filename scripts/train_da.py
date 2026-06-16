"""
Unified training script for domain-adaptation baselines: DANN, CDAN, MCC, MDD.

Usage:
    python scripts/train_da.py --method dann  --dataset bge     --seed 42
    python scripts/train_da.py --method cdan  --dataset sen12ms --seed 123
    python scripts/train_da.py --method mdd   --dataset eurosat --seed 456 --exp_dir da_baselines

Results saved to experiments/{exp_dir}/YYYYMMDD-HHMMSS_{method}_{dataset}_seed{N}/
  config.json      — hyperparameters
  test_results.json — {modality_s1, modality_{s2|rgb}} accuracy + F1-macro, average_test_accuracy
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

# Allow imports from repo root (tllib, src)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter, ProgressMeter

from src.data import bigearthnet, eurosat, sen12ms
from src.models.projections import CNNProjectionLayer, BasicBlock

# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

_DS_CFG = {
    "bge": {
        "data_root": "data/bge",
        "mod1": "s1", "mod1_ch": 2,
        "mod2": "s2", "mod2_ch": 12,
        "num_classes": 6,
        "get_transform_fn": bigearthnet.get_transform,
        "Dataset": bigearthnet.BigEarthNetFast,
        "file_ext": "pth",
    },
    "sen12ms": {
        "data_root": "data/sen12ms",
        "mod1": "s1", "mod1_ch": 2,
        "mod2": "s2", "mod2_ch": 13,
        "num_classes": 7,
        "get_transform_fn": sen12ms.get_transform,
        "Dataset": sen12ms.SEN12MSFast,
        "file_ext": "pt",
    },
    "eurosat": {
        "data_root": "data/eurosat",
        "mod1": "s1", "mod1_ch": 2,
        "mod2": "rgb", "mod2_ch": 3,
        "num_classes": 10,
        "get_transform_fn": eurosat.get_transform,
        "Dataset": eurosat.EuroSATFast,
        "file_ext": "pth",
    },
}


def _make_loader(dataset_name, bands, split, data_root, per_class_count, batch_size):
    cfg = _DS_CFG[dataset_name]
    transform = cfg["get_transform_fn"](bands, split, data_root)
    ds = cfg["Dataset"](bands, transform, per_class_count, split, data_root)
    shuffle = (split == "train")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=4, pin_memory=True, drop_last=(split == "train"))


def get_loaders(dataset_name, batch_size=32, per_class_count=100):
    cfg = _DS_CFG[dataset_name]
    data_root = cfg["data_root"]
    m1, m2 = cfg["mod1"], cfg["mod2"]

    loaders = {}
    for mod in (m1, m2):
        for split in ("train", "val", "test"):
            loaders[f"{mod}_{split}"] = _make_loader(
                dataset_name, mod, split, data_root, per_class_count, batch_size
            )
    return loaders, cfg["num_classes"], m1, m2


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_backbones(mod1_ch, mod2_ch, device):
    num_blocks = [2, 2, 2, 2]
    m1 = CNNProjectionLayer(mod1_ch, BasicBlock, num_blocks, avg_pool=True, flatten=True).to(device)
    m2 = CNNProjectionLayer(mod2_ch, BasicBlock, num_blocks, avg_pool=True, flatten=True).to(device)
    return m1, m2


def make_shared_classifier(num_classes, device):
    return nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    ).to(device)


# MDD custom module (from notebook — two-head architecture with GRL)
from tllib.alignment.mdd import WarmStartGradientReverseLayer


class MDDModule(nn.Module):
    """Two-backbone, two-head module for MDD (adapted from notebook GeneralModule)."""

    def __init__(self, backbone_1, backbone_2, num_classes, bottleneck_dim=256, width=256):
        super().__init__()
        self.backbone_1 = backbone_1
        self.backbone_2 = backbone_2
        self.num_classes = num_classes

        self.bottleneck_1 = nn.Sequential(
            nn.Linear(512, bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)
        )
        self.bottleneck_2 = nn.Sequential(
            nn.Linear(512, bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)
        )
        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5), nn.Linear(width, num_classes)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5), nn.Linear(width, num_classes)
        )
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False)

    def forward(self, x1, x2):
        f1 = self.bottleneck_1(self.backbone_1(x1))
        f2 = self.bottleneck_2(self.backbone_2(x2))
        features = torch.cat((f1, f2), dim=0)
        outputs = self.head(features)
        if self.training:
            features_adv = self.grl(features)
            outputs_adv = self.adv_head(features_adv)
            return outputs, outputs_adv
        return outputs

    def step(self):
        self.grl.step()

    def get_parameters(self, base_lr=1.0):
        return [
            {"params": self.backbone_1.parameters(), "lr": 0.1 * base_lr},
            {"params": self.backbone_2.parameters(), "lr": 0.1 * base_lr},
            {"params": self.bottleneck_1.parameters(), "lr": base_lr},
            {"params": self.bottleneck_2.parameters(), "lr": base_lr},
            {"params": self.head.parameters(), "lr": base_lr},
            {"params": self.adv_head.parameters(), "lr": base_lr},
        ]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_two_mod(mod1_model, mod2_model, classifier, loader1, loader2, device):
    """Returns (acc_m1, acc_m2, f1_m1, f1_m2)."""
    for m in (mod1_model, mod2_model, classifier):
        m.eval()

    preds1, tgts1, preds2, tgts2 = [], [], [], []
    with torch.no_grad():
        for (x1, y1), (x2, y2) in zip(loader1, loader2):
            x1, x2 = x1.to(device).float(), x2.to(device).float()
            p1 = classifier(mod1_model(x1)).argmax(1).cpu()
            p2 = classifier(mod2_model(x2)).argmax(1).cpu()
            preds1.append(p1); tgts1.append(y1)
            preds2.append(p2); tgts2.append(y2)

    p1 = torch.cat(preds1); t1 = torch.cat(tgts1)
    p2 = torch.cat(preds2); t2 = torch.cat(tgts2)
    acc1 = (p1 == t1).float().mean().item()
    acc2 = (p2 == t2).float().mean().item()
    f1_1 = f1_score(t1.numpy(), p1.numpy(), average="macro")
    f1_2 = f1_score(t2.numpy(), p2.numpy(), average="macro")
    return acc1, acc2, f1_1, f1_2


def _eval_mdd(classifier, loader1, loader2, device):
    classifier.eval()
    preds1, tgts1, preds2, tgts2 = [], [], [], []
    with torch.no_grad():
        for (x1, y1), (x2, y2) in zip(loader1, loader2):
            x1, x2 = x1.to(device).float(), x2.to(device).float()
            out = classifier(x1, x2)
            o1, o2 = out.chunk(2, dim=0)
            preds1.append(o1.argmax(1).cpu()); tgts1.append(y1)
            preds2.append(o2.argmax(1).cpu()); tgts2.append(y2)

    p1 = torch.cat(preds1); t1 = torch.cat(tgts1)
    p2 = torch.cat(preds2); t2 = torch.cat(tgts2)
    acc1 = (p1 == t1).float().mean().item()
    acc2 = (p2 == t2).float().mean().item()
    f1_1 = f1_score(t1.numpy(), p1.numpy(), average="macro")
    f1_2 = f1_score(t2.numpy(), p2.numpy(), average="macro")
    return acc1, acc2, f1_1, f1_2


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_dann(model_s1, model_s2, classifier, domain_adv, optimizer, lr_scheduler,
               train_iter1, train_iter2, val_loader1, val_loader2, iters_per_epoch,
               num_epochs, device, trade_off, save_dir):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model_s1.train(); model_s2.train(); classifier.train(); domain_adv.train()
        losses = AverageMeter("Loss", ":6.3f")
        progress = ProgressMeter(iters_per_epoch, [losses], prefix=f"[DANN] Epoch {epoch}")

        for i in range(iters_per_epoch):
            x_s, y_s = next(train_iter1)
            x_t, y_t = next(train_iter2)
            x_s, y_s = x_s.to(device), y_s.to(device)
            x_t, y_t = x_t.to(device), y_t.to(device)

            f_s = model_s1(x_s)
            f_t = model_s2(x_t)
            y_s_pred = classifier(f_s)
            y_t_pred = classifier(f_t)

            cls_loss = F.cross_entropy(y_s_pred, y_s) + F.cross_entropy(y_t_pred, y_t)
            transfer_loss = domain_adv(f_s, f_t)
            loss = cls_loss + trade_off * transfer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            losses.update(loss.item(), x_s.size(0))
            if i % 50 == 0:
                progress.display(i)

        acc1, acc2, _, _ = _eval_two_mod(model_s1, model_s2, classifier, val_loader1, val_loader2, device)
        val_acc = (acc1 + acc2) / 2.0
        print(f"  Val acc: {val_acc*100:.2f}%  (m1={acc1*100:.2f}  m2={acc2*100:.2f})")
        if val_acc > best_acc:
            best_acc = val_acc
            if save_dir:
                torch.save(model_s1.state_dict(), os.path.join(save_dir, "best_m1.pth"))
                torch.save(model_s2.state_dict(), os.path.join(save_dir, "best_m2.pth"))
                torch.save(classifier.state_dict(), os.path.join(save_dir, "best_cls.pth"))

    return best_acc


def train_cdan(model_s1, model_s2, classifier, domain_adv, optimizer, lr_scheduler,
               train_iter1, train_iter2, val_loader1, val_loader2, iters_per_epoch,
               num_epochs, device, trade_off, save_dir):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model_s1.train(); model_s2.train(); classifier.train(); domain_adv.train()
        losses = AverageMeter("Loss", ":6.3f")
        progress = ProgressMeter(iters_per_epoch, [losses], prefix=f"[CDAN] Epoch {epoch}")

        for i in range(iters_per_epoch):
            x_s, y_s = next(train_iter1)
            x_t, y_t = next(train_iter2)
            x_s, y_s = x_s.to(device), y_s.to(device)
            x_t, y_t = x_t.to(device), y_t.to(device)

            f_s = model_s1(x_s)
            f_t = model_s2(x_t)
            y_s_pred = classifier(f_s)
            y_t_pred = classifier(f_t)

            cls_loss = F.cross_entropy(y_s_pred, y_s) + F.cross_entropy(y_t_pred, y_t)
            transfer_loss = domain_adv(y_s_pred, f_s, y_t_pred, f_t)
            loss = cls_loss + trade_off * transfer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), x_s.size(0))
            if i % 50 == 0:
                progress.display(i)
        lr_scheduler.step()

        acc1, acc2, _, _ = _eval_two_mod(model_s1, model_s2, classifier, val_loader1, val_loader2, device)
        val_acc = (acc1 + acc2) / 2.0
        print(f"  Val acc: {val_acc*100:.2f}%  (m1={acc1*100:.2f}  m2={acc2*100:.2f})")
        if val_acc > best_acc:
            best_acc = val_acc
            if save_dir:
                torch.save(model_s1.state_dict(), os.path.join(save_dir, "best_m1.pth"))
                torch.save(model_s2.state_dict(), os.path.join(save_dir, "best_m2.pth"))
                torch.save(classifier.state_dict(), os.path.join(save_dir, "best_cls.pth"))

    return best_acc


def train_mcc(model_s1, model_s2, classifier, mcc_loss_fn, optimizer, lr_scheduler,
              train_iter1, train_iter2, val_loader1, val_loader2, iters_per_epoch,
              num_epochs, device, trade_off, save_dir):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model_s1.train(); model_s2.train(); classifier.train()
        losses = AverageMeter("Loss", ":6.3f")
        progress = ProgressMeter(iters_per_epoch, [losses], prefix=f"[MCC] Epoch {epoch}")

        for i in range(iters_per_epoch):
            x_s, y_s = next(train_iter1)
            x_t, y_t = next(train_iter2)
            x_s, y_s = x_s.to(device), y_s.to(device)
            x_t, y_t = x_t.to(device), y_t.to(device)

            f_s = model_s1(x_s)
            f_t = model_s2(x_t)
            y_s_pred = classifier(f_s)
            y_t_pred = classifier(f_t)

            cls_loss = F.cross_entropy(y_s_pred, y_s) + F.cross_entropy(y_t_pred, y_t)
            transfer_loss = mcc_loss_fn(y_t_pred) + mcc_loss_fn(y_s_pred)
            loss = cls_loss + trade_off * transfer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), x_s.size(0))
            if i % 50 == 0:
                progress.display(i)
        lr_scheduler.step()

        acc1, acc2, _, _ = _eval_two_mod(model_s1, model_s2, classifier, val_loader1, val_loader2, device)
        val_acc = (acc1 + acc2) / 2.0
        print(f"  Val acc: {val_acc*100:.2f}%  (m1={acc1*100:.2f}  m2={acc2*100:.2f})")
        if val_acc > best_acc:
            best_acc = val_acc
            if save_dir:
                torch.save(model_s1.state_dict(), os.path.join(save_dir, "best_m1.pth"))
                torch.save(model_s2.state_dict(), os.path.join(save_dir, "best_m2.pth"))
                torch.save(classifier.state_dict(), os.path.join(save_dir, "best_cls.pth"))

    return best_acc


def train_mdd(mdd_module, mdd_loss_fn, optimizer, lr_scheduler,
              train_iter1, train_iter2, val_loader1, val_loader2, iters_per_epoch,
              num_epochs, device, trade_off, save_dir):
    best_acc = 0.0
    for epoch in range(num_epochs):
        mdd_module.train(); mdd_loss_fn.train()
        losses = AverageMeter("Loss", ":6.3f")
        progress = ProgressMeter(iters_per_epoch, [losses], prefix=f"[MDD] Epoch {epoch}")

        for i in range(iters_per_epoch):
            x_s, y_s = next(train_iter1)
            x_t, y_t = next(train_iter2)
            x_s, y_s = x_s.to(device), y_s.to(device)
            x_t, y_t = x_t.to(device), y_t.to(device)

            outputs, outputs_adv = mdd_module(x_s, x_t)
            y_s_pred, y_t_pred = outputs.chunk(2, dim=0)
            y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

            cls_loss = F.cross_entropy(y_s_pred, y_s) + F.cross_entropy(y_t_pred, y_t)
            transfer_loss = -mdd_loss_fn(y_s_pred, y_s_adv, y_t_pred, y_t_adv)
            loss = cls_loss + trade_off * transfer_loss
            mdd_module.step()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), x_s.size(0))
            if i % 50 == 0:
                progress.display(i)
        lr_scheduler.step()

        acc1, acc2, _, _ = _eval_mdd(mdd_module, val_loader1, val_loader2, device)
        val_acc = (acc1 + acc2) / 2.0
        print(f"  Val acc: {val_acc*100:.2f}%  (m1={acc1*100:.2f}  m2={acc2*100:.2f})")
        if val_acc > best_acc:
            best_acc = val_acc
            if save_dir:
                torch.save(mdd_module.state_dict(), os.path.join(save_dir, "best_mdd.pth"))

    return best_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["dann", "cdan", "mcc", "mdd"])
    parser.add_argument("--dataset", required=True, choices=["bge", "sen12ms", "eurosat"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--per_class_count", type=int, default=100)
    parser.add_argument("--trade_off", type=float, default=1.0)
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="Subdirectory under experiments/ (e.g. da_baselines)")
    parser.add_argument("--no_save", action="store_true",
                        help="Delete checkpoints after eval")
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Method: {args.method}  Dataset: {args.dataset}  Seed: {args.seed}")

    # --- output directory ---
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{ts}_{args.method}_{args.dataset}_seed{args.seed}"
    base = f"experiments/{args.exp_dir}" if args.exp_dir else "experiments"
    exp_dir = os.path.join(base, run_name)
    os.makedirs(exp_dir, exist_ok=True)

    cfg = _DS_CFG[args.dataset]
    num_classes = cfg["num_classes"]
    m1_name, m2_name = cfg["mod1"], cfg["mod2"]

    # --- data ---
    loaders, _, m1_name, m2_name = get_loaders(args.dataset, args.batch_size, args.per_class_count)
    train_iter1 = ForeverDataIterator(loaders[f"{m1_name}_train"])
    train_iter2 = ForeverDataIterator(loaders[f"{m2_name}_train"])
    iters_per_epoch = len(loaders[f"{m1_name}_train"])

    # --- build models ---
    model_s1, model_s2 = make_backbones(cfg["mod1_ch"], cfg["mod2_ch"], device)

    # --- method-specific setup and training ---
    if args.method == "dann":
        from tllib.alignment.dann import DomainAdversarialLoss
        from tllib.modules.domain_discriminator import DomainDiscriminator

        classifier = make_shared_classifier(num_classes, device)
        domain_disc = DomainDiscriminator(in_feature=512, hidden_size=1024).to(device)
        domain_adv = DomainAdversarialLoss(domain_disc).to(device)

        params = (list(model_s1.parameters()) + list(model_s2.parameters()) +
                  list(classifier.parameters()) + list(domain_disc.parameters()))
        optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-3, nesterov=True)
        lr_gamma, lr_decay = 0.001, 0.75
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda x: 0.01 * (1.0 + lr_gamma * x) ** (-lr_decay)
        )

        train_dann(model_s1, model_s2, classifier, domain_adv, optimizer, lr_scheduler,
                   train_iter1, train_iter2,
                   loaders[f"{m1_name}_val"], loaders[f"{m2_name}_val"],
                   iters_per_epoch, args.num_epochs, device, args.trade_off, exp_dir)

        # load best for test
        model_s1.load_state_dict(torch.load(os.path.join(exp_dir, "best_m1.pth"), weights_only=True))
        model_s2.load_state_dict(torch.load(os.path.join(exp_dir, "best_m2.pth"), weights_only=True))
        classifier.load_state_dict(torch.load(os.path.join(exp_dir, "best_cls.pth"), weights_only=True))
        acc1, acc2, f1_1, f1_2 = _eval_two_mod(
            model_s1, model_s2, classifier,
            loaders[f"{m1_name}_test"], loaders[f"{m2_name}_test"], device
        )

    elif args.method == "cdan":
        from tllib.alignment.cdan import ConditionalDomainAdversarialLoss
        from tllib.modules.domain_discriminator import DomainDiscriminator

        classifier = make_shared_classifier(num_classes, device)
        domain_disc = DomainDiscriminator(512 * num_classes, hidden_size=1024).to(device)
        domain_adv = ConditionalDomainAdversarialLoss(
            domain_disc, entropy_conditioning=False,
            num_classes=num_classes, features_dim=512,
            randomized=False, randomized_dim=1024,
        ).to(device)

        params = (list(model_s1.parameters()) + list(model_s2.parameters()) +
                  list(classifier.parameters()) + list(domain_disc.parameters()))
        optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

        train_cdan(model_s1, model_s2, classifier, domain_adv, optimizer, lr_scheduler,
                   train_iter1, train_iter2,
                   loaders[f"{m1_name}_val"], loaders[f"{m2_name}_val"],
                   iters_per_epoch, args.num_epochs, device, args.trade_off, exp_dir)

        model_s1.load_state_dict(torch.load(os.path.join(exp_dir, "best_m1.pth"), weights_only=True))
        model_s2.load_state_dict(torch.load(os.path.join(exp_dir, "best_m2.pth"), weights_only=True))
        classifier.load_state_dict(torch.load(os.path.join(exp_dir, "best_cls.pth"), weights_only=True))
        acc1, acc2, f1_1, f1_2 = _eval_two_mod(
            model_s1, model_s2, classifier,
            loaders[f"{m1_name}_test"], loaders[f"{m2_name}_test"], device
        )

    elif args.method == "mcc":
        from tllib.self_training.mcc import MinimumClassConfusionLoss

        classifier = make_shared_classifier(num_classes, device)
        mcc_loss_fn = MinimumClassConfusionLoss(temperature=2.5).to(device)

        params = (list(model_s1.parameters()) + list(model_s2.parameters()) +
                  list(classifier.parameters()))
        optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

        train_mcc(model_s1, model_s2, classifier, mcc_loss_fn, optimizer, lr_scheduler,
                  train_iter1, train_iter2,
                  loaders[f"{m1_name}_val"], loaders[f"{m2_name}_val"],
                  iters_per_epoch, args.num_epochs, device, args.trade_off, exp_dir)

        model_s1.load_state_dict(torch.load(os.path.join(exp_dir, "best_m1.pth"), weights_only=True))
        model_s2.load_state_dict(torch.load(os.path.join(exp_dir, "best_m2.pth"), weights_only=True))
        classifier.load_state_dict(torch.load(os.path.join(exp_dir, "best_cls.pth"), weights_only=True))
        acc1, acc2, f1_1, f1_2 = _eval_two_mod(
            model_s1, model_s2, classifier,
            loaders[f"{m1_name}_test"], loaders[f"{m2_name}_test"], device
        )

    elif args.method == "mdd":
        from tllib.alignment.mdd import ClassificationMarginDisparityDiscrepancy as MDD

        mdd_module = MDDModule(model_s1, model_s2, num_classes, bottleneck_dim=256, width=256).to(device)
        mdd_loss_fn = MDD(margin=4.0).to(device)

        optimizer = torch.optim.AdamW(mdd_module.get_parameters(base_lr=0.001), lr=0.001, weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

        train_mdd(mdd_module, mdd_loss_fn, optimizer, lr_scheduler,
                  train_iter1, train_iter2,
                  loaders[f"{m1_name}_val"], loaders[f"{m2_name}_val"],
                  iters_per_epoch, args.num_epochs, device, args.trade_off, exp_dir)

        mdd_module.load_state_dict(torch.load(os.path.join(exp_dir, "best_mdd.pth"), weights_only=True))
        acc1, acc2, f1_1, f1_2 = _eval_mdd(
            mdd_module, loaders[f"{m1_name}_test"], loaders[f"{m2_name}_test"], device
        )

    # --- save results ---
    avg_acc = (acc1 + acc2) / 2.0
    print(f"\n=== Test results ({args.method} / {args.dataset} / seed {args.seed}) ===")
    print(f"  {m1_name}: acc={acc1*100:.2f}%  F1={f1_1*100:.2f}%")
    print(f"  {m2_name}: acc={acc2*100:.2f}%  F1={f1_2*100:.2f}%")
    print(f"  Mean:  acc={avg_acc*100:.2f}%")

    results = {
        f"modality_{m1_name}": {"test_accuracy": acc1, "test_f1_macro": f1_1},
        f"modality_{m2_name}": {"test_accuracy": acc2, "test_f1_macro": f1_2},
        "average_test_accuracy": avg_acc,
    }
    config_out = {
        "method": args.method, "dataset": args.dataset, "seed": args.seed,
        "num_epochs": args.num_epochs, "batch_size": args.batch_size,
        "per_class_count": args.per_class_count, "trade_off": args.trade_off,
        "experiment_name": exp_dir,
    }

    with open(os.path.join(exp_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_out, f, indent=4)
    print(f"Results → {exp_dir}/test_results.json")

    if args.no_save:
        for ckpt in ["best_m1.pth", "best_m2.pth", "best_cls.pth", "best_mdd.pth"]:
            p = os.path.join(exp_dir, ckpt)
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    main()
