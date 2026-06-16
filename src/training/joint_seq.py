"""Sequential single-GPU joint trainer for the 'w/o BN calibration' baseline.

Trains a shared standard-BN backbone with each modality's mini-batch passed
sequentially.  Because BN running stats update after every modality's pass,
they drift toward the last modality seen each step — the 'sequential stat
mixing' effect.  At eval time with recompute_batchnorm=False this drift is
visible as degraded per-modality accuracy, demonstrating why BN calibration
is needed.

Usage (via train.py):
    training: joint_seq
    backbone: cnn_backbone_resnet   # standard BN — required for this mode
    sequential_backbone: true       # marker so eval knows no module. prefix
"""

import json
import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.backbones import get_backbone_model
from src.models.projections import get_projection_model
from src.data.dataloader import get_loader, get_num_channels
from src.utils.scheduler import get_scheduler
from src.utils.utils import set_requires_grad

warnings.filterwarnings("ignore")


class MMFLJointSeq:
    """Single-GPU joint trainer with sequential per-modality backbone calls."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.modalities = config["modality"]

        self.train_loaders, self.val_loaders = self._create_loaders()
        self.projection_models = self._init_projections()
        self.global_backbone = get_backbone_model(config, config["num_classes"]).to(self.device)

        self.optimizer, self.loss_fn, self.scheduler = self._setup_training()
        self.metrics = self._init_metrics()

        tqdm.write("MMFLJointSeq initialized (sequential standard-BN mode).")

    # ------------------------------------------------------------------ setup

    def _init_projections(self):
        models = {}
        for mod in self.modalities:
            n_ch = get_num_channels(mod, self.config["dataset"])
            models[f"modality_{mod}"] = get_projection_model(self.config, n_ch).to(self.device)
        return models

    def _create_loaders(self):
        train, val = [], []
        for mod in self.modalities:
            train.append(get_loader(mod, self.config, "train"))
            val.append(get_loader(mod, self.config, "val", shuffle=False))
        return train, val

    def _setup_training(self):
        proj_params = [p for m in self.projection_models.values() for p in m.parameters()]
        all_params = proj_params + list(self.global_backbone.parameters())
        optimizer = getattr(optim, self.config["optimizer_CLS"])(
            all_params,
            lr=self.config["lr_CLS"],
            weight_decay=self.config["weight_decay_CLS"],
            betas=tuple(self.config["betas_CLS"]),
        )
        loss_fn = getattr(nn, self.config["loss_function"])(reduction="none")
        scheduler = get_scheduler(self.config, optimizer, model_type="classifier")
        return optimizer, loss_fn, scheduler

    def _init_metrics(self):
        return {
            f"modality_{m}": {
                "train_loss_cls_list": [],
                "train_accuracy_cls_list": [],
                "val_loss_cls_list": [],
                "val_accuracy_cls_list": [],
            }
            for m in self.modalities
        }

    # ----------------------------------------------------------------- train

    def train_epoch(self, epoch):
        self.global_backbone.train()
        for m in self.projection_models.values():
            m.train()

        n_total = len(self.modalities)
        correct = torch.zeros(n_total)
        total = torch.zeros(n_total)
        losses = torch.zeros(n_total)

        n_batches_epoch = min(len(l) for l in self.train_loaders)
        it = tqdm(zip(*self.train_loaders), total=n_batches_epoch,
                  desc=f"Epoch {epoch}", leave=False)

        n_batches = 0
        for batch in it:
            self.optimizer.zero_grad()

            # Sequential per-modality forward: BN running stats update after
            # each modality → stats drift toward the last modality each step.
            batch_loss = None
            for i, (mod, (data, target)) in enumerate(zip(self.modalities, batch)):
                data, target = data.to(self.device), target.to(self.device)
                emb = self.projection_models[f"modality_{mod}"](data)
                out = self.global_backbone(emb)
                loss = self.loss_fn(out, target)
                mod_loss = loss.mean()
                batch_loss = mod_loss if batch_loss is None else batch_loss + mod_loss

                with torch.no_grad():
                    preds = out.argmax(1)
                    correct[i] += (preds == target).sum().item()
                    total[i] += target.numel()
                    losses[i] += mod_loss.item()

            (batch_loss / n_total).backward()
            self.optimizer.step()
            n_batches += 1

        losses /= n_batches
        acc = correct / total

        for i, mod in enumerate(self.modalities):
            self.metrics[f"modality_{mod}"]["train_loss_cls_list"].append(losses[i].item())
            self.metrics[f"modality_{mod}"]["train_accuracy_cls_list"].append(acc[i].item())

        return losses.mean().item(), acc.mean().item()

    # ----------------------------------------------------------------- val

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.global_backbone.eval()
        for m in self.projection_models.values():
            m.eval()

        n_total = len(self.modalities)
        correct = torch.zeros(n_total)
        total = torch.zeros(n_total)
        losses = torch.zeros(n_total)
        n_batches = 0

        for batch in zip(*self.val_loaders):
            for i, (mod, (data, target)) in enumerate(zip(self.modalities, batch)):
                data, target = data.to(self.device), target.to(self.device)
                emb = self.projection_models[f"modality_{mod}"](data)
                out = self.global_backbone(emb)
                loss = self.loss_fn(out, target)
                preds = out.argmax(1)
                correct[i] += (preds == target).sum().item()
                total[i] += target.numel()
                losses[i] += loss.mean().item()
            n_batches += 1

        losses /= n_batches
        acc = correct / total

        for i, mod in enumerate(self.modalities):
            self.metrics[f"modality_{mod}"]["val_loss_cls_list"].append(losses[i].item())
            self.metrics[f"modality_{mod}"]["val_accuracy_cls_list"].append(acc[i].item())

        return losses.mean().item(), acc.mean().item()

    # ----------------------------------------------------------------- main

    def train(self):
        best_val_acc = 0.0
        set_requires_grad(self.global_backbone, True)
        for m in self.projection_models.values():
            set_requires_grad(m, True)

        for epoch in tqdm(range(self.config["num_epochs"]), desc="Epochs", leave=False):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)
            self.scheduler.step()

            tqdm.write(f"Ep {epoch} | train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                       f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

            if val_acc > best_val_acc and epoch >= self.config["save_after_epoch"]:
                best_val_acc = val_acc
                self.config["best_epoch"] = epoch
                if self.config.get("save_model", True):
                    for mod in self.modalities:
                        torch.save(
                            self.projection_models[f"modality_{mod}"].state_dict(),
                            os.path.join(self.config["experiment_name"],
                                         f"best_projection_model_{mod}.pth"),
                        )
                    torch.save(
                        self.global_backbone.state_dict(),
                        os.path.join(self.config["experiment_name"],
                                     "best_global_backbone_model.pth"),
                    )

        tqdm.write(f"Training done. Best val acc: {best_val_acc:.4f} @ epoch "
                   f"{self.config.get('best_epoch')}")
        self.metrics["best_epoch"] = self.config.get("best_epoch")
        with open(os.path.join(self.config["experiment_name"], "metrics_rank_0.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)
