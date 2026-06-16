import os
import json
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn import SyncBatchNorm
from itertools import zip_longest
from tqdm import tqdm

from src.models.projections import get_projection_model
from src.models.backbones import get_backbone_model
from src.data.dataloader import get_loader, get_num_channels
from src.utils.scheduler import get_scheduler
from src.utils.utils import set_requires_grad

warnings.filterwarnings("ignore")


class MMFLJointDDP:
    """Multi-Modal Federated Learning — centralized joint training with PyTorch DDP."""

    def __init__(self, rank, world_size, config):
        self.rank = rank
        self.world_size = world_size
        self.rank_map = config["rank_map"]
        self.mod_label_map = config["modality_map"]
        self.config = config
        self.device = torch.device(f"cuda:{rank}")

        self.selected_modalities = [
            m for m in config["modality"] if self.rank_map[m] == rank
        ]

        self.metrics = self._init_metrics()
        self.train_loader, self.val_loader = self._create_loaders()
        self.total_batches = sum(len(l) for l in self.train_loader)

        self.projection_models = self._init_projections()
        self.global_backbone = get_backbone_model(config, config["num_classes"]).to(self.device)
        if config["backbone"] == "cnn_backbone_resnet":
            self.global_backbone = SyncBatchNorm.convert_sync_batchnorm(self.global_backbone)
        self.global_backbone = nn.parallel.DistributedDataParallel(
            self.global_backbone, device_ids=[rank]
        )

        self.optimizer, self.loss_fn, self.scheduler = self._setup_training()

        if rank == 0:
            tqdm.write("MMFLJointDDP initialized.")

    # ------------------------------------------------------------------ setup

    def _init_projections(self):
        models = {}
        dataset = self.config["dataset"]
        for mod in self.selected_modalities:
            n_ch = get_num_channels(mod, dataset)
            models[f"modality_{mod}"] = get_projection_model(self.config, n_ch).to(self.device)
        return models

    def _init_metrics(self):
        return {
            f"modality_{m}": {
                "train_loss_cls_list": [],
                "train_accuracy_cls_list": [],
                "val_loss_cls_list": [],
                "val_accuracy_cls_list": [],
            }
            for m in self.selected_modalities
        }

    def _create_loaders(self):
        train, val = [], []
        for mod in self.selected_modalities:
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

    # ----------------------------------------------------------------- train

    def _process_train_batch(self, batch):
        emb_all, target_all = [], []
        with self.global_backbone.join():
            for mod, (data, target) in zip(self.selected_modalities, batch):
                data, target = data.to(self.device), target.to(self.device)
                emb = self.projection_models[f"modality_{mod}"](data)
                emb_all.append(emb)
                target_all.append(target)

        emb_all = torch.cat(emb_all, dim=0)
        target_all = torch.cat(target_all, dim=0)
        out = self.global_backbone(emb_all)
        per_sample_loss = self.loss_fn(out, target_all)

        n_local = len(self.selected_modalities)
        n_total = len(self.config["modality"])
        total_loss = per_sample_loss.mean() * (n_local / n_total)
        total_loss.backward()
        self.optimizer.step()

        # per-modality metrics
        per_mod_loss = per_sample_loss.detach().view(n_local, -1).mean(1)
        preds = out.detach().argmax(1).view(n_local, -1).T
        targets = target_all.view(n_local, -1).T

        for i, mod in enumerate(self.selected_modalities):
            gi = self.mod_label_map[mod]
            self.train_correct[gi] += (preds[:, i] == targets[:, i]).sum().item()
            self.train_total[gi] += targets[:, i].numel()
            self.train_loss[gi] += per_mod_loss[i].item()

    def train_epoch(self, epoch):
        self.global_backbone.train()
        for m in self.projection_models.values():
            m.train()

        n_total = len(self.config["modality"])
        self.train_correct = torch.zeros(n_total, device=self.device)
        self.train_total = torch.zeros(n_total, device=self.device)
        self.train_loss = torch.zeros(n_total, device=self.device)

        # zip (not zip_longest): stops at the shorter loader.
        # Safe for equal-size case (same length → identical result).
        # Required for unequal sizes: zip_longest fills exhausted loaders with
        # None, which crashes the batch unpacking; with multi-GPU DDP it also
        # deadlocks when ranks have different iteration counts.
        n_batches_epoch = min(len(l) for l in self.train_loader)
        it = (
            tqdm(zip(*self.train_loader),
                 total=n_batches_epoch,
                 desc=f"Epoch {epoch}", leave=False)
            if self.rank == 0 else zip(*self.train_loader)
        )
        n_batches = 0
        for batch in it:
            self.optimizer.zero_grad()
            self._process_train_batch(batch)
            n_batches += 1

        dist.barrier()
        for t in (self.train_correct, self.train_total, self.train_loss):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        acc = self.train_correct / self.train_total
        self.train_loss.div_(n_batches)

        for mod in self.selected_modalities:
            gi = self.mod_label_map[mod]
            self.metrics[f"modality_{mod}"]["train_loss_cls_list"].append(self.train_loss[gi].item())
            self.metrics[f"modality_{mod}"]["train_accuracy_cls_list"].append(acc[gi].item())

        return self.train_loss.mean().item(), acc.mean().item()

    # --------------------------------------------------------------- validate

    @torch.no_grad()
    def _process_val_batch(self, batch):
        emb_all, target_all = [], []
        for mod, (data, target) in zip(self.selected_modalities, batch):
            data, target = data.to(self.device), target.to(self.device)
            emb = self.projection_models[f"modality_{mod}"](data)
            emb_all.append(emb)
            target_all.append(target)

        emb_all = torch.cat(emb_all, dim=0)
        target_all = torch.cat(target_all, dim=0)
        out = self.global_backbone(emb_all)
        per_sample_loss = self.loss_fn(out, target_all)

        n_local = len(self.selected_modalities)
        per_mod_loss = per_sample_loss.view(n_local, -1).mean(1)
        preds = out.argmax(1).view(n_local, -1).T
        targets = target_all.view(n_local, -1).T

        for i, mod in enumerate(self.selected_modalities):
            gi = self.mod_label_map[mod]
            self.val_correct[gi] += (preds[:, i] == targets[:, i]).sum().item()
            self.val_total[gi] += targets[:, i].numel()
            self.val_loss[gi] += per_mod_loss[i].item()

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.global_backbone.eval()
        for m in self.projection_models.values():
            m.eval()

        n_total = len(self.config["modality"])
        self.val_correct = torch.zeros(n_total, device=self.device)
        self.val_total = torch.zeros(n_total, device=self.device)
        self.val_loss = torch.zeros(n_total, device=self.device)

        it = (
            tqdm(zip_longest(*self.val_loader), desc=f"Val {epoch}", leave=False)
            if self.rank == 0 else zip_longest(*self.val_loader)
        )
        n_batches = 0
        for batch in it:
            self._process_val_batch(batch)
            n_batches += 1

        dist.barrier()
        for t in (self.val_correct, self.val_total, self.val_loss):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        acc = self.val_correct / self.val_total
        self.val_loss.div_(n_batches)

        for mod in self.selected_modalities:
            gi = self.mod_label_map[mod]
            self.metrics[f"modality_{mod}"]["val_loss_cls_list"].append(self.val_loss[gi].item())
            self.metrics[f"modality_{mod}"]["val_accuracy_cls_list"].append(acc[gi].item())

        return self.val_loss.mean().item(), acc.mean().item()

    # ------------------------------------------------------------------- main

    def train(self):
        use_wandb = self.config.get("use_wandb", False)
        if use_wandb and self.rank == 0:
            import wandb

        best_val_acc = 0.0
        set_requires_grad(self.global_backbone, True)
        for m in self.projection_models.values():
            set_requires_grad(m, True)

        epochs = (
            tqdm(range(self.config["num_epochs"]), desc="Epochs", leave=False)
            if self.rank == 0 else range(self.config["num_epochs"])
        )
        for epoch in epochs:
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)
            self.scheduler.step()

            if self.rank == 0:
                tqdm.write(f"Ep {epoch} | train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                           f"val_loss={val_loss:.4f} acc={val_acc:.4f}")
                if use_wandb:
                    import wandb
                    wandb.log({"train_loss": train_loss, "train_acc": train_acc,
                               "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch})

            if val_acc > best_val_acc and epoch >= self.config["save_after_epoch"]:
                best_val_acc = val_acc
                self.config["best_epoch"] = epoch
                if self.config.get("save_model", True):
                    for mod in self.selected_modalities:
                        torch.save(
                            self.projection_models[f"modality_{mod}"].state_dict(),
                            os.path.join(self.config["experiment_name"], f"best_projection_model_{mod}.pth"),
                        )
                    if self.rank == 0:
                        torch.save(
                            self.global_backbone.state_dict(),
                            os.path.join(self.config["experiment_name"], "best_global_backbone_model.pth"),
                        )

        if self.rank == 0:
            tqdm.write(f"Training done. Best val acc: {best_val_acc:.4f} @ epoch {self.config.get('best_epoch')}")
            self.metrics["best_epoch"] = self.config.get("best_epoch")
            with open(os.path.join(self.config["experiment_name"], f"metrics_rank_{self.rank}.json"), "w") as f:
                json.dump(self.metrics, f, indent=4)
