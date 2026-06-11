import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.projections import get_projection_model
from src.models.backbones import get_backbone_model
from src.data.dataloader import get_loader, get_num_channels
from src.utils.scheduler import get_scheduler
from src.utils.utils import set_requires_grad


class MMFLFedAvg:
    """Multi-modal federated learning with FedAvg backbone aggregation (single-process)."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.metrics = self._init_metrics()
        self.train_loaders, self.val_loaders = self._create_loaders()

        self.projections = self._init_projections()
        self.backbones = self._init_backbones()
        self.optimizers, self.loss_fns, self.schedulers = self._setup_training()

        print("MMFLFedAvg initialized.")

    # ------------------------------------------------------------------ setup

    def _init_projections(self):
        models = {}
        for mod in self.config["modality"]:
            n_ch = get_num_channels(mod, self.config["dataset"])
            models[f"modality_{mod}"] = get_projection_model(self.config, n_ch).to(self.device)
        return models

    def _init_backbones(self):
        models = {}
        for mod in self.config["modality"]:
            models[f"modality_{mod}"] = get_backbone_model(
                self.config, self.config["num_classes"]
            ).to(self.device)
        return models

    def _init_metrics(self):
        return {
            f"modality_{m}": {
                "train_loss_cls_list": [],
                "train_accuracy_cls_list": [],
                "val_loss_cls_list": [],
                "val_accuracy_cls_list": [],
            }
            for m in self.config["modality"]
        }

    def _create_loaders(self):
        train, val = [], []
        for mod in self.config["modality"]:
            train.append(get_loader(mod, self.config, "train"))
            val.append(get_loader(mod, self.config, "val", shuffle=False))
        return train, val

    def _setup_training(self):
        optimizers, loss_fns, schedulers = {}, {}, {}
        for mod in self.config["modality"]:
            key = f"modality_{mod}"
            params = list(self.projections[key].parameters()) + list(self.backbones[key].parameters())
            opt = getattr(optim, self.config["optimizer_CLS"])(
                params, lr=self.config["lr_CLS"],
                weight_decay=self.config["weight_decay_CLS"],
                betas=tuple(self.config["betas_CLS"]),
            )
            optimizers[key] = opt
            loss_fns[key] = getattr(nn, self.config["loss_function"])()
            schedulers[key] = get_scheduler(self.config, opt, "classifier")
        return optimizers, loss_fns, schedulers

    # ---------------------------------------------------------------- per-mod train/val

    def _train_modality_epoch(self, mod, loader):
        key = f"modality_{mod}"
        self.projections[key].train()
        self.backbones[key].train()

        total_loss, correct, total = 0.0, 0, 0
        for data, target in tqdm(loader, desc=f"Train {mod}", leave=False):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizers[key].zero_grad()
            out = self.backbones[key](self.projections[key](data))
            loss = self.loss_fns[key](out, target)
            loss.backward()
            self.optimizers[key].step()
            total_loss += loss.item()
            correct += out.argmax(1).eq(target).sum().item()
            total += target.size(0)

        return total_loss / len(loader), correct / total

    @torch.no_grad()
    def _validate_modality(self, mod, loader):
        key = f"modality_{mod}"
        self.projections[key].eval()
        self.backbones[key].eval()

        total_loss, correct, total = 0.0, 0, 0
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            out = self.backbones[key](self.projections[key](data))
            total_loss += self.loss_fns[key](out, target).item()
            correct += out.argmax(1).eq(target).sum().item()
            total += target.size(0)

        return total_loss / len(loader), correct / total

    # ----------------------------------------------------------------- aggregation

    def _aggregate(self):
        sizes = [len(l.dataset) for l in self.train_loaders]
        total = sum(sizes)
        weights = [self.backbones[f"modality_{m}"].state_dict() for m in self.config["modality"]]
        avg = {}
        for key in weights[0]:
            avg[key] = sum(w[key] * (s / total) for w, s in zip(weights, sizes))
        for mod in self.config["modality"]:
            self.backbones[f"modality_{mod}"].load_state_dict(avg)

    # ----------------------------------------------------------------- main loop

    def train(self):
        use_wandb = self.config.get("use_wandb", False)
        best_val_acc = 0.0
        save_threshold = self.config.get("save_after_epoch", 150)

        for rnd in range(self.config["global_rounds"]):
            # local epochs
            for _ in range(self.config["local_epochs"]):
                for i, mod in enumerate(self.config["modality"]):
                    loss, acc = self._train_modality_epoch(mod, self.train_loaders[i])
                    self.metrics[f"modality_{mod}"]["train_loss_cls_list"].append(loss)
                    self.metrics[f"modality_{mod}"]["train_accuracy_cls_list"].append(acc)
                    self.schedulers[f"modality_{mod}"].step()

            self._aggregate()

            # validate
            val_accs = []
            for i, mod in enumerate(self.config["modality"]):
                loss, acc = self._validate_modality(mod, self.val_loaders[i])
                self.metrics[f"modality_{mod}"]["val_loss_cls_list"].append(loss)
                self.metrics[f"modality_{mod}"]["val_accuracy_cls_list"].append(acc)
                val_accs.append(acc)

            avg_val = sum(val_accs) / len(val_accs)
            effective_epoch = (rnd + 1) * self.config["local_epochs"]
            tqdm.write(f"Round {rnd+1} | avg_val_acc={avg_val:.4f}")

            if use_wandb:
                import wandb
                wandb.log({"avg_val_acc": avg_val, "round": rnd})

            if avg_val > best_val_acc and effective_epoch >= save_threshold:
                best_val_acc = avg_val
                self.config["best_round"] = rnd
                if self.config.get("save_model", True):
                    for mod in self.config["modality"]:
                        torch.save(
                            self.projections[f"modality_{mod}"].state_dict(),
                            os.path.join(self.config["experiment_name"], f"best_projection_model_{mod}.pth"),
                        )
                        torch.save(
                            self.backbones[f"modality_{mod}"].state_dict(),
                            os.path.join(self.config["experiment_name"], f"best_backbone_model_{mod}.pth"),
                        )

        tqdm.write(f"FedAvg done. Best val acc: {best_val_acc:.4f}")
        self.metrics["best_round"] = self.config.get("best_round")
        with open(os.path.join(self.config["experiment_name"], "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)
