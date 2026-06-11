import os
import copy
import json
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.models.projections import get_projection_model
from src.models.backbones import get_backbone_model, replace_gbn_with_bn, recompute_batchnorm_running_stats
from src.data.dataloader import get_loader, get_num_channels


# ------------------------------------------------------------------ per-model eval

def test_model(device, proj, backbone, dataloader, loss_fn=None):
    """Evaluate a projection + backbone pair. Returns (loss, accuracy, weighted_f1)."""
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    proj.to(device).eval()
    backbone.to(device).eval()

    total_loss, correct = 0.0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            out = backbone(proj(data))
            total_loss += loss_fn(out, target).item()
            preds = out.argmax(1)
            correct += preds.eq(target).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    n = len(dataloader.dataset)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted")
    return total_loss / len(dataloader), correct / n, macro_f1, weighted_f1


# ------------------------------------------------------------------ batch evaluation

def _load_projection_models(config):
    models = {}
    for mod in config["modality"]:
        n_ch = get_num_channels(mod, config["dataset"])
        models[f"modality_{mod}"] = get_projection_model(config, n_ch)
    return models


def evaluate_test_models(config, device, split="test"):
    """Load best checkpoints, optionally run BN calibration, evaluate on test set.

    Handles both joint (shared global backbone) and fedavg (per-modality backbone) modes.
    """
    results = {}
    recompute = config.get("recompute_batchnorm", True)
    training_mode = config.get("training", "joint")

    proj_models = _load_projection_models(config)

    # load global backbone (strip DDP 'module.' prefix)
    if training_mode == "joint":
        global_backbone = get_backbone_model(config, config["num_classes"])
        sd = torch.load(
            os.path.join(config["experiment_name"], "best_global_backbone_model.pth"),
            map_location=device,
        )
        clean_sd = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in sd.items())
        global_backbone.load_state_dict(clean_sd)
        global_backbone.to(device)

    avg_acc = []
    for mod in config["modality"]:
        key = f"modality_{mod}"
        proj = proj_models[key].to(device)
        proj.load_state_dict(
            torch.load(
                os.path.join(config["experiment_name"], f"best_projection_model_{mod}.pth"),
                map_location=device,
            )
        )

        # per-modality backbone copy
        if training_mode == "joint":
            backbone = copy.deepcopy(global_backbone)
        else:  # fedavg — each modality has its own saved backbone
            backbone = get_backbone_model(config, config["num_classes"]).to(device)
            backbone.load_state_dict(
                torch.load(
                    os.path.join(config["experiment_name"], f"best_backbone_model_{mod}.pth"),
                    map_location=device,
                )
            )

        test_loader = get_loader(mod, config, data_type=split, shuffle=False)

        if recompute:
            cal_bs = min(1024, config["per_class_count"] * config["num_classes"])
            cal_config = {**config, "batch_size": cal_bs}
            replace_gbn_with_bn(backbone)
            replace_gbn_with_bn(proj)
            cal_loader = get_loader(mod, cal_config, data_type="train", shuffle=True)
            recompute_batchnorm_running_stats(backbone, proj, cal_loader, device)

        loss, acc, macro_f1, weighted_f1 = test_model(device, proj, backbone, test_loader)
        results[key] = {"test_loss": loss, "test_accuracy": acc, "test_f1_macro": macro_f1, "test_f1_weighted": weighted_f1}
        avg_acc.append(acc)
        tqdm.write(f"Modality {mod}: loss={loss:.4f} acc={acc:.4f} f1_macro={macro_f1:.4f} f1_weighted={weighted_f1:.4f}")

    avg = sum(avg_acc) / len(avg_acc)
    tqdm.write(f"Average test accuracy: {avg:.4f}")
    return results, avg


# ------------------------------------------------------------------ t-SNE

def _calc_embeddings(proj, dataloader, device, modality):
    proj.to(device).eval()
    embs, labels, domains = [], [], []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            out = proj(data)
            if out.dim() > 2:
                out = out.mean(dim=(-2, -1))
            embs.append(out.cpu())
            labels.extend(target.tolist())
            domains.extend([modality] * data.size(0))
    return torch.cat(embs, 0), labels, domains


def get_tsne_embeddings(config, device, split="test"):
    from sklearn.manifold import TSNE

    proj_models = _load_projection_models(config)
    for mod in config["modality"]:
        proj_models[f"modality_{mod}"].load_state_dict(
            torch.load(
                os.path.join(config["experiment_name"], f"best_projection_model_{mod}.pth"),
                map_location=device,
            )
        )

    all_embs, all_labels, all_domains = [], [], []
    for mod in tqdm(config["modality"], desc="Embeddings"):
        loader = get_loader(mod, config, data_type=split, shuffle=False)
        embs, labels, domains = _calc_embeddings(proj_models[f"modality_{mod}"], loader, device, mod)
        all_embs.append(embs)
        all_labels.extend(labels)
        all_domains.extend(domains)

    all_embs = torch.cat(all_embs, 0).numpy()
    tsne = TSNE(n_components=2, perplexity=30, verbose=0).fit_transform(all_embs)

    save_dir = os.path.join(config["experiment_name"], f"tsne_{split}")
    os.makedirs(save_dir, exist_ok=True)

    COLORS = ["#d62728", "#8c564b", "#005000", "#1E90FF", "#043C69", "#DAA520",
              "#9467bd", "#e377c2", "#7f7f7f", "#bcbd22"]
    MARKERS = ["o", "s", "^", "D", "*", "P", "X", "+", "x", "h"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for mi, mod in enumerate(config["modality"]):
        idxs = [i for i, d in enumerate(all_domains) if d == mod]
        ax.scatter(tsne[idxs, 0], tsne[idxs, 1],
                   label=f"mod_{mod}", marker=MARKERS[mi % len(MARKERS)],
                   c=[COLORS[all_labels[i] % len(COLORS)] for i in idxs],
                   alpha=0.6, edgecolors="k", linewidths=0.5)
    ax.legend()
    ax.set_title(f"t-SNE ({split})")
    fig.savefig(os.path.join(save_dir, "tsne_all.png"))
    return fig
