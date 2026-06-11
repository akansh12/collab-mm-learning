from datetime import datetime
from pathlib import Path
import random
import torch
import numpy as np


def generate_experiment_name(model_name, base_dir="experiments", seed=None):
    exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"_seed{seed}" if seed is not None else ""
    experiment_name = Path(base_dir) / (exp_name + "_" + model_name + suffix)
    experiment_name.mkdir(parents=True, exist_ok=True)
    return experiment_name


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
