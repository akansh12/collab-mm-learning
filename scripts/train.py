"""Unified training entry point for collab-mm-learning.

Usage:
    # Centralized joint training (DDP, uses all available GPUs):
    python scripts/train.py --config configs/bge_joint_s1_s2.yaml

    # Override seed for multi-seed sweeps:
    python scripts/train.py --config configs/bge_joint_s1_s2.yaml --seed 123

    # Federated training (single process, no DDP):
    python scripts/train.py --config configs/bge_fedavg_s1_s2.yaml
"""

import argparse
import json
import os
import socket
import sys
import warnings

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from pathlib import Path
from torch.distributed import destroy_process_group, init_process_group

# allow `import src.*` when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.utils import generate_experiment_name, setup_seed

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ DDP helpers

def _is_port_free(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def _ddp_setup(rank, world_size, port, seed=42):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    setup_seed(seed)


# ------------------------------------------------------------------ W&B helpers

def _wandb_login(api_file="wandb_api.txt"):
    p = Path(api_file)
    if p.exists():
        os.environ["WANDB_API_KEY"] = p.read_text().strip()


def _wandb_init(config):
    _wandb_login()
    import wandb
    wandb.init(
        project=config.get("wandb_project", "collab-mm-learning"),
        name=config["model_name"],
        config=config,
        dir=config["experiment_name"],
        resume="allow",
    )
    Path(config["experiment_name"], "wandb_run_id.txt").write_text(wandb.run.id)
    wandb.run.log_code(".")


# ------------------------------------------------------------------ joint (DDP)

def _joint_worker(rank, world_size, config, port):
    from src.training.joint_ddp import MMFLJointDDP

    _ddp_setup(rank, world_size, port, seed=config.get("seed", 42))

    if rank == 0 and config.get("use_wandb", False):
        _wandb_init(config)

    trainer = MMFLJointDDP(rank, world_size, config)
    trainer.train()

    if rank == 0 and config.get("use_wandb", False):
        import wandb
        wandb.finish()

    destroy_process_group()


def _run_joint(config):
    n_mod = len(config["modality"])
    n_gpu = torch.cuda.device_count()
    world_size = min(n_gpu, n_mod) if n_gpu > 0 else 1

    rank_map = {mod: i % world_size for i, mod in enumerate(config["modality"])}
    mod_map = {mod: i for i, mod in enumerate(config["modality"])}
    config["rank_map"] = rank_map
    config["modality_map"] = mod_map

    port = config.get("port", 12398)
    if not _is_port_free(port):
        port = int(np.random.randint(10000, 20000))
        print(f"Port busy, using {port}")
    config["port"] = port

    print(f"world_size={world_size}  rank_map={rank_map}")
    mp.spawn(_joint_worker, args=(world_size, config, port), nprocs=world_size, join=True)


# ------------------------------------------------------------------ joint_seq (sequential standard-BN, single GPU)

def _run_joint_seq(config):
    from src.training.joint_seq import MMFLJointSeq
    from src.utils.utils import setup_seed

    if config["backbone"] != "cnn_backbone_resnet":
        raise ValueError(
            "joint_seq mode requires backbone: cnn_backbone_resnet (standard BN). "
            f"Got: {config['backbone']}"
        )
    setup_seed(config.get("seed", 42))
    trainer = MMFLJointSeq(config)
    trainer.train()


# ------------------------------------------------------------------ fedavg

def _run_fedavg(config):
    from src.training.fedavg import MMFLFedAvg

    if config.get("use_wandb", False):
        _wandb_init(config)

    setup_seed(config.get("seed", 42))
    trainer = MMFLFedAvg(config)
    trainer.train()

    if config.get("use_wandb", False):
        import wandb
        wandb.finish()


# ------------------------------------------------------------------ eval

def _run_eval(config):
    from src.training.testing import evaluate_test_models

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results, avg_acc = evaluate_test_models(config, device)
    results["average_test_accuracy"] = avg_acc

    out_path = os.path.join(config["experiment_name"], config.get("eval_out", "test_results.json"))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Test results → {out_path}")

    if config.get("use_wandb", False):
        _wandb_login()
        import wandb
        run_id_file = Path(config["experiment_name"]) / "wandb_run_id.txt"
        if run_id_file.exists():
            wandb.init(
                project=config.get("wandb_project", "collab-mm-learning"),
                name=config["model_name"],
                id=run_id_file.read_text().strip(),
                config=config,
                resume="must",
                dir=config["experiment_name"],
            )
        flat = {f"{mod}/{k}": v for mod, metrics in results.items()
                if isinstance(metrics, dict) for k, v in metrics.items()}
        wandb.log({**flat, "average_test_accuracy": avg_acc})
        wandb.finish()


# ------------------------------------------------------------------ main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=False, default=None, help="Path to YAML config file (not needed with --eval_only)")
    p.add_argument("--seed", type=int, default=None, help="Override config seed")
    p.add_argument("--exp_dir", type=str, default=None, help="Subdirectory under experiments/ (e.g. baseline_bge)")
    p.add_argument("--no_save", action="store_true", help="Disable model checkpoint saving")
    p.add_argument("--per_class_count", type=int, default=None, help="Override per_class_count from config")
    p.add_argument("--bn_calib_split", type=str, default=None, choices=["train", "test"], help="Override BN calibration split (default: train)")
    p.add_argument("--mod_count_override", type=str, default=None,
                   help='Per-modality per_class_count, e.g. "s1=50,s2=100"')
    p.add_argument("--eval_only", type=str, default=None, metavar="EXPERIMENT_DIR",
                   help="Skip training; re-evaluate existing checkpoints in EXPERIMENT_DIR")
    p.add_argument("--eval_out", type=str, default="test_results.json",
                   help="Output filename for eval results (default: test_results.json)")
    return p.parse_args()


def main():
    args = parse_args()

    # --eval_only: load config from existing experiment dir, skip training
    if args.eval_only:
        with open(os.path.join(args.eval_only, "config.json")) as f:
            config = json.load(f)
        config["experiment_name"] = args.eval_only
        if args.bn_calib_split is not None:
            config["bn_calib_split"] = args.bn_calib_split
        config["eval_out"] = args.eval_out
        _run_eval(config)
        return

    if args.config is None:
        print("error: --config is required when not using --eval_only")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["seed"] = args.seed
    if args.per_class_count is not None:
        config["per_class_count"] = args.per_class_count
    if args.bn_calib_split is not None:
        config["bn_calib_split"] = args.bn_calib_split
    if args.mod_count_override is not None:
        overrides = {}
        for part in args.mod_count_override.split(","):
            mod, count = part.strip().split("=")
            overrides[mod.strip()] = int(count)
        config["per_class_count_override"] = overrides
    # --no_save: save checkpoints only long enough for eval, then delete them
    delete_after_eval = args.no_save
    setup_seed(config.get("seed", 42))

    base_dir = f"experiments/{args.exp_dir}" if args.exp_dir else "experiments"
    config["experiment_name"] = str(generate_experiment_name(
        config["model_name"], base_dir=base_dir, seed=config.get("seed")
    ))
    os.makedirs(config["experiment_name"], exist_ok=True)
    with open(os.path.join(config["experiment_name"], "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"Experiment: {config['experiment_name']}")
    print(f"Dataset: {config['dataset']}  |  Training: {config.get('training', 'joint')}")

    try:
        training = config.get("training", "joint")
        if training == "joint":
            _run_joint(config)
        elif training == "joint_seq":
            _run_joint_seq(config)
        elif training == "fedavg":
            _run_fedavg(config)
        else:
            raise ValueError(f"Unknown training mode: {training}")

        _run_eval(config)

        if delete_after_eval:
            from pathlib import Path as _Path
            for pth in _Path(config["experiment_name"]).glob("*.pth"):
                pth.unlink()
            print("Deleted model checkpoints (--no_save).")

        print("Done.")

    except KeyboardInterrupt:
        print("Interrupted.")


if __name__ == "__main__":
    main()
