import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from timm.scheduler import CosineLRScheduler


def get_scheduler(config, optimizer, model_type="classifier"):
    if model_type == "classifier":
        sched = config["scheduler_CLS"]
        if sched == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config["step_size_CLS"],
                gamma=config["gamma_CLS"],
            )
        elif sched == "cosine":
            return CosineLRScheduler(
                optimizer,
                t_initial=config["num_epochs"] - config["warmup_epochs_CLS"],
                lr_min=1e-5,
                warmup_lr_init=1e-6,
                warmup_t=config["warmup_epochs_CLS"],
                cycle_limit=1,
            )
        elif sched == "lambda_warmup":
            lr_lambda = _make_warmup_step_lambda(
                initial_lr=1e-6,
                base_lr=config["lr_CLS"],
                warmup_epochs=config["warmup_epochs_CLS"],
                step_size=config["step_size_CLS"],
                decay_rate=config["gamma_CLS"],
            )
            return LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            raise ValueError(f"Unknown scheduler_CLS: {sched}")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _make_warmup_step_lambda(initial_lr, base_lr, warmup_epochs, step_size, decay_rate):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            current_lr = initial_lr + (epoch / warmup_epochs) * (base_lr - initial_lr)
            return current_lr / base_lr
        num_decays = (epoch - warmup_epochs) // step_size
        return decay_rate ** num_decays
    return lr_lambda
