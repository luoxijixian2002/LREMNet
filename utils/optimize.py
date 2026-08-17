"""Optimizer factory used by the training scripts."""
import torch.optim as optim


def get_optimizer(config, parameters):
    if config.optim.optimizer == "Adam":
        betas = config.optim.betas if hasattr(config.optim, "betas") else (0.9, 0.999)
        weight_decay = config.optim.weight_decay if hasattr(config.optim, "weight_decay") else 0.0
        return optim.Adam(parameters, lr=config.optim.lr, betas=betas,
                          weight_decay=weight_decay)
    elif config.optim.optimizer == "AdamW":
        weight_decay = config.optim.weight_decay if hasattr(config.optim, "weight_decay") else 0.0
        return optim.AdamW(parameters, lr=config.optim.lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError(
            f"Optimizer {config.optim.optimizer} not supported, add it here.")
