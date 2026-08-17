"""Logging / checkpoint / image helpers (compatible with the official
``utils.logging.save_image / save_checkpoint / load_checkpoint`` API)."""
import os
import torch
import torchvision.utils as tvu


def save_image(tensor, path):
    """Save a (1, 3, H, W) / (B, 3, H, W) tensor in [0, 1] as a PNG."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tvu.save_image(tensor.float().clamp(0, 1), path)


def save_checkpoint(state, filename):
    path = filename + ".pth.tar" if not filename.endswith(".pth.tar") else filename
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def load_checkpoint(ckpt_path, device):
    if device == "cuda":
        state = torch.load(ckpt_path, map_location="cuda")
    else:
        state = torch.load(ckpt_path, map_location="cpu")
    return state


def strip_module_prefix(state_dict):
    """Remove ``module.`` prefixes added by nn.DataParallel."""
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def calculate_psnr(img1, img2):
    """PSNR between two (B, 3, H, W) tensors in [0, 1]."""
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
