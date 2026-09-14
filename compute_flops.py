"""计算 LREMNet 当前实现的 FLOPs 与参数量。

说明：
- FLOPs 依赖输入尺寸，同时给出 400x600（LOL 协议）与 256x256 两个尺寸；
- 选择性扫描回退实现是 TorchScript 循环，计算器无法统计其内部算子，
  因此这里临时将其替换为等价的普通 Python 循环再统计（MAC 数相同）。

用法：
    python compute_flops.py
"""
import torch
from thop import profile, clever_format

from models.lremnet import LREMNet


def plain_ssm_scan_core(u, delta, A, B, C, D):
    # 与 selective_scan_fallback._ssm_scan_core 等价的普通 Python 实现
    # u, delta: (B, D, L)；A: (B, D, N)；B, C: (B, N, L)；D: (B, D)
    Bb, Dd, L = u.shape
    N = A.shape[2]
    h = torch.zeros(Bb, Dd, N, dtype=torch.float32, device=u.device)
    ys = torch.empty(Bb, Dd, L, dtype=torch.float32, device=u.device)
    for i in range(L):
        dA = torch.exp(delta[:, :, i].unsqueeze(-1) * A)
        dB = delta[:, :, i].unsqueeze(-1) * B[:, :, i].unsqueeze(1)
        h = dA * h + dB * u[:, :, i].unsqueeze(-1)
        ys[:, :, i] = torch.sum(h * C[:, :, i].unsqueeze(1), dim=-1) \
            + D * u[:, :, i]
    return ys


def main():
    # 让回退模块在统计时使用普通 Python 循环
    import models.archs.selective_scan_fallback as ssf
    ssf._ssm_scan_core = plain_ssm_scan_core

    model = LREMNet(channels=64).eval()
    n_params = sum(p.numel() for p in model.parameters())

    for size in [(400, 600), (256, 256)]:
        x = torch.rand(1, 3, size[0], size[1])
        flops, params = profile(model, inputs=(x,), verbose=False)
        print(f"输入 {size[0]}x{size[1]}: "
              f"FLOPs = {clever_format([flops], '%.2f')[0]}, "
              f"Params = {params / 1e6:.2f} M")
    print(f"参数量（直接统计）: {n_params / 1e6:.2f} M")


if __name__ == "__main__":
    main()
