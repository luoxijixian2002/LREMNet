"""Smoke tests: verify that every component of LREMNet builds and runs.

CPU-friendly: uses the pure-PyTorch selective-scan fallback when
``mamba_ssm`` is not installed.

Usage:
    python test_models.py [--device cpu]
"""
import argparse
import sys

import torch


def test_ctdn(device):
    from models.decom import CTDN
    print("\n[1/6] CTDN 潜空间 Retinex 分解")
    net = CTDN(channels=64).to(device).eval()
    x = torch.rand(2, 6, 96, 96, device=device)
    with torch.no_grad():
        out = net(x, pred_fea=None)
        pred = net(x, pred_fea=out["low_fea"])
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"{k} contains NaN/Inf"
        print(f"   {k:8s}: {tuple(v.shape)}")
    print(f"   pred_img : {tuple(pred['pred_img'].shape)}")


def test_illum_unet(device):
    from models.illum import IllumUNet
    print("\n[2/6] IllumUNet 光照增强分支")
    net = IllumUNet().to(device).eval()
    x = torch.rand(2, 3, 50, 75, device=device)
    with torch.no_grad():
        y = net(x)
    print(f"   input : {tuple(x.shape)}")
    print(f"   output: {tuple(y.shape)}")


def test_edge_mamba(device):
    from models.edgemamba import GradStateSpaceBlock, GradientExtractor
    print("\n[3/6] EdgeMamba (梯度先验引导的状态空间模型)")
    B, C, H, W = 2, 32, 24, 24
    block = GradStateSpaceBlock(dim=C, d_state=16).to(device).eval()
    x = torch.rand(B, H * W, C, device=device)
    with torch.no_grad():
        y = block(x, (H, W))
    assert torch.isfinite(y).all(), "EdgeMamba output contains NaN/Inf"
    print(f"   sequence input : {tuple(x.shape)}")
    print(f"   sequence output: {tuple(y.shape)}")

    extractor = GradientExtractor().to(device).eval()
    g = extractor(torch.rand(B, C, H, W, device=device))
    print(f"   gradient prior : {tuple(g.shape)}")


def test_pieces_mamba(device):
    from models.piecesmamba import PiecesMamba
    print("\n[4/6] PiecesMamba (12 局部块独立状态空间建模)")
    net = PiecesMamba(input_channels=3, num_blocks=12, num_mamba_layers=2).to(device).eval()
    x = torch.rand(2, 3, 50, 75, device=device)
    with torch.no_grad():
        y = net(x)
    assert y.shape == x.shape, f"shape mismatch {y.shape} vs {x.shape}"
    assert torch.isfinite(y).all(), "PiecesMamba output contains NaN/Inf"
    print(f"   input : {tuple(x.shape)}")
    print(f"   output: {tuple(y.shape)}")


def test_losses(device):
    from models.loss_lremnet import (decomposition_loss, enhancement_loss,
                                     gradient_magnitude, color_angle_loss)
    print("\n[5/6] 损失函数")
    out = {
        "low_R": torch.rand(2, 3, 12, 12, device=device),
        "low_L": torch.rand(2, 3, 12, 12, device=device),
        "low_fea": torch.rand(2, 3, 12, 12, device=device),
        "high_R": torch.rand(2, 3, 12, 12, device=device),
        "high_L": torch.rand(2, 3, 12, 12, device=device),
        "high_fea": torch.rand(2, 3, 12, 12, device=device),
    }
    losses = decomposition_loss(out)
    for k, v in losses.items():
        print(f"   decom/{k:11s}: {float(v):.5f}")

    out2 = {
        "L_enh": torch.rand(2, 3, 12, 12, device=device),
        "F_edge": torch.rand(2, 1, 12, 12, device=device),
        "I_enh": torch.rand(2, 3, 96, 96, device=device),
    }
    high = torch.rand(2, 3, 96, 96, device=device)
    L_gt = torch.rand(2, 3, 12, 12, device=device)
    losses2 = enhancement_loss(out2, high, L_gt)
    for k, v in losses2.items():
        print(f"   enh/{k:11s}: {float(v):.5f}")
    assert torch.isfinite(losses2["total"]), "enhancement loss is NaN"
    print(f"   grad(I_gt)     : {tuple(gradient_magnitude(high).shape)}")


def test_lremnet(device):
    from models.lremnet import LREMNet
    print("\n[6/6] LREMNet 完整网络")
    net = LREMNet(channels=64).to(device).eval()
    n_total = sum(p.numel() for p in net.parameters())
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"   总参数量: {n_total / 1e6:.3f} M")
    print(f"   可训练  : {n_train / 1e6:.3f} M")

    x = torch.rand(1, 3, 96, 96, device=device)
    with torch.no_grad():
        out = net(x)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"{k} contains NaN/Inf"
        print(f"   {k:6s}: {tuple(v.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    try:
        import mamba_ssm  # noqa: F401
        print("selective scan: mamba_ssm (CUDA kernel)")
    except ImportError:
        print("selective scan: 纯 PyTorch 回退实现 (安装 mamba_ssm 可获得更快速度)")

    test_ctdn(device)
    test_illum_unet(device)
    test_edge_mamba(device)
    test_pieces_mamba(device)
    test_losses(device)
    test_lremnet(device)
    print("\n全部冒烟测试通过!")


if __name__ == "__main__":
    sys.exit(main())
