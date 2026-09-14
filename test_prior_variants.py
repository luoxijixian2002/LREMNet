"""验证 E3 消融实验的 4 个先验变体（CPU 可运行）。

对应论文 §4.4 "Effect of the prior type" 消融：
    gradient (本文) | texture | frequency | none

用法：
    python test_prior_variants.py [--device cpu]
"""
import argparse

import torch

from models.edgemamba import (GradientExtractor, TextureExtractor,
                              FrequencyExtractor, NoPriorExtractor,
                              build_prior_extractor, GradStateSpaceBlock)
from models.lremnet import LREMNet


def test_extractors(device):
    print("\n[1/3] 先验提取器单独测试（输出应为 (B,1,H,W)）")
    x = torch.rand(2, 32, 24, 24, device=device)
    for name in ["gradient", "texture", "frequency", "none"]:
        extractor = build_prior_extractor(name).to(device).eval()
        with torch.no_grad():
            out = extractor(x)
        assert out.shape == (2, 1, 24, 24), f"{name}: {out.shape}"
        assert torch.isfinite(out).all(), f"{name}: 输出含 NaN/Inf"
        print(f"   {name:9s}: {tuple(out.shape)}  "
              f"min={out.min().item():.4f} max={out.max().item():.4f}")
    assert isinstance(build_prior_extractor('gradient'), GradientExtractor)
    assert isinstance(build_prior_extractor('texture'), TextureExtractor)
    assert isinstance(build_prior_extractor('frequency'), FrequencyExtractor)
    assert isinstance(build_prior_extractor('none'), NoPriorExtractor)
    assert isinstance(build_prior_extractor('no_prior'), NoPriorExtractor)


def test_blocks(device):
    print("\n[2/3] GradStateSpaceBlock（同一机制 + 不同先验）")
    B, C, H, W = 2, 32, 24, 24
    for name in ["gradient", "texture", "frequency", "none"]:
        block = GradStateSpaceBlock(dim=C, d_state=16, prior_type=name).to(device).eval()
        x = torch.rand(B, H * W, C, device=device)
        with torch.no_grad():
            y = block(x, (H, W))
        assert y.shape == x.shape and torch.isfinite(y).all()
        print(f"   {name:9s}: OK")
    # 未知类型应报错
    try:
        build_prior_extractor("unknown")
        raise AssertionError("应当抛出 ValueError")
    except ValueError:
        print("   未知 prior_type 正确抛出 ValueError")


def test_lremnet(device):
    print("\n[3/3] LREMNet 全网络前向（4 变体）")
    x = torch.rand(1, 3, 96, 96, device=device)
    for name in ["gradient", "texture", "frequency", "none"]:
        net = LREMNet(channels=64, prior_type=name).to(device).eval()
        with torch.no_grad():
            out = net(x)
        for k in ["L_enh", "R_enh", "F_edge", "I_enh"]:
            assert torch.isfinite(out[k]).all(), f"{name}/{k} 含 NaN/Inf"
        print(f"   {name:9s}: I_enh {tuple(out['I_enh'].shape)} OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    test_extractors(device)
    test_blocks(device)
    test_lremnet(device)
    print("\n4 个先验变体全部通过!")


if __name__ == "__main__":
    main()
