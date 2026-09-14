"""Measure inference latency / FPS / peak GPU memory for LREMNet.

对应论文 §4.3 效率分析（回应 R2-4, R4-6, R4-7）：
在单张 RTX 3090 上、以 400x600 输入测推理延迟、吞吐量与峰值显存。

用法：
    CUDA_VISIBLE_DEVICES=0 python benchmark_efficiency.py \
        --ckpt ckpt/stage2/stage2_weight.pth.tar \
        --size 400 600 --runs 100 --warmup 10 [--device cuda]

注意：
    * 对比方法请用各自官方代码 + 相同协议（同一 GPU、同尺寸、同预热/次数）测量；
    * 若未安装 mamba_ssm，测量结果为纯 PyTorch 回退实现的速度，
      论文中应注明安装 mamba_ssm 后的速度（官方 CUDA 内核）。
"""
import argparse
import time

import torch

from models.lremnet import LREMNet
import utils


def main():
    parser = argparse.ArgumentParser(description='LREMNet 效率测量')
    parser.add_argument('--ckpt', default='ckpt/stage2/stage2_weight.pth.tar',
                        help='模型权重路径（不存在则用随机权重测速度）')
    parser.add_argument('--size', nargs=2, type=int, default=[400, 600],
                        help='输入尺寸 H W（需为 8 的倍数）')
    parser.add_argument('--runs', type=int, default=100, help='计时运行次数')
    parser.add_argument('--warmup', type=int, default=10, help='预热次数')
    parser.add_argument('--device', default='cuda', help='设备')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    H, W = args.size
    print(f"=> 设备: {device}, 输入: {H}x{W}")

    model = LREMNet(channels=64).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    import os
    if os.path.isfile(args.ckpt):
        ckpt = utils.logging.load_checkpoint(args.ckpt, str(device))
        sd = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(utils.logging.strip_module_prefix(sd), strict=True)
        print(f"=> 已加载权重: {args.ckpt}")
    else:
        print(f"=> 未找到权重 {args.ckpt}，使用随机权重测量（速度基本一致）")

    x = torch.rand(1, 3, H, W, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # timing
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.runs):
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    else:
        peak_mem = float('nan')

    elapsed = time.time() - t0
    latency_ms = elapsed / args.runs * 1000.0
    fps = 1000.0 / latency_ms

    print("\n===== 测量结果 =====")
    print(f"  参数量        : {n_params / 1e6:.2f} M")
    print(f"  推理延迟      : {latency_ms:.2f} ms / 图")
    print(f"  吞吐量 (FPS)  : {fps:.2f} 图/秒")
    print(f"  峰值显存      : {peak_mem:.2f} GB")


if __name__ == "__main__":
    main()
