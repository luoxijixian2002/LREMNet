"""Compute PSNR / SSIM / LPIPS on a paired test set.

Usage:
    python calculate_metrics.py --filelist data/LOL/LOL_val.txt \
        --ckpt ckpt/stage2/stage2_weight.pth.tar \
        [--output_dir results/metrics] [--device cuda] [--save_images]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from models.lremnet import LREMNet
from models.loss import SSIM as SSIMLoss
import utils


def parse_filelist(filelist):
    pairs = []
    with open(filelist) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            low, high = line.split(' ')[0], line.split(' ')[1]
            pairs.append((low, high))
    return pairs


def main():
    parser = argparse.ArgumentParser(description='LREMNet 测试集定量评估')
    parser.add_argument('--filelist', required=True, help='测试列表 txt')
    parser.add_argument('--ckpt', default='ckpt/stage2/stage2_weight.pth.tar')
    parser.add_argument('--output_dir', default='results/metrics')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save_images', action='store_true', help='保存增强结果')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"=> 设备: {device}")

    # model
    model = LREMNet(channels=64).to(device).eval()
    checkpoint = utils.logging.load_checkpoint(args.ckpt, str(device))
    sd = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(utils.logging.strip_module_prefix(sd), strict=True)

    # metrics
    ssim_fn = SSIMLoss().to(device)
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    except Exception as e:
        print(f"!! LPIPS 不可用: {e}")

    pairs = parse_filelist(args.filelist)
    print(f"=> 共 {len(pairs)} 对测试图像")

    psnr_list, ssim_list, lpips_list = [], [], []
    to_tensor = transforms.ToTensor()
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, (low_path, high_path) in enumerate(pairs):
        low_pil = Image.open(low_path).convert('RGB')
        high_pil = Image.open(high_path).convert('RGB')

        w, h = low_pil.size
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        low_t = to_tensor(low_pil).unsqueeze(0)
        high_t = to_tensor(high_pil).unsqueeze(0)
        if pad_h > 0 or pad_w > 0:
            low_t = F.pad(low_t, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            out = model(low_t.to(device))
            I_enh = out['I_enh'][:, :, :h, :w].clamp(0, 1)

        high_t = high_t.to(device)
        psnr_list.append(utils.logging.calculate_psnr(I_enh, high_t))
        ssim_list.append(ssim_fn(I_enh, high_t).item())
        if lpips_fn is not None:
            lpips_list.append(lpips_fn(I_enh, high_t).item())

        if args.save_images:
            utils.logging.save_image(
                I_enh, os.path.join(args.output_dir, f"{idx:04d}_enh.png"))

        if (idx + 1) % 10 == 0:
            print(f"   [{idx + 1}/{len(pairs)}]")

    print("\n===== 测试集结果 =====")
    print(f"  PSNR : {np.mean(psnr_list):.3f} dB")
    print(f"  SSIM : {np.mean(ssim_list):.4f}")
    if lpips_fn is not None:
        print(f"  LPIPS: {np.mean(lpips_list):.4f}")


if __name__ == "__main__":
    main()
