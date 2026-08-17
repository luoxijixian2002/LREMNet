"""Inference with the full LREMNet model.

Usage:
    python eval_lremnet.py --input_path <image|folder> \
        --ckpt ./ckpt/stage2/stage2_weight.pth.tar \
        --output_dir ./results/lremnet/ [--device cuda]
"""
import argparse
import glob
import os

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from models.lremnet import LREMNet
import utils

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def load_model(ckpt_path, device, channels=64):
    model = LREMNet(channels=channels)
    checkpoint = utils.logging.load_checkpoint(ckpt_path, str(device))
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state_dict = utils.logging.strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def get_image_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        paths = []
        for ext in IMG_EXTENSIONS:
            paths.extend(sorted(glob.glob(os.path.join(input_path, f'*{ext}'))))
            paths.extend(sorted(glob.glob(os.path.join(input_path, f'*{ext.upper()}'))))
        return paths
    else:
        raise FileNotFoundError(f"路径不存在: {input_path}")


@torch.no_grad()
def enhance_image(model, image_tensor, device):
    """image_tensor: (1, 3, H, W) in [0, 1]; H, W padded to multiples of 8."""
    image_tensor = image_tensor.to(device)
    out = model(image_tensor)
    return out["I_enh"]


def process_one(model, image_path, output_dir, device):
    img = Image.open(image_path).convert('RGB')
    w, h = img.size

    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    I_enh = enhance_image(model, img_tensor, device)
    I_enh = I_enh[:, :, :h, :w].clamp(0, 1)

    basename = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    utils.logging.save_image(I_enh, os.path.join(output_dir, f'{basename}_enh.png'))
    return basename


def main():
    parser = argparse.ArgumentParser(description='LREMNet 推理: 低光图像增强')
    parser.add_argument('--input_path', required=True, help='低光图像路径 或 文件夹路径')
    parser.add_argument('--ckpt', default='ckpt/stage2/stage2_weight.pth.tar',
                        help='模型权重路径')
    parser.add_argument('--output_dir', default='results/lremnet/', help='输出目录')
    parser.add_argument('--device', default='cuda', help='设备')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"=> 设备: {device}")

    print(f"=> 加载模型: {args.ckpt}")
    model = load_model(args.ckpt, device)

    image_paths = get_image_paths(args.input_path)
    print(f"=> 共检测到 {len(image_paths)} 张图像")

    for i, img_path in enumerate(image_paths):
        name = process_one(model, img_path, args.output_dir, device)
        print(f"   [{i+1}/{len(image_paths)}] {name} 完成")

    print(f"\n=> 全部完成，结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
