"""Stage 2: train the full LREMNet.

The latent-space Retinex decomposition network (CTDN) is loaded from the
stage-1 checkpoint and kept frozen.  The enhancement branches are trained
with the paper's composite loss (Eq. 13):

    L_total = 0.5*L_ill + 0.5*L_edge + 0.5*L_color + 1.0*L_rec

Usage:
    python train_stage2.py --config stage2.yml [--resume PATH] \
        [--image_folder results/stage2/] [--device cuda] [--seed 230]
"""
import argparse
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

import datasets
import utils
from models.lremnet import LREMNet
from models.loss_lremnet import enhancement_loss
from utils.logging import calculate_psnr


def dict2namespace(config_dict):
    namespace = argparse.Namespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Stage2: LREMNet training')
    parser.add_argument("--config", default='stage2.yml', type=str,
                        help="配置文件路径 (configs/ 目录下)")
    parser.add_argument('--resume', default='', type=str, help='恢复训练的 checkpoint')
    parser.add_argument("--image_folder", default='results/stage2/', type=str,
                        help="验证图像保存路径")
    parser.add_argument('--device', default='cuda', type=str, help='设备 cuda/cpu')
    parser.add_argument('--seed', default=230, type=int, help='随机种子')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = dict2namespace(yaml.safe_load(f))
    return args, config


def load_stage1_decom(model, ckpt_path, device):
    """Load CTDN weights from a stage-1 checkpoint into ``model.decom``."""
    checkpoint = utils.logging.load_checkpoint(ckpt_path, str(device))
    sd = checkpoint['model'] if 'model' in checkpoint else checkpoint
    sd = utils.logging.strip_module_prefix(sd)

    # stage-1 keys look like ``ReconNet.xxx`` while the submodule is
    # ``decom.ReconNet.xxx`` inside LREMNet -- add the prefix when needed
    sd = {('decom.' + k if not k.startswith('decom.') else k): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k.startswith('decom.')]
    if missing:
        raise RuntimeError(f"Stage-1 checkpoint incomplete, missing keys: {missing[:10]} ...")
    print(f"=> 从 {ckpt_path} 加载了分解网络权重")
    print(f"   (忽略的非分解网络键: {len(unexpected)} 个)")


@torch.no_grad()
def validate(model, val_loader, device, args, global_step, epoch):
    model.eval()
    save_dir = os.path.join(args.image_folder, f"epoch{epoch}_step{global_step}")
    os.makedirs(save_dir, exist_ok=True)

    psnr_list = []
    for i, (x, img_ids) in enumerate(val_loader):
        x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
        x = x.to(device)
        low_img = x[:, :3, :, :]
        high_img = x[:, 3:, :, :]

        out = model(low_img)
        I_enh = out["I_enh"]

        for j in range(low_img.shape[0]):
            psnr_list.append(calculate_psnr(I_enh[j:j + 1], high_img[j:j + 1]))

        # save a few sample images
        if i < 2:
            for j in range(min(x.shape[0], 2)):
                img_id = img_ids[j] if isinstance(img_ids, list) else f"{i}_{j}"
                utils.logging.save_image(low_img[j:j + 1],
                                         os.path.join(save_dir, f"{img_id}_low.png"))
                utils.logging.save_image(I_enh[j:j + 1],
                                         os.path.join(save_dir, f"{img_id}_enh.png"))
                utils.logging.save_image(high_img[j:j + 1],
                                         os.path.join(save_dir, f"{img_id}_gt.png"))

    model.train()
    mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    print(f"  验证 PSNR: {mean_psnr:.3f} dB (图像保存在 {save_dir})")
    return mean_psnr


def main():
    args, config = parse_args_and_config()

    device = torch.device(args.device if torch.cuda.is_available() else torch.device("cpu"))
    print(f"=> 使用设备: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # ---------------- data ----------------
    print(f"=> 加载数据集: '{config.data.train_dataset}'")
    DATASET = datasets.__dict__[config.data.type](config)
    train_loader, val_loader = DATASET.get_loaders()
    print(f"   训练集批次数: {len(train_loader)}, 验证集批次数: {len(val_loader)}")

    # ---------------- model ----------------
    model = LREMNet(
        channels=config.model.channels,
        num_edge_blocks=config.model.num_edge_blocks,
        pieces_num_blocks=config.model.pieces_num_blocks,
        pieces_layers=config.model.pieces_layers,
        prior_type=getattr(config.model, 'prior_type', 'gradient'),
        latent_dim=getattr(config.model, 'latent_dim', 3),
    )
    model = model.to(device)

    if os.path.isfile(config.data.stage1_ckpt):
        load_stage1_decom(model, config.data.stage1_ckpt, device)
    else:
        print(f"!! 未找到 Stage-1 权重 {config.data.stage1_ckpt}，分解网络将随机初始化 !!")

    # freeze the decomposition network
    for p in model.decom.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"=> 可训练参数量: {sum(p.numel() for p in trainable) / 1e6:.3f} M")

    # ---------------- optimizer & scheduler ----------------
    optimizer = utils.optimize.get_optimizer(config, trainable)
    if config.optim.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config.training.n_epochs, eta_min=1e-7)
    elif config.optim.lr_scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=config.optim.step_size,
                           gamma=config.optim.gamma)
    else:
        scheduler = None

    # ---------------- loss weights (Eq. 13) ----------------
    loss_cfg = config.loss
    print(f"=> 损失权重: ill={loss_cfg.weight_illumination}, "
          f"edge={loss_cfg.weight_edge}, color={loss_cfg.weight_color}, "
          f"rec={loss_cfg.weight_recon}")

    # ---------------- resume ----------------
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = utils.logging.load_checkpoint(args.resume, str(device))
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('step', 0)
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"=> 从 checkpoint 恢复: epoch={start_epoch}, step={global_step}")

    os.makedirs(config.data.ckpt_dir, exist_ok=True)

    # ---------------- training loop ----------------
    print("\n" + "=" * 60)
    print("开始 Stage 2 训练 — LREMNet 完整网络")
    print(f"  学习率: {config.optim.lr}, 批大小: {config.training.batch_size}, "
          f"轮数: {config.training.n_epochs}")
    print("=" * 60 + "\n")

    for epoch in range(start_epoch, config.training.n_epochs):
        model.train()
        epoch_total, epoch_ill, epoch_edge = 0.0, 0.0, 0.0
        epoch_color, epoch_rec = 0.0, 0.0
        data_start = time.time()
        data_time = 0.0

        for i, (x, img_ids) in enumerate(train_loader):
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            x = x.to(device)
            low_img = x[:, :3, :, :]
            high_img = x[:, 3:, :, :]

            out = model(low_img)

            # GT illumination: decomposed from the normal-light image
            _, L_gt = model.decompose_high(high_img)

            losses = enhancement_loss(
                out, high_img, L_gt,
                w_ill=loss_cfg.weight_illumination,
                w_edge=loss_cfg.weight_edge,
                w_color=loss_cfg.weight_color,
                w_rec=loss_cfg.weight_recon,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            clip = config.training.clip_grad_norm
            if clip and clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=clip)
            optimizer.step()

            global_step += 1
            data_time += time.time() - data_start

            epoch_total += losses["total"].item()
            epoch_ill += losses["ill"].item()
            epoch_edge += losses["edge"].item()
            epoch_color += losses["color"].item()
            epoch_rec += losses["rec"].item()

            if global_step % 10 == 0:
                print(f"Epoch[{epoch}] Step[{global_step}] "
                      f"Loss:{losses['total'].item():.5f} "
                      f"Ill:{losses['ill'].item():.4f} "
                      f"Edge:{losses['edge'].item():.4f} "
                      f"Color:{losses['color'].item():.4f} "
                      f"Rec:{losses['rec'].item():.4f} "
                      f"LR:{optimizer.param_groups[0]['lr']:.7f} "
                      f"Time:{data_time / (i + 1):.3f}")

            data_start = time.time()

            if global_step % config.training.validation_freq == 0:
                print(f"\n=> 第 {global_step} 步验证...")
                validate(model, val_loader, device, args, global_step, epoch)

                save_dict = {
                    'epoch': epoch,
                    'step': global_step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if scheduler is not None:
                    save_dict['scheduler'] = scheduler.state_dict()
                utils.logging.save_checkpoint(
                    save_dict,
                    filename=os.path.join(config.data.ckpt_dir, 'model_latest'))
                utils.logging.save_checkpoint(
                    save_dict,
                    filename=os.path.join(config.data.ckpt_dir, 'stage2_weight'))
                print(f"=> Checkpoint 已保存到 {config.data.ckpt_dir}/\n")

        if scheduler is not None:
            scheduler.step()

        num_batches = i + 1
        print(f"\n===== Epoch [{epoch}] 完成 =====\n"
              f"  Avg Total: {epoch_total / num_batches:.5f}\n"
              f"  Avg Ill:   {epoch_ill / num_batches:.5f}\n"
              f"  Avg Edge:  {epoch_edge / num_batches:.5f}\n"
              f"  Avg Color: {epoch_color / num_batches:.5f}\n"
              f"  Avg Rec:   {epoch_rec / num_batches:.5f}\n"
              f"  LR:        {optimizer.param_groups[0]['lr']:.7f}\n")

        utils.logging.save_checkpoint(
            {'epoch': epoch, 'step': global_step,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict()},
            filename=os.path.join(config.data.ckpt_dir, f'epoch_{epoch}')
        )

    print("\n" + "=" * 60)
    print("Stage 2 训练完成!")
    print(f"最终模型保存在: {config.data.ckpt_dir}/stage2_weight.pth.tar")
    print("可以使用 eval.py 进行推理")
    print("=" * 60)


if __name__ == "__main__":
    main()
