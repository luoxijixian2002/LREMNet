LREMNet — Internal Physical Prior-Guided Latent-Space Retinex Mamba Network
============================================================================

PyTorch implementation of **LREMNet** for Low-Light Image Enhancement
(paper: *Internal Physical Prior-Guided Latent-Space Retinex Mamba Network
for Low-Light Image Enhancement*), built on the official repository
https://github.com/luoxijixian2002/LREMNet and completed to a fully
runnable two-stage pipeline.

Method overview
---------------
The network consists of five components (paper Fig. 2):

1. **Latent-space Retinex Decomposition** (`models/decom.py`): a feature
   pyramid + channel bottleneck (256 -> 3) decomposes the input in the
   latent space with self-attention / cross-attention, producing
   ``L_low`` and ``R_low`` at H/8 x W/8.
2. **Illumination Enhancement Branch** (`models/illum.py`): a U-Net
   maps ``L_low`` to the enhanced illumination ``L_enh``.
3. **Internal Physical Prior Self-mining Network**
   (`models/edgemamba.py`, `models/lremnet.py`): cross-attention between
   ``L_enh`` and ``R_low``, then **EdgeMamba** — a gradient-magnitude
   guided selective state-space model — and a Sobel-based edge map
   ``F_edge`` gives the self-mined prior ``F_edge'``.
4. **Reflectance Enhancement** (`models/piecesmamba.py`): cross-attention
   guided by ``F_edge'``, EdgeMamba + **PiecesMamba** (12 local patches,
   independent SS2D) produce ``R_enh``.
5. **Fusion Module** (`models/lremnet.py`): cross-attention of ``L_enh``
   and ``R_enh`` + EdgeMamba + PiecesMamba + decoder produce ``I_enh``.

Total loss (Eq. 13): ``L = 0.5*L_ill + 0.5*L_edge + 0.5*L_color + 1.0*L_rec``.

Directory layout
----------------
```
LREMNet-code/
├── configs/            # stage1.yml (分解网络预训练), stage2.yml (完整网络)
├── data/LOL/           # 数据集目录（见下文）
├── datasets/           # 数据集与数据增强
├── models/             # 模型定义
│   ├── decom.py        #   CTDN 潜空间 Retinex 分解网络
│   ├── illum.py        #   光照增强 U-Net
│   ├── edgemamba.py    #   EdgeMamba（梯度先验引导 SSM）
│   ├── piecesmamba.py  #   PiecesMamba
│   ├── ss2d.py         #   SS2D 2D 选择性扫描
│   ├── CrossAttetion.py#   交叉/自注意力
│   ├── lremnet.py      #   LREMNet 完整网络
│   ├── loss_lremnet.py #   分解损失 + 总损失(Eq.13-17)
│   └── archs/          #   基础模块与选择性扫描回退实现
├── utils/              # 日志/checkpoint/优化器
├── train.py            # Stage 1：训练分解网络
├── train_stage2.py     # Stage 2：训练完整 LREMNet
├── eval.py             # Stage 1 推理（可视化分解结果）
├── eval_lremnet.py     # 完整模型推理（低光图像增强）
├── calculate_metrics.py# 在测试集上计算 PSNR / SSIM / LPIPS
└── test_models.py      # 冒烟测试（CPU 可跑）
```

Install
-------
```bash
pip install -r requirements.txt

# 可选：安装 mamba_ssm 以获得融合 CUDA 内核（需要 CUDA 工具链）
# pip install mamba-ssm
```
未安装 `mamba_ssm` 时，代码会自动使用纯 PyTorch 的选择性扫描回退实现
（`models/archs/selective_scan_fallback.py`），结果一致但速度较慢。

Data preparation
----------------
按以下结构准备配对数据集（如 LOL-v1 / LOL-v2 / LSRW）：

```
data/LOL/
├── input/            # 低光图像
├── target/           # 正常光参考图像
├── LOL_train.txt     # 每行: <low_path> <high_path>
└── LOL_val.txt
```

生成文件列表：
```bash
python datasets/transition.py --data_dir data/LOL --output_txt LOL_train.txt
# 划分 train/val 后可自行拆分成两个 txt
```

冒烟测试（无数据、无 GPU 也可运行）：
```bash
python test_models.py
```

Training
--------
Stage 1 — 预训练潜空间 Retinex 分解网络：
```bash
python train.py --config stage1.yml
```

Stage 2 — 训练完整 LREMNet（冻结分解网络，论文设置：Adam, lr=1e-4,
batch=8, 每 50 epoch 学习率 ×0.5，输入统一 resize 到 400x600）：
```bash
python train_stage2.py --config stage2.yml
```

Evaluation
----------
```bash
# 单张/文件夹推理
python eval_lremnet.py --input_path path/to/image --ckpt ckpt/stage2/stage2_weight.pth.tar

# 测试集定量指标 (PSNR / SSIM / LPIPS)
python calculate_metrics.py --filelist data/LOL/LOL_val.txt \
    --ckpt ckpt/stage2/stage2_weight.pth.tar
```
