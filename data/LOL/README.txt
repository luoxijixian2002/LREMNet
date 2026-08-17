数据集目录说明
==============

在此目录下放置配对数据集，结构如下：

data/LOL/
├── input/            # 低光图像 (low-light images)
├── target/           # 正常光参考图像 (normal-light ground truth)
├── LOL_train.txt     # 训练列表: 每行 "<low路径> <high路径>"
└── LOL_val.txt       # 验证列表

生成文件列表：
    python datasets/transition.py --data_dir data/LOL --output_txt LOL_train.txt

如需测试其它数据集（LOL-v2-real / LSRW 等），在 configs/*.yml 中修改
train_dataset / val_dataset 名称并准备对应 txt 文件即可。
