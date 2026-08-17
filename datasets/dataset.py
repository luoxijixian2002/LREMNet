"""Datasets for LREMNet.

* ``LLdataset`` (stage 1): random-crop patch pairs for pre-training the
  latent Retinex decomposition network.
* ``LLdatasetStage2`` (stage 2): full-image pairs resized to the paper's
  400x600 training resolution with random hflip / rotation / color jitter.
"""
import os
import torch
import torch.utils.data
from PIL import Image
from datasets.data_augment import (PairCompose, PairToTensor,
                                   PairRandomHorizontalFilp, PairRandomCrop,
                                   PairRandomRotate)


class LLdataset:
    """Stage-1 dataset loader (random patches)."""

    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        train_dataset = AllWeatherDataset(
            self.config.data.data_dir,
            patch_size=self.config.data.patch_size,
            filelist='{}_train.txt'.format(self.config.data.train_dataset),
            train=True)
        val_dataset = AllWeatherDataset(
            self.config.data.data_dir,
            patch_size=self.config.data.patch_size,
            filelist='{}_val.txt'.format(self.config.data.val_dataset),
            train=False)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.training.batch_size,
            shuffle=True, num_workers=self.config.data.num_workers,
            pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config.sampling.batch_size,
            shuffle=False, num_workers=self.config.data.num_workers,
            pin_memory=True)
        return train_loader, val_loader


class LLdatasetStage2:
    """Stage-2 dataset loader (full images resized to 400x600)."""

    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        train_dataset = AllWeatherDataset(
            self.config.data.data_dir,
            patch_size=None,
            filelist='{}_train.txt'.format(self.config.data.train_dataset),
            train=True,
            size=tuple(self.config.data.size))
        val_dataset = AllWeatherDataset(
            self.config.data.data_dir,
            patch_size=None,
            filelist='{}_val.txt'.format(self.config.data.val_dataset),
            train=False,
            size=tuple(self.config.data.size))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.training.batch_size,
            shuffle=True, num_workers=self.config.data.num_workers,
            pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config.sampling.batch_size,
            shuffle=False, num_workers=self.config.data.num_workers,
            pin_memory=True)
        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    """Paired low / normal light dataset driven by a file list.

    Each line of the file list contains: ``<low_path> <high_path>``.
    """

    def __init__(self, dir, patch_size=256, filelist=None, train=True, size=None):
        super().__init__()
        self.dir = dir
        self.patch_size = patch_size
        self.size = size  # (H, W) for stage 2 resize
        self.train = train

        train_list = os.path.join(dir, filelist)
        with open(train_list) as f:
            contents = f.readlines()
            self.input_names = [i.strip() for i in contents]

        if train:
            transforms = [PairRandomHorizontalFilp()]
            if patch_size is not None:
                transforms.append(PairRandomCrop((patch_size, patch_size)))
            if size is not None:
                transforms.append(PairResize(size))
                transforms.append(PairRandomRotate())
            transforms.append(PairToTensor())
        else:
            transforms = []
            if size is not None:
                transforms.append(PairResize(size))
            transforms.append(PairToTensor())
        self.transforms = PairCompose(transforms)

    def get_images(self, index):
        input_name = self.input_names[index]
        low_img_name, high_img_name = input_name.split(' ')[0], input_name.split(' ')[1]
        img_id = os.path.basename(low_img_name)
        low_img = Image.open(low_img_name).convert('RGB')
        high_img = Image.open(high_img_name).convert('RGB')
        low_img, high_img = self.transforms(low_img, high_img)
        return torch.cat([low_img, high_img], dim=0), img_id

    def __getitem__(self, index):
        return self.get_images(index)

    def __len__(self):
        return len(self.input_names)


class PairResize:
    """Resize both images to (size[1], size[0]) = (W, H)."""

    def __init__(self, size):
        self.size = size  # (H, W)

    def __call__(self, img, label):
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR), \
            label.resize((self.size[1], self.size[0]), Image.BILINEAR)
