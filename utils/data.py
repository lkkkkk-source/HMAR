# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

# Adpated from: https://github.com/FoundationVision/VAR/blob/main/utils/data.py

import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    # build dataset
    train_root = osp.join(data_path, 'train')
    val_root = None
    for split_name in ('validate', 'val'):
        candidate = osp.join(data_path, split_name)
        if osp.isdir(candidate):
            val_root = candidate
            break

    if val_root is None:
        raise FileNotFoundError(
            f'No validation split found under {data_path}. Expected "validate" or "val".'
        )

    train_set = DatasetFolder(root=train_root, loader=pil_loader, extensions=IMG_EXTENSIONS, transform=train_aug)
    val_set = DatasetFolder(root=val_root, loader=pil_loader, extensions=IMG_EXTENSIONS, transform=val_aug)
    num_classes = len(train_set.classes)
    if train_set.classes != val_set.classes:
        raise ValueError(
            f'Train/val classes do not match: train={train_set.classes}, val={val_set.classes}'
        )
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')
    
    return num_classes, train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
