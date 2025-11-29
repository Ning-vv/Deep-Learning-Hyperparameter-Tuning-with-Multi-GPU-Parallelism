import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import medmnist
from medmnist import INFO

# 使用 PathMNIST：单标签 9 类分类任务，适合 CrossEntropyLoss
DATA_FLAG = "pathmnist"
INFO_DICT = INFO[DATA_FLAG]
N_CLASSES = len(INFO_DICT["label"])  # PathMNIST 一共有 9 个类别

def get_transforms():
    """图像预处理：Resize 到 224x224，转 tensor，并做简单归一化。"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

def get_datasets(as_rgb: bool = True, download: bool = True):
    """
    返回 train / val / test 三个 Dataset。
    PathMNIST 本身是彩色图像，as_rgb=True 就按 3 通道读。
    """
    DataClass = getattr(medmnist, INFO_DICT["python_class"])
    transform = get_transforms()

    train_dataset = DataClass(
        split="train",
        transform=transform,
        as_rgb=as_rgb,
        download=download,
    )
    val_dataset = DataClass(
        split="val",
        transform=transform,
        as_rgb=as_rgb,
        download=download,
    )
    test_dataset = DataClass(
        split="test",
        transform=transform,
        as_rgb=as_rgb,
        download=download,
    )

    return train_dataset, val_dataset, test_dataset

def get_dataloaders(batch_size: int = 64,
                    num_workers: int = 4,
                    download: bool = True):
    """
    返回 train_loader 和 val_loader，给训练和验证使用。
    """
    train_dataset, val_dataset, _ = get_datasets(download=download)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
