from pathlib import Path
from pickle import load
from typing import Any, Callable

import torch
import torchvision.transforms as tf
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, random_split

from src.util import BatchType, FTType


class CIFAR100(Dataset):
    def __init__(self, path: Path, train: bool, device: torch.device) -> None:
        super().__init__()
        with (path / ('train' if train else 'test')).open('rb') as f:
            data: dict[bytes, Any] = load(f, encoding='bytes')

        self._images: FTType = torch.tensor(data[b'data'], device=device).view(-1, 3, 32, 32) / 255
        self._data_size: int = self._images.size(0)
        self._labels: FTType = torch.zeros((self._data_size, 100), device=device)
        self._labels[list(range(self._data_size)), data[b'fine_labels']] = 1

    @staticmethod
    def get_label_names(path: Path) -> list[str]:
        with (path / 'meta').open('rb') as f:
            label_names: list[bytes] = load(f, encoding='bytes')[b'fine_label_names']
        return [name.decode('utf8') for name in label_names]

    def __len__(self) -> int:
        return self._data_size

    def __getitem__(self, index: int) -> BatchType:
        return self._images[index], self._labels[index]


def make_collate(transform: tf.Compose) -> Callable[[list[BatchType]], BatchType]:
    return lambda batch: (transform(torch.stack([x[0] for x in batch])),
                          torch.stack([x[1] for x in batch]))


def make_loaders(path: Path, train_ratio: float, batch_size: tuple[int, int, int],
                 num_workers: int, pin_memory: bool,
                 device: torch.device) -> tuple[DataLoader, DataLoader, DataLoader]:
    raw_train_set: CIFAR100 = CIFAR100(path, True, device)
    test_set: CIFAR100 = CIFAR100(path, False, device)

    train_size: int = round(len(raw_train_set) * train_ratio)
    valid_size: int = len(raw_train_set) - train_size
    train_set: Dataset
    valid_set: Dataset
    train_set, valid_set = random_split(raw_train_set, [train_size, valid_size])

    normalize: tf.Normalize = tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform: tf.Compose = tf.Compose([
        tf.RandomCrop(32, padding=4), tf.RandomHorizontalFlip(), normalize
    ])

    test_transform: tf.Compose = tf.Compose([
        tf.Pad(4), tf.TenCrop(32), tf.Lambda(lambda x: torch.stack([normalize(y) for y in x]))
    ])

    train_loader: DataLoader = DataLoader(train_set, batch_size=batch_size[0], shuffle=True,
                                          num_workers=num_workers, pin_memory=pin_memory,
                                          collate_fn=make_collate(train_transform))

    valid_loader: DataLoader = DataLoader(valid_set, batch_size=batch_size[1], shuffle=False,
                                          num_workers=num_workers, pin_memory=pin_memory,
                                          collate_fn=make_collate(test_transform))

    test_loader: DataLoader = DataLoader(test_set, batch_size=batch_size[2], shuffle=False,
                                         num_workers=num_workers, pin_memory=pin_memory,
                                         collate_fn=make_collate(test_transform))

    return train_loader, valid_loader, test_loader


def make_display_loader(path: Path, train: bool, n_sample: int) -> DataLoader:
    dataset: CIFAR100 = CIFAR100(path, train, torch.device('cpu'))
    return DataLoader(dataset, batch_size=n_sample)
