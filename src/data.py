from pathlib import Path
from pickle import load
from typing import Any

import torch
import torchvision.transforms as tf
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, random_split

from src.util import BatchType, FTType


class CIFAR100(Dataset):
    def __init__(self, path: Path, train: bool, data_sec: float, device: torch.device) -> None:
        super().__init__()
        with (path / ('train' if train else 'test')).open('rb') as f:
            data: dict[bytes, Any] = load(f, encoding='bytes')

        self._images: FTType = torch.tensor(data[b'data'], device=device).view(-1, 3, 32, 32) / 255
        self._data_size: int = round(self._images.size(0) * data_sec)
        self._images = self._images[:self._data_size]
        self._labels: FTType = torch.zeros((self._data_size, 100), device=device)
        self._labels[list(range(self._data_size)), data[b'fine_labels'][:self._data_size]] = 1

    @staticmethod
    def get_label_names(path: Path) -> list[str]:
        with (path / 'meta').open('rb') as f:
            label_names: list[bytes] = load(f, encoding='bytes')[b'fine_label_names']
        return [name.decode('utf8') for name in label_names]

    def __len__(self) -> int:
        return self._data_size

    def __getitem__(self, index: int) -> BatchType:
        return self._images[index], self._labels[index]


normalize: tf.Normalize = tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transform: tf.Compose = tf.Compose([
    tf.RandomCrop(32, padding=4), tf.RandomHorizontalFlip(), normalize
])

test_transform: tf.Compose = tf.Compose([
    tf.Pad(4), tf.TenCrop(32), tf.Lambda(lambda x: torch.stack([normalize(y) for y in x]))
])


def train_collate(batch: list[BatchType]) -> BatchType:
    return train_transform(torch.stack([x[0] for x in batch])), torch.stack([x[1] for x in batch])


def test_one_collate(batch: list[BatchType]) -> BatchType:
    return normalize(torch.stack([x[0] for x in batch])), torch.stack([x[1] for x in batch])


def test_ten_collate(batch: list[BatchType]) -> BatchType:
    return test_transform(torch.stack([x[0] for x in batch])), torch.stack([x[1] for x in batch])


def make_loaders(path: Path, device: torch.device, data_sec: float = 1, valid_sec: float = 0.1,
                 batch_size: int = 64) -> tuple[DataLoader, DataLoader, DataLoader]:
    raw_train_set: CIFAR100 = CIFAR100(path, True, data_sec, device)
    test_set: CIFAR100 = CIFAR100(path, False, data_sec, device)

    valid_size: int = round(len(raw_train_set) * valid_sec)
    train_size: int = len(raw_train_set) - valid_size
    train_set: Dataset
    valid_set: Dataset
    train_set, valid_set = random_split(raw_train_set, [train_size, valid_size])

    return tuple(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
                 for dataset, shuffle, collate_fn, in ((train_set, True, train_collate),
                                                       (valid_set, False, test_one_collate),
                                                       (test_set, False, test_one_collate)))


def make_display_loader(path: Path, device: torch.device, train: bool, n_sample: int) -> DataLoader:
    return DataLoader(CIFAR100(path, train, 0.1, device), batch_size=n_sample)


def make_test_ten_loader(path: Path, device: torch.device, batch_size: int) -> DataLoader:
    return DataLoader(CIFAR100(path, False, 1, device), batch_size=batch_size,
                      collate_fn=test_ten_collate)
