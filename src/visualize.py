from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pylab import style
from torch.utils.data.dataloader import DataLoader

from src.augmentation import BatchAugmentation
from src.util import BatchType, FTType


def visualize_augmentation(loader: DataLoader, n_samples: int, label_names: list[str],
                           name: str, path: Path) -> None:
    mixup_augmentation: BatchAugmentation = BatchAugmentation(False, True)
    cutout_augmentation: BatchAugmentation = BatchAugmentation(True, False)
    cutmix_augmentation: BatchAugmentation = BatchAugmentation(True, True)

    style.use('Solarize_Light2')
    plt.figure(figsize=(12, 3 * n_samples))

    for batch in loader:
        batch: BatchType
        shuffled_batch: BatchType = BatchAugmentation.shuffle_batch(batch, True)

        baseline_image: FTType
        baseline_label: FTType
        baseline_image, baseline_label = batch

        mixup_image: FTType
        mixup_label: FTType
        mixup_image, mixup_label = mixup_augmentation.augment(batch, shuffled_batch)

        cutout_image: FTType
        cutout_label: FTType
        cutout_image, cutout_label = cutout_augmentation.augment(batch, shuffled_batch)

        cutmix_image: FTType
        cutmix_label: FTType
        cutmix_image, cutmix_label = cutmix_augmentation.augment(batch, shuffled_batch)

        for index, (method, image, label) in enumerate([
            ('Baseline', baseline_image, baseline_label),
            ('Mixup', mixup_image, mixup_label),
            ('Cutout', cutout_image, cutout_label),
            ('Cutmix', cutmix_image, cutmix_label),
        ]):
            method: str
            image: FTType
            label: FTType

            for i in range(n_samples):
                current_label: list[tuple[str, float]] = [
                    (label_names[j], label[i, j].item())
                    for j in torch.nonzero(label[i]).ravel().tolist()
                ]

                plt.subplot(n_samples, 4, i * 4 + index + 1)
                plt.imshow(image[i].movedim(0, 2).numpy())
                plt.grid(visible=False)
                plt.xticks([])
                plt.yticks([])
                plt.xlabel(' | '.join([f'{name}: {prob:.2f}' for name, prob in current_label]))
                plt.title(method)

        break

    plt.suptitle(f'{name.capitalize()} Set Data Augmentation Examples', size=20)
    plt.tight_layout()
    plt.savefig(path / f'{name}-augmentation.png', dpi=150)
    plt.close()
