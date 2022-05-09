from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pylab import style

from src.util import BatchType, FTType


def visualize_augmentation(data: list[tuple[str, BatchType]], n_sample: int, label_map: list[str],
                           name: str, path: Path) -> None:

    style.use('Solarize_Light2')
    plt.figure(figsize=(12, 3 * n_sample))
    image: FTType
    label: FTType

    for index, (method, (image, label)) in enumerate(data):
        for i in range(n_sample):
            label_name: list[tuple[str, float]] = [
                (label_map[j], label[i, j].item()) for j in torch.nonzero(label[i]).ravel().tolist()
            ]

            plt.subplot(n_sample, 4, i * 4 + index + 1)
            plt.imshow(image[i].movedim(0, 2).cpu().numpy())
            plt.grid(visible=False)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(' | '.join([f'{name}: {prob:.2f}' for name, prob in label_name]))
            plt.title(method)


    plt.suptitle(f'{name.capitalize()} Set Data Augmentation Examples', size=20)
    plt.tight_layout()
    plt.savefig(path / f'{name}-augmentation.png', dpi=150)
    plt.close()
