from argparse import ArgumentParser
from logging import Logger
from pathlib import Path

import torch

from src.augmentation import BatchAugmentation, augmentation_param
from src.data import CIFAR100, make_display_loader
from src.init import init_devices
from src.logger import make_logger
from src.util import BatchType, get_path
from src.visualize import visualize_augmentation


def visualize(root_path: Path, train: bool, device: torch.device, logger: Logger) -> None:
    logger.info(f'Visualizing {"training" if train else "test"} set ...')
    data_path: Path = get_path(root_path, 'data')
    img_path: Path = get_path(root_path, 'img')
    batch: BatchType

    for batch in make_display_loader(data_path, train, 16, device):
        shuffled_batch: BatchType = BatchAugmentation.shuffle_batch(batch, True)

        visualize_augmentation([
            (method.capitalize(), BatchAugmentation(*param).forward(
                batch, shuffled_batch=shuffled_batch, ratio_range=(0.4, 0.6)
            )) for method, param in augmentation_param.items()
        ], 3, CIFAR100.get_label_names(data_path), 'training', img_path)

        break


if __name__ == '__main__':
    root_path: Path = Path('.')

    parser: ArgumentParser = ArgumentParser(description='CIFAR-100 data augmentation test')
    parser.add_argument('-g', '--gpu', action='store_true', help='enable GPU support')
    device: torch.device = init_devices(19260817, parser.parse_args().gpu)[0]

    logger: Logger = make_logger('augmentation', root_path, True, direct=True)
    logger.info('Start visualizing data augmentation samples ...')

    for train in (True, False):
        visualize(root_path, train, device, logger)
    logger.info('Finished visualizing data augmentation samples.')
