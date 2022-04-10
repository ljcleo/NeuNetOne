from argparse import ArgumentParser
from logging import Logger
from pathlib import Path

import torch

from src.data import CIFAR100, make_display_loader
from src.logger import make_logger
from src.util import get_path, init_device
from src.visualize import visualize_augmentation

if __name__ == '__main__':
    root_path: Path = Path('.')
    data_path: Path = get_path(root_path, 'data')
    img_path: Path = get_path(root_path, 'img')

    parser: ArgumentParser = ArgumentParser(description='CIFAR-100 data augmentation test')
    parser.add_argument('-g', '--gpu', action='store_true', help='enable GPU support')
    device: torch.device = init_device(19260817, parser.parse_args().gpu)

    logger: Logger = make_logger('augmentation', root_path, True, direct=True)
    logger.info('Start visualizing data augmentation samples ...')
    label_names: list[str] = CIFAR100.get_label_names(data_path)

    logger.info('Visualizing training set ...')
    visualize_augmentation(make_display_loader(data_path, True, 16, device), 3, label_names,
                           'training', img_path)

    logger.info('Visualizing test set ...')
    visualize_augmentation(make_display_loader(data_path, False, 16, device), 3, label_names,
                           'test', img_path)

    logger.info('Finished visualizing data augmentation samples.')
