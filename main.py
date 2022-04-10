from argparse import ArgumentParser, Namespace
from csv import writer
from logging import Logger
from pathlib import Path
from typing import Any

import torch

from src.logger import make_logger
from src.util import get_config, get_path, init_device
from src.task import train_all_models


if __name__ == '__main__':
    root_path: Path = Path('.')

    parser: ArgumentParser = ArgumentParser(description='CIFAR-100 ResNeXt model training')
    parser.add_argument('-c', '--config', default='main', help='model config')
    parser.add_argument('-g', '--gpu', action='store_true', help='enable GPU support')
    args: Namespace = parser.parse_args()
    use_gpu: bool = args.gpu

    config_name: str = args.config
    logger: Logger = make_logger(f'{config_name}-main', root_path, True)
    logger.info('Initializing devices ...')
    device: torch.device = init_device(19260817, use_gpu)

    logger.info(f'Loading model configuration from "{config_name}.yaml" ...')
    config: dict[str, Any] = get_config(root_path, config_name)

    compare_result: list[tuple[str, int, int]] = train_all_models(
        config, use_gpu, device, f'{config_name}-{config["name"]}', root_path
    )

    with (get_path(root_path, 'out', config_name) / f'{config_name}.csv').open('w', encoding='utf8',
                                                                               newline='') as f:
        csv_writer = writer(f)
        csv_writer.writerow(('name', 'optimizer', 'scheduler', 'batch_size',
                             'best_lr', 'best_l2', 'best_hidden', 'test_acc'))
        csv_writer.writerows(compare_result)

    logger.info('Finished writing results.')
