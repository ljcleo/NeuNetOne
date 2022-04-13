from argparse import ArgumentParser, Namespace
from csv import writer
from logging import Logger
from multiprocessing import set_start_method
from pathlib import Path
from typing import Any

import torch

from src.init import init_devices
from src.logger import make_logger
from src.task import train_all_models
from src.util import get_config, get_path

if __name__ == '__main__':
    set_start_method('spawn')
    root_path: Path = Path('.')

    parser: ArgumentParser = ArgumentParser(description='CIFAR-100 ResNeXt model training')
    parser.add_argument('-c', '--config', default='main', help='model config')
    parser.add_argument('-g', '--gpu', action='store_true', help='enable GPU support')
    args: Namespace = parser.parse_args()
    use_gpu: bool = args.gpu

    config_name: str = args.config
    logger: Logger = make_logger(f'{config_name}-main', root_path, True)
    logger.info('Initializing devices ...')
    devices: list[torch.device] = init_devices(19260817, use_gpu)

    logger.info(f'Loading model configuration from "{config_name}.yaml" ...')
    config: dict[str, Any] = get_config(root_path, config_name)

    compare_result: list[tuple[str, float, float, float]] = train_all_models(
        config, devices, f'{config_name}-{config["name"]}', root_path
    )

    with (get_path(root_path, 'out', config_name) /
          f'{config_name}-main.csv').open('w', encoding='utf8', newline='') as f:
        csv_writer = writer(f)
        csv_writer.writerow(('method', 'test_loss', 'test_acc', 'elapsed_time'))
        csv_writer.writerows(compare_result)

    logger.info('Finished writing results.')
