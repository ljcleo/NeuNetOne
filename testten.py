from argparse import ArgumentParser, Namespace
from csv import writer
from logging import Logger
from pathlib import Path
from typing import Any

import torch
from torch.utils.data.dataloader import DataLoader

from src.augment import augmentation_params
from src.data import make_test_ten_loader
from src.eval import evaluate_loss_acc
from src.init import init_devices
from src.logger import make_logger
from src.model import ResNeXt
from src.util import get_config, get_path

if __name__ == '__main__':
    root_path: Path = Path('.')

    parser: ArgumentParser = ArgumentParser(description='CIFAR-100 ResNeXt model training')
    parser.add_argument('-c', '--config', default='main', help='model config')
    parser.add_argument('-g', '--gpu', action='store_true', help='enable GPU support')
    args: Namespace = parser.parse_args()
    use_gpu: bool = args.gpu

    config_name: str = args.config
    logger: Logger = make_logger(f'{config_name}-ten', root_path, True)
    logger.info('Initializing devices ...')
    device: torch.device = init_devices(19260817, use_gpu)[0]

    logger.info(f'Loading model configuration from "{config_name}.yaml" ...')
    config: dict[str, Any] = get_config(root_path, config_name)

    logger.info('Loading test data ...')
    loader: DataLoader = make_test_ten_loader(get_path(root_path, 'data'), device,
                                              config['data']['batch_size'])

    compare_result: list[tuple[str, float, float]] = []

    for method in augmentation_params:
        logger.info(f'Loading model "{method}" ...')
        model: ResNeXt = ResNeXt(3, 100, config['stages'])
        model.load_state_dict(torch.load(get_path(root_path, 'model', config_name) /
                                         f'{config_name}-{config["name"]}-{method}.pt'))

        logger.info(f'Testing model "{method}" ...')
        compare_result.append((method, ) + evaluate_loss_acc(model.to(device), loader))

    with (get_path(root_path, 'out', config_name) /
          f'{config_name}-ten.csv').open('w', encoding='utf8', newline='') as f:
        csv_writer = writer(f)
        csv_writer.writerow(('method', 'test_loss', 'test_acc'))
        csv_writer.writerows(compare_result)

    logger.info('Finished writing results.')
