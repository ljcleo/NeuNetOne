from argparse import ArgumentParser, Namespace
from logging import Logger
from pathlib import Path
from time import time
from typing import Any

import torch
from torchinfo import summary

from src.logger import make_logger
from src.model import ResNeXt
from src.util import FTType, get_config, init_device


def split_test(devices: list[torch.device], stages: list[dict[str, int]], batch_size: int,
               base_split_size: int, repeat: int, logger: Logger) -> None:
    model: ResNeXt = ResNeXt(3, 100, stages, devices)
    model.train()
    split_sizes: tuple[int] = tuple(int(x * base_split_size) for x in (0.25, 0.5, 1, 2, 4))
    logger.info('Testing split size in train mode ...')

    for split_size in split_sizes:
        model.split_size = split_size
        total_time: float = 0

        for _ in range(repeat):
            start_time: float = time()
            model.zero_grad()
            x: FTType = model.forward(torch.zeros(batch_size, 3, 32, 32))
            x.sum().backward()
            total_time += time() - start_time

        logger.info(f'Split size: {split_size} Average elapsed time: {total_time / repeat:.2f}s')

    model.eval()
    logger.info('Testing split size in test mode ...')

    for split_size in split_sizes:
        model.split_size = split_size
        total_time: float = 0

        for _ in range(repeat):
            start_time: float = time()
            with torch.no_grad():
                x: FTType = model.forward(torch.zeros(batch_size, 3, 32, 32))
            total_time += time() - start_time

        logger.info(f'Split size: {split_size} Average elapsed time: {total_time / repeat:.2f}s')

    logger.info('Finish testing model.')


if __name__ == '__main__':
    root_path: Path = Path('.')

    parser: ArgumentParser = ArgumentParser(description='CIFAR100 ResNeXt model summary')
    parser.add_argument('-c', '--config', default='main', help='model config')
    parser.add_argument('-g', '--gpu', action='store_true', help='use GPU during test')
    parser.add_argument('-x', '--extra', action='store_true', help='perform extra tests')
    parser.add_argument('-r', '--repeat', type=int, default=5, help='extra test repeat times')
    args: Namespace = parser.parse_args()

    config_name: str = args.config
    logger: Logger = make_logger(f'{config_name}-summary', root_path, True)
    logger.info('Initializing devices ...')
    devices: list[torch.device] = init_device(19260817, args.gpu)

    logger.info(f'Loading model configuration from "{config_name}.yaml" ...')
    config: dict[str, Any] = get_config(root_path, config_name)
    stages: list[dict[str, int]] = config['stages']
    batch_size: int = config['batch']['size']

    model: ResNeXt = ResNeXt(3, 100, stages, devices=devices)
    logger.info('Model summary: \n' + str(
        summary(model, input_size=(batch_size, 3, 32, 32), device=devices[0], verbose=0)
    ))

    if args.extra:
        split_test(devices, stages, batch_size, config['batch']['split'], args.repeat, logger)
