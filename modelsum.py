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


def split_test(stages: list[dict[str, int]], batch_size: int, repeat: int,
               device: torch.device, logger: Logger) -> None:
    model: ResNeXt = ResNeXt(3, 100, stages).to(device)
    model.train()
    logger.info('Testing forward and backward time in training mode ...')
    total_forward_time: float = 0
    total_backward_time: float = 0

    for _ in range(repeat):
        model.zero_grad()

        start_time: float = time()
        x: FTType = model.forward(torch.zeros(batch_size, 3, 32, 32, device=device))
        total_forward_time += time() - start_time

        start_time = time()
        x.sum().backward()
        total_backward_time += time() - start_time

    logger.info(f'Average Forward Time: {total_forward_time / repeat:.2f}s ' +
                f'Average Backward Time: {total_backward_time / repeat:.2f}s')

    model.eval()
    logger.info('Testing forward time in test mode ...')
    total_forward_time = 0

    with torch.no_grad():
        for _ in range(repeat):
            start_time: float = time()
            model.forward(torch.zeros(batch_size, 3, 32, 32, device=device))
            total_forward_time += time() - start_time

    logger.info(f'Average Forward Time: {total_forward_time / repeat:.2f}s')


if __name__ == '__main__':
    root_path: Path = Path('.')

    parser: ArgumentParser = ArgumentParser(description='CIFAR-100 ResNeXt model summary')
    parser.add_argument('-c', '--config', default='main', help='model config')
    parser.add_argument('-g', '--gpu', action='store_true', help='enable GPU support')
    parser.add_argument('-x', '--extra', action='store_true', help='perform extra tests')
    parser.add_argument('-r', '--repeat', type=int, default=5, help='extra test repeat times')
    args: Namespace = parser.parse_args()

    config_name: str = args.config
    logger: Logger = make_logger(f'{config_name}-summary', root_path, True)
    logger.info('Initializing devices ...')
    device: torch.device = init_device(19260817, args.gpu)

    logger.info(f'Loading model configuration from "{config_name}.yaml" ...')
    config: dict[str, Any] = get_config(root_path, config_name)
    stages: list[dict[str, int]] = config['stages']

    data_config: dict[str, Any] = config['data']
    batch_size: int = data_config['batch_size']

    model: ResNeXt = ResNeXt(3, 100, stages).to(device)
    logger.info('Model summary: \n' + str(
        summary(model, input_size=(batch_size, 3, 32, 32), depth=7, device=device, verbose=0)
    ))

    if args.extra:
        split_test(stages, batch_size, args.repeat, device, logger)
    logger.info('Finished model summary.')
