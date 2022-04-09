from logging import Logger
from pathlib import Path
from time import time

import torch
from torchinfo import summary

from src.logger import make_logger
from src.model import ResNeXt
from src.util import FTType, init_device

if __name__ == '__main__':
    root_path: Path = Path('.')
    logger: Logger = make_logger('summary', root_path, True, direct=True)

    devices: torch.device = init_device(19260817, True)
    batch_size: int = 128
    model: ResNeXt = ResNeXt(3, 100, [(3, 32, 4, 256), (3, 32, 8, 512), (3, 32, 16, 1024)])
    logger.info(f'Summary:\n{summary(model, (batch_size, 3, 32, 32), device="cpu", verbose=0)}')

    batch_size = 128
    model = ResNeXt(3, 100, [(3, 32, 4, 256), (3, 32, 8, 512), (3, 32, 16, 1024)], devices)
    model.train()
    logger.info('Testing split size in train mode ...')

    for split_size in (8, 16, 32, 64, 128):
        model.split_size = split_size
        total_time: float = 0

        for _ in range(5):
            start_time: float = time()
            model.zero_grad()
            x: FTType = model.forward(torch.zeros(batch_size, 3, 32, 32))
            x.sum().backward()
            total_time += time() - start_time

        logger.info(f'Split size: {split_size} Average elapsed time: {total_time / 5:.2f}s')

    model.eval()
    logger.info('Testing split size in test mode ...')

    for split_size in (8, 16, 32, 64, 128):
        model.split_size = split_size
        total_time: float = 0

        for _ in range(5):
            start_time: float = time()
            with torch.no_grad():
                x: FTType = model.forward(torch.zeros(batch_size, 3, 32, 32))
            total_time += time() - start_time

        logger.info(f'Split size: {split_size} Average elapsed time: {total_time / 5:.2f}s')

    logger.info('Finish testing model.')
