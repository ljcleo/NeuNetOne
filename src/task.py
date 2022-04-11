from logging import Logger
from multiprocessing import Manager, Pool
from pathlib import Path
from time import time
from typing import Any

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from src.augmentation import BatchAugmentation, augmentation_param
from src.board import make_tensorboard
from src.data import make_loaders
from src.logger import make_logger
from src.model import ResNeXt
from src.trainer import Trainer
from src.util import get_path


def train_all_models(config: dict[str, Any], devices: list[torch.device], name: str,
                     root_path: Path) -> list[tuple[str, int, int]]:
    total_start: float = time()
    logger: Logger = make_logger(name, root_path, True)
    logger.info('Start training models with different augmentation ...')

    augmentation_methods: list[str] = config['augmentation']
    max_parallel: int = len(devices)

    with Manager() as manager:
        result_dict: dict[str, tuple[float, float]] = manager.dict()
        offset: int = 0

        while offset < len(augmentation_methods):
            with Pool(processes=max_parallel) as pool:
                for i, method in enumerate(augmentation_methods[offset:offset + max_parallel]):
                    logger.info(f'Launching training process for {method} ...')

                    pool.apply_async(train_model, (
                        BatchAugmentation(*augmentation_param[method]), config,
                        devices[i % len(devices)], result_dict, f'{name}-{method}', root_path
                    ), callback=lambda x: logger.info(
                        f'Augmentation Method: {method} Test Loss: {x[0]:.4f} ' +
                        f'Test Accuracy: {x[1]:.4f} Elapsed Time: {x[2]:.2f}s'
                    ))

                pool.close()
                pool.join()

            offset += max_parallel

        result: list[tuple[str, float, float, float]] = [
            (method, ) + result_dict[method] for method in augmentation_methods
        ]

    logger.info(f'Finished training all models. Elapsed Time: {time() - total_start:.2f}s')
    return result


def train_model(augmentation: BatchAugmentation, config: dict[str, Any], device: torch.device,
                result_dict: dict[str, tuple[float, float, float]], name: str,
                root_path: Path) -> tuple[float, float, float]:
    process_start: float = time()
    logger: Logger = make_logger(name, root_path, False)
    writer: SummaryWriter = make_tensorboard(name, root_path)
    logger.info('Loading CIFAR-100 dataset ...')

    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader
    train_loader, valid_loader, test_loader = make_loaders(get_path(root_path, 'data'), device,
                                                           **config['data'])

    logger.info('Creating model ...')
    model: ResNeXt = ResNeXt(3, 100, config['stages']).to(device)
    trainer: Trainer = Trainer(model, config['optimizer'], config['scheduler'])

    train_result: tuple[float, float] = trainer.train(
        train_loader, valid_loader, test_loader, augmentation, config['max_epoch'], logger, writer
    )

    torch.save(model.state_dict, get_path(root_path, 'model', name) / f'{name}.pt')
    result: tuple[float, float, float] = train_result + (time() - process_start, )
    result_dict[name.split('-')[-1]] = result
    return result
