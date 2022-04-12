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
                        f'Augmentation Method: {x[0]} Test Loss: {x[1]:.4f} ' +
                        f'Test Accuracy: {x[2]:.4f} Elapsed Time: {x[3]:.2f}s'
                    ), error_callback=lambda e: logger.error(f'Training Failed: {e}'))

                pool.close()
                pool.join()

            offset += max_parallel

        result: list[tuple[str, float, float, float]] = [
            (method, ) + result_dict.get(method, (float('nan'), float('nan'), float('nan')))
            for method in augmentation_methods
        ]

    logger.info(f'Finished training all models. Elapsed Time: {time() - total_start:.2f}s')
    return result


def train_model(augmentation: BatchAugmentation, config: dict[str, Any], device: torch.device,
                result_dict: dict[str, tuple[float, float, float]], name: str,
                root_path: Path) -> tuple[str, float, float, float]:
    logger: Logger = make_logger(name, root_path, False)
    logger.info('Loading CIFAR-100 dataset ...')
    method: str = name.split('-')[-1]

    try:
        process_start: float = time()
        writer: SummaryWriter = make_tensorboard(name, root_path)

        train_loader: DataLoader
        valid_loader: DataLoader
        test_loader: DataLoader
        train_loader, valid_loader, test_loader = make_loaders(get_path(root_path, 'data'), device,
                                                            **config['data'])

        logger.info('Creating model ...')
        model: ResNeXt = ResNeXt(3, 100, config['stages']).to(device)
        trainer: Trainer = Trainer(model, config['optimizer'], config['scheduler'])

        train_result: tuple[float, float] = trainer.train(
            train_loader, valid_loader, test_loader, augmentation,
            config['max_epoch'], logger, writer
        )

        torch.save(model.state_dict, get_path(root_path, 'model', name) / f'{name}.pt')
        result: tuple[float, float, float] = train_result + (time() - process_start, )
        result_dict[method] = result
        return (method, ) + result
    except Exception:
        logger.error('Training failed!', exc_info=True, stack_info=True)
        raise
