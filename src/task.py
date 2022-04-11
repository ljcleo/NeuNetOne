from logging import Logger
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


def train_all_models(config: dict[str, Any], device: torch.device, name: str,
                     root_path: Path) -> list[tuple[str, int, int]]:
    total_start: float = time()
    logger: Logger = make_logger(name, root_path, True)
    logger.info('Loading CIFAR-100 dataset ...')

    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader
    train_loader, valid_loader, test_loader = make_loaders(get_path(root_path, 'data'), device,
                                                           **config['data'])

    logger.info('Start training models with different augmentation ...')
    result: list[tuple[str, int, int]] = []

    for method in config['augmentation']:
        model_start: float = time()
        new_name: str = f'{name}-{method}'
        test_loss: float
        test_acc: float

        test_loss, test_acc = train_model(
            BatchAugmentation(*augmentation_param[method]), train_loader, valid_loader, test_loader,
            config['stages'], config['optimizer'], config['scheduler'], config['max_epoch'], device,
            new_name, root_path
        )

        result.append((new_name, test_loss, test_acc))
        logger.info(f'Augmentation Method: {method} Test Loss: {test_loss:.4f} ' +
                    f'Test Accuracy: {test_acc:.4f} Elapsed Time: {time() - model_start:.2f}s')

    logger.info(f'Finished training all models. Elapsed Time: {time() - total_start:.2f}s')
    return result


def train_model(augmentation: BatchAugmentation, train_loader: DataLoader, valid_loader: DataLoader,
                test_loader: DataLoader, stages: list[dict[str, int]], optim_param: dict[str, Any],
                sched_param: dict[str, Any], max_epoch: int, device: torch.device,
                name: str, root_path: Path) -> tuple[float, float]:
    logger: Logger = make_logger(name, root_path, True)
    writer: SummaryWriter = make_tensorboard(name, root_path)
    logger.info('Creating model ...')

    model: ResNeXt = ResNeXt(3, 100, stages).to(device)
    trainer: Trainer = Trainer(model, optim_param, sched_param)

    result: tuple[float, float] = trainer.train(train_loader, valid_loader, test_loader,
                                                augmentation, max_epoch, logger, writer)

    torch.save(model.state_dict, get_path(root_path, 'model', name) / f'{name}.pt')
    return result
