from logging import Logger
from time import time
from typing import Any, Optional, Type

from torch.optim.lr_scheduler import (CosineAnnealingWarmRestarts, MultiStepLR,
                                      _LRScheduler)
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from src.augment import BatchAugmentation
from src.eval import evaluate_loss_acc, prob_ce
from src.model import ResNeXt
from src.optimizer import ASAM
from src.util import BatchType, FTType, check_inf_nan

optimizer_dict: dict[str, Type[Optimizer]] = {'sgd': SGD, 'asam': ASAM}

scheduler_dict: dict[str, Type[_LRScheduler]] = {
    'cosine': CosineAnnealingWarmRestarts,
    'multistep': MultiStepLR
}


class Trainer:
    def __init__(self, model: ResNeXt, optim_params: dict[str, Any],
                 sched_params: dict[str, Any]) -> None:
        self.model: ResNeXt = model
        self.optimizer: Optimizer = optimizer_dict[optim_params['type']](self.model.parameters(),
                                                                         **optim_params['params'])

        self.scheduler: Optional[_LRScheduler] = None if sched_params['type'] is None \
            else scheduler_dict[sched_params['type']](self.optimizer, **sched_params['params'])

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader,
              augmentation: BatchAugmentation, max_epoch: int,
              logger: Logger,  writer: SummaryWriter) -> tuple[float, float]:
        train_start: float = time()
        logger.info('Start training model ...')

        try:
            for epoch in range(max_epoch):
                epoch_start: float = time()
                train_loss: float = self._train_epoch(train_loader, augmentation, logger)

                if self.scheduler is not None:
                    self.scheduler.step()

                valid_loss: float
                valid_acc: float
                valid_loss, valid_acc = evaluate_loss_acc(self.model, valid_loader)

                writer.add_scalars('Loss', {'Train': train_loss, 'Valid': valid_loss}, epoch)
                writer.add_scalars('Accuracy', {'Valid': valid_acc}, epoch)
                writer.flush()

                logger.info(f'Epoch: {epoch + 1} Training Loss: {train_loss:.4f} ' +
                            f'Validation Loss: {valid_loss:.4f} ' +
                            f'Validation Accuracy: {valid_acc:.4f} ' +
                            f'Elapsed Time: {time() - epoch_start:.2f}s')

            test_loss: float
            test_acc: float
            test_loss, test_acc = evaluate_loss_acc(self.model, test_loader)

            logger.info(f'Finished training model. Test Loss: {test_loss:.4f} ' +
                        f'Test Accuracy: {test_acc:.4f} Elapsed time: {time() - train_start:.2f}s')
        except ValueError as e:
            logger.warn(e, exc_info=False)
            logger.info(f'Model training terminated. Elapsed time: {time() - train_start:.2f}s')
            return float('nan'), float('nan')
        except Exception as e:
            logger.info(f'Model training terminated. Elapsed time: {time() - train_start:.2f}s')
            raise

        return test_loss, test_acc

    def _train_epoch(self, loader: DataLoader, augmentation: BatchAugmentation,
                     logger: Logger) -> float:
        total_loss: float = 0
        total_count: int = 0
        self.model.train()

        total_iter: int = len(loader)
        verbose: list[int] = [round(total_iter / 10 * (i + 1)) for i in range(10)]

        cur_elapsed_time: float = 0
        cur_total_iter: int = 0
        cur_total_loss: float = 0
        cur_total_count: int = 0
        batch: BatchType

        for index, batch in enumerate(loader):
            images: FTType
            labels: FTType
            iter_start: float = time()
            self.optimizer.zero_grad()

            images, labels = augmentation.forward(batch)
            if any(check_inf_nan(images)):
                raise ValueError(f'found inf or nan after augmentation: {check_inf_nan(images)}')

            pred: FTType = self.model.forward(images)
            if any(check_inf_nan(pred)):
                raise ValueError(f'found inf or nan after prediction: {check_inf_nan(pred)}')

            loss: FTType = prob_ce(pred, labels)
            if any(check_inf_nan(loss)):
                raise ValueError(f'found inf or nan after loss calculation: {check_inf_nan(loss)}')

            loss.backward()

            if isinstance(self.optimizer, ASAM):
                self.optimizer.step_1(zero_grad=True)
                prob_ce(self.model.forward(images), labels).backward()
                self.optimizer.step_2(zero_grad=True)
            else:
                self.optimizer.step()

            index += 1
            cur_total_iter += 1
            batch_size: int = labels.size(0)
            cur_total_count += batch_size
            cur_total_loss += loss.item() * batch_size
            cur_elapsed_time += (time() - iter_start) * 1000

            if index in verbose:
                progress: float = index / total_iter
                loss_mean: float = cur_total_loss / cur_total_count
                time_mean: float = cur_elapsed_time / cur_total_iter

                logger.info(f'Iteration: {index} Progress: {progress:.0%} Loss: {loss_mean:.4f} ' +
                            f'Speed: {time_mean:.4f} ms/iter')

                total_count += cur_total_count
                total_loss += cur_total_loss
                cur_elapsed_time = cur_total_loss = cur_total_iter = cur_total_count = 0

        return total_loss / total_count
