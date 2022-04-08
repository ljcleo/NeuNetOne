from typing import Optional

import torch
from torch.distributions.beta import Beta

from src.util import BatchType, FTType, LTType


class BatchAugmentation:
    def __init__(self, cut: bool, mix: bool) -> None:
        self.cut: bool = cut
        self.mix: bool = mix

    def augment(self, batch: BatchType, shuffled_batch: Optional[BatchType] = None) -> BatchType:
        input_a: FTType
        output_a: FTType
        input_a, output_a = batch
        batch_size: int = input_a.size(0)

        if self.mix:
            input_b: FTType
            output_b: FTType

            if shuffled_batch is None:
                input_b, output_b = self.shuffle_batch(batch)
            else:
                input_b, output_b = shuffled_batch

            ratio: FTType = Beta(1, 1).sample((batch_size, )).to(input_a.device)

            if self.cut:
                lower: LTType
                upper: LTType
                lower, upper = self._generate_boundary(batch_size, torch.sqrt(1 - ratio) * 16)
                input_a = input_a.clone()

                for i in range(batch_size):
                    input_a[i, :, lower[i, 0]:upper[i, 0], lower[i, 1]:upper[i, 1]] = \
                        input_b[i, :, lower[i, 0]:upper[i, 0], lower[i, 1]:upper[i, 1]]
                    ratio[i] = 1 - (upper[i, 0] - lower[i, 0]) * (upper[i, 1] - lower[i, 1]) / 1024

                output_a = output_a * ratio.view(-1, 1) + output_b * (1 - ratio.view(-1, 1))
            else:
                input_a = input_a * ratio.view(-1, 1, 1, 1) + \
                    input_b * (1 - ratio.view(-1, 1, 1, 1))
                output_a = output_a * ratio.view(-1, 1) + output_b * (1 - ratio.view(-1, 1))
        elif self.cut:
            lower: LTType
            upper: LTType
            lower, upper = self._generate_boundary(batch_size, torch.tensor(4.0))
            input_a = input_a.clone()

            for i in range(batch_size):
                input_a[i, :, lower[i, 0]:upper[i, 0], lower[i, 1]:upper[i, 1]] = 0

        return input_a, output_a

    @staticmethod
    def shuffle_batch(batch: BatchType, force_different: bool = False) -> BatchType:
        batch_size: int = batch[0].size(0)
        base_index: LTType = torch.arange(batch_size, dtype=torch.long)

        while force_different:
            shuffle_index: LTType = torch.randperm(batch_size, dtype=torch.long)
            if not torch.any(base_index == shuffle_index):
                break

        return batch[0][shuffle_index], batch[1][shuffle_index]

    @staticmethod
    def _generate_boundary(batch_size: int, offset: FTType) -> tuple[LTType, LTType]:
        offset = offset.view(-1, 1)
        center: LTType = torch.randint(0, 32, (batch_size, 2))
        lower: LTType = torch.round(torch.clip(center - offset, min=0)).to(torch.long)
        upper: LTType = torch.round(torch.clip(center + offset, max=32)).to(torch.long)
        return lower, upper
