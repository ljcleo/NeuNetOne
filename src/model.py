import torch
from torch import nn

from src.util import FTType


def init_weight(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.1)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight.data)
        nn.init.zeros_(module.bias.data)


class BottleNeck(nn.Module):
    def __init__(self, in_channel: int, stride: int, n_group: int, mid_channel: int,
                 out_channel: int) -> None:
        super().__init__()
        group_channel: int = n_group * mid_channel

        self.conv1: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channel, group_channel, 1, bias=False),
            nn.BatchNorm2d(group_channel),
            nn.ReLU(inplace=True)
        )

        self.conv2: nn.Sequential = nn.Sequential(
            nn.Conv2d(group_channel, group_channel, 3, stride=stride, padding=1,
                      groups=n_group, bias=False),
            nn.BatchNorm2d(group_channel),
            nn.ReLU(inplace=True)
        )

        self.conv3: nn.Sequential = nn.Sequential(
            nn.Conv2d(group_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.shortcut: nn.Module = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel)
        ) if in_channel != out_channel else nn.Identity()

        self.relu: nn.ReLU = nn.ReLU(inplace=True)

    def forward(self, x: FTType) -> FTType:
        return self.relu.forward(
            self.shortcut.forward(x) +
            self.conv3.forward(self.conv2.forward(self.conv1.forward(x)))
        )


class ConvStage(nn.Module):
    def __init__(self, in_channel: int, stride: int, n_block: int, n_group: int, mid_channel: int,
                 out_channel: int) -> None:
        super().__init__()

        blocks: list[BottleNeck] = [BottleNeck(in_channel, stride, n_group, mid_channel,
                                               out_channel)]
        blocks.extend([BottleNeck(out_channel, 1, n_group, mid_channel, out_channel)
                       for _ in range(1, n_block)])

        self.blocks: nn.Sequential = nn.Sequential(*blocks)

    def forward(self, x: FTType) -> FTType:
        return self.blocks.forward(x)


class ResNeXt(nn.Module):
    def __init__(self, in_channel: int, n_class: int, stages: list[dict[str, int]],
                 devices: list[torch.device] | None = None, split_size: int | None = None) -> None:
        super().__init__()
        if devices is None:
            devices = [torch.device('cpu')]

        self.split_size: int | None = split_size
        self.devices: list[torch.devices] = []
        out_channel: int = 64

        self.stages: nn.ModuleList = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )])

        in_channel = out_channel
        stride: int = 1

        for stage in stages:
            self.stages.append(ConvStage(in_channel, stride, **stage))
            in_channel = stage['out_channel']
            stride = 2

        self.stages.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channel, n_class)
        ))

        for i, module in enumerate(self.stages):
            self.devices.append(devices[i % len(devices)])
            module = module.to(self.devices[-1])

        self.apply(init_weight)

    def forward(self, x: FTType) -> FTType:
        if self.split_size is None:
            for i, module in enumerate(self.stages):
                x = module.forward(x.to(self.devices[i]))
            return x
        else:
            self._queue: list[FTType] = [None for _ in self.stages]
            self._result: list[FTType] = []

            for x in torch.split(x, self.split_size):
                self._pipeline_push(x)
            while any(x is not None for x in self._queue):
                self._pipeline_push(None)

            return torch.cat(self._result)

    def _pipeline_push(self, next_batch: FTType | None) -> None:
        n_stages: int = len(self.stages)

        for i in range(n_stages - 1, -1, -1):
            if self._queue[i] is not None:
                output: FTType = self.stages[i].forward(self._queue[i].to(self.devices[i]))

                if i == n_stages - 1:
                    self._result.append(output)
                else:
                    self._queue[i + 1] = output
            elif i < n_stages - 1:
                self._queue[i + 1] = None

        self._queue[0] = next_batch
