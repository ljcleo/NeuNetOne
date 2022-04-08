import torch
from torch import nn

from src.util import FTType


class BottleNeck(nn.Module):
    def __init__(self, in_channel: int, n_group: int, mid_channel: int, stride: int,
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
    def __init__(self, in_channel: int, n_block: int, n_group: int, mid_channel: int, stride: int,
                 out_channel: int) -> None:
        super().__init__()

        blocks: list[BottleNeck] = [BottleNeck(in_channel, n_group, mid_channel, stride,
                                               out_channel)]
        blocks.extend([BottleNeck(out_channel, n_group, mid_channel, 1, out_channel)
                       for _ in range(1, n_block)])

        self.blocks: nn.Sequential = nn.Sequential(*blocks)

    def forward(self, x: FTType) -> FTType:
        return self.blocks.forward(x)


class ResNeXt(nn.Module):
    def __init__(self, in_channel: int, n_class: int, stages: list[tuple[int, int, int, int]],
                 devices: list[torch.device] | None = None) -> None:
        super().__init__()
        out_channel: int = 64
        self.devices: list[torch.device] = []
        current_device: int = 0

        if devices is None:
            devices = [torch.device('cpu')]

        self.prelude: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        ).to(devices[current_device])

        self.devices.append(devices[current_device])
        current_device = (current_device + 1) % len(devices)

        self.major: nn.ModuleList = nn.ModuleList()
        in_channel = out_channel
        stride: int = 1

        for n_block, n_group, mid_channel, out_channel in stages:
            self.major.append(ConvStage(in_channel, n_block, n_group, mid_channel, stride,
                                        out_channel).to(devices[current_device]))

            self.devices.append(devices[current_device])
            current_device = (current_device + 1) % len(devices)
            in_channel = out_channel
            stride = 2

        self.conclude: nn.Sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channel, n_class)
        ).to(devices[current_device])

        self.devices.append(devices[current_device])

    def forward(self, x: FTType) -> FTType:
        current_device: int = 0
        x = self.prelude.forward(x.to(self.devices[current_device]))
        current_device += 1
        stage: ConvStage

        for stage in self.major:
            x = stage.forward(x.to(self.devices[current_device]))
            current_device += 1

        return self.conclude.forward(x.to(self.devices[current_device]))
