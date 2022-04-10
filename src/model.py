import torch.nn.functional as F
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


class ConvLayer(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int = 1,
                 n_group: int = 1, relu: bool = True) -> None:
        super().__init__()

        self.conv: nn.Conv2d = nn.Conv2d(c_in, c_out, kernel_size, stride=stride,
                                         padding=kernel_size >> 1, groups=n_group, bias=False)
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(c_out)
        self.relu: bool = relu

    def forward(self, x: FTType) -> FTType:
        x = self.bn.forward(self.conv.forward(x))
        return x if self.relu is None else F.relu(x, inplace=True)


class BottleNeck(nn.Module):
    def __init__(self, c_in: int, c_mid: int, c_out: int, n_group: int, stride: int = 1) -> None:
        super().__init__()
        c_group: int = n_group * c_mid

        self.conv: nn.Sequential = nn.Sequential(
            ConvLayer(c_in, c_group, 1),
            ConvLayer(c_group, c_group, 3, stride=stride, n_group=n_group),
            ConvLayer(c_group, c_out, 1, relu=False)
        )

        self.shortcut: ConvLayer | None = None if c_in == c_out else \
            ConvLayer(c_in, c_out, 1, stride=stride, relu=False)

    def forward(self, x: FTType) -> FTType:
        return F.relu((x if self.shortcut is None else self.shortcut.forward(x)) +
                      self.conv.forward(x), inplace=True)


class ConvStage(nn.Module):
    def __init__(self, c_in: int, stride: int, n_block: int, n_group: int, c_mid: int,
                 c_out: int) -> None:
        super().__init__()
        blocks: list[BottleNeck] = [BottleNeck(c_in, c_mid, c_out, n_group, stride=stride)]
        blocks.extend([BottleNeck(c_out, c_mid, c_out, n_group) for _ in range(1, n_block)])
        self.blocks: nn.Sequential = nn.Sequential(*blocks)

    def forward(self, x: FTType) -> FTType:
        return self.blocks.forward(x)


class ResNeXt(nn.Module):
    def __init__(self, c_in: int, n_class: int, stages: list[dict[str, int]]) -> None:
        super().__init__()
        self.pre: ConvLayer = ConvLayer(c_in, 64, 3)
        c_in = 64
        stride: int = 1
        main_stages: list[ConvStage] = []

        for stage in stages:
            main_stages.append(ConvStage(c_in, stride=stride, **stage))
            c_in = stage['c_out']
            stride = 2

        self.main: nn.Sequential = nn.Sequential(*main_stages)
        self.post: nn.Sequential = nn.Linear(c_in, n_class)
        self.apply(init_weight)

    def forward(self, x: FTType) -> FTType:
        x = self.main.forward(self.pre.forward(x))
        return self.post.forward(F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1))
