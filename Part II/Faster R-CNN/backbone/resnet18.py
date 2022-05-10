from typing import Tuple

import torchvision
from torch import nn

import backbone.base


class ResNet18(backbone.base.Base):

    def __init__(self, pretrained: bool):
        super().__init__(pretrained)

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        resnet18 = torchvision.models.resnet18(pretrained=self._pretrained)

        # list(resnet18.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        children = list(resnet18.children())
        features = children[:-3]
        num_features_out = 256

        hidden = children[-3]
        num_hidden_out = 512

        for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
            for parameter in parameters:
                parameter.requires_grad = False

        features = nn.Sequential(*features)

        return features, hidden, num_features_out, num_hidden_out
