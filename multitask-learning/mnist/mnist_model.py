"""The model for the MNIST variant of the multitask experiment."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def assert_shape(x: Tensor, shape: (int, int)):
    """Raises an exception if the Tensor doesn't have the given final two dimensions."""
    assert tuple(x.shape[-2:]) == tuple(shape), f'Expected shape ending {shape}, got {x.shape}'


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=12, padding=0, stride=2)
        self._conv2 = nn.Conv2d(in_channels=25, out_channels=64, kernel_size=5, padding=2)

    def forward(self, x):
        assert_shape(x, (28, 28))

        x = F.relu(self._conv1(x))
        assert_shape(x, (9, 9))

        x = F.relu(self._conv2(x))
        assert_shape(x, (9, 9))

        x = F.max_pool2d(x, 2)
        assert_shape(x, (4, 4))

        return x


class Decoder(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self._fc1 = nn.Linear(in_features=4 * 4 * 64, out_features=1024)
        self._fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        assert_shape(x, (4, 4))

        x = x.view(x.shape[0], -1)
        x = F.relu(self._fc1(x))
        x = self._fc2(x)
        return x


class MultitaskMnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._encoder = Encoder()
        self._decoder1 = Decoder(num_classes=3)
        self._decoder2 = Decoder(num_classes=10)

        self._weight1 = nn.Parameter(torch.tensor([1.0]))
        self._weight2 = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        assert_shape(x, (28, 28))

        x = self._encoder(x)
        x1 = self._decoder1(x)
        x2 = self._decoder2(x)
        return x1, x2

    def get_loss_weights(self) -> (nn.Parameter, nn.Parameter):
        return self._weight1, self._weight2
