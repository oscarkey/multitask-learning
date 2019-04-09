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
        self._encoder = nn.Sequential(nn.Conv2d(1, 16, 3, stride=3, padding=1),  # [batch x 16 x 10 x 10]
                                      nn.ReLU(True), nn.MaxPool2d(2, stride=2),  # [batch x 16 x 5 x 5]
                                      nn.Conv2d(16, 8, 3, stride=2, padding=1),  # [batch x 8 x 3 x 3]
                                      nn.ReLU(True), nn.MaxPool2d(2, stride=1))  # [batch x 8 x 2 x 2]

    def forward(self, x):
        assert_shape(x, (28, 28))
        return self._encoder(x)

    @staticmethod
    def get_out_features():
        return 8


class Encoder2(nn.Module):
    """A larger encoder."""

    def __init__(self):
        super().__init__()
        self._conv1 = nn.Conv2d(1, 32, 5, stride=2, padding=4)
        self._conv2 = nn.Conv2d(32, 16, 3, stride=2, padding=1)
        self._conv3 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

    def forward(self, x):
        assert_shape(x, (28, 28))

        x = F.relu(self._conv1(x))
        assert_shape(x, (16, 16))
        x = F.max_pool2d(x, 2, stride=2)
        assert_shape(x, (8, 8))

        x = F.relu(self._conv2(x))
        assert_shape(x, (4, 4))

        x = F.relu(self._conv3(x))
        assert_shape(x, (4, 4))
        x = F.max_pool2d(x, 2, stride=2)
        assert_shape(x, (2, 2))

        return x

    @staticmethod
    def get_out_features():
        return 16


class Encoder3(nn.Module):
    """A much larger encoder."""

    def __init__(self):
        super().__init__()
        self._conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self._conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self._conv3 = nn.Conv2d(32, 32, 2, stride=1, padding=0)
        self._conv4 = nn.Conv2d(32, 16, 2, stride=1, padding=0)
        self._conv5 = nn.Conv2d(16, 16, 2, stride=2, padding=1)

    def forward(self, x):
        assert_shape(x, (28, 28))

        x = F.relu(self._conv1(x))
        assert_shape(x, (28, 28))
        x = F.relu(self._conv2(x))
        assert_shape(x, (28, 28))
        x = F.max_pool2d(x, 2, stride=2)
        assert_shape(x, (14, 14))

        x = F.relu(self._conv3(x))
        assert_shape(x, (13, 13))
        x = F.relu(self._conv4(x))
        assert_shape(x, (12, 12))
        x = F.max_pool2d(x, 2, stride=2)
        assert_shape(x, (6, 6))

        x = F.relu(self._conv5(x))
        assert_shape(x, (4, 4))
        x = F.max_pool2d(x, 2, stride=2)
        assert_shape(x, (2, 2))

        return x

    @staticmethod
    def get_out_features():
        return 16


class EncoderFC(nn.Module):
    """A fully connected encoder."""

    def __init__(self):
        super().__init__()
        self._layers = nn.Sequential(nn.Linear(28 * 28, 512),  #
                                     nn.ReLU(),  #
                                     nn.Linear(512, 256),  #
                                     nn.ReLU(),  #
                                     nn.Linear(256, 32),  #
                                     nn.ReLU())

    def forward(self, x):
        assert_shape(x, (28, 28))
        x = x.view(-1, 28 * 28)
        x = self._layers(x)
        # Return in shape (2, 2) so as to be compatible with the other decoders.
        return x.view(-1, 8, 2, 2)

    @staticmethod
    def get_out_features():
        return 8


class Classifier(nn.Module):
    def __init__(self, num_classes: int, in_features: int):
        super().__init__()
        self._fc1 = nn.Linear(in_features=2 * 2 * in_features, out_features=128)
        self._fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self._fc1(x))
        x = self._fc2(x)
        return x


class Classifier2(nn.Module):
    """A larger fully connected classifier."""
    def __init__(self, num_classes: int, in_features: int):
        super().__init__()
        self._layers = nn.Sequential(nn.Linear(in_features=2 * 2 * in_features, out_features=128),  #
                                     nn.ReLU(),  #
                                     nn.Linear(in_features=128, out_features=128),  #
                                     nn.ReLU(),  #
                                     nn.Linear(in_features=128, out_features=num_classes))

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self._layers(x)
        return x


class Reconstructor(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self._decoder = nn.Sequential(nn.ConvTranspose2d(in_features, 16, 3, stride=2),  # b, 16, 5, 5
                                      nn.ReLU(True),  #
                                      nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
                                      nn.ReLU(True),  #
                                      nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
                                      nn.Tanh())

    def forward(self, x):
        x = self._decoder(x)
        assert_shape(x, (28, 28))
        return x


class Reconstructor2(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self._convt1 = nn.ConvTranspose2d(in_features, 16, 2, stride=2)
        self._convt2 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1)
        self._convt3 = nn.ConvTranspose2d(8, 4, 2, stride=2)
        self._convt4 = nn.ConvTranspose2d(4, 1, 2, stride=2)

    def forward(self, x):
        x = F.relu(self._convt1(x))
        assert_shape(x, (4, 4))
        x = F.relu(self._convt2(x))
        assert_shape(x, (7, 7))
        x = F.relu(self._convt3(x))
        assert_shape(x, (14, 14))
        x = F.relu(self._convt4(x))
        assert_shape(x, (28, 28))
        return torch.tanh(x)


class ReconstructorFC(nn.Module):
    """A fully connected reconstruction model."""

    def __init__(self, in_features: int):
        super().__init__()
        self._in_features = in_features
        self._layers = nn.Sequential(nn.Linear(in_features * 2 * 2, 1024),  #
                                     nn.ReLU(),  #
                                     nn.Linear(1024, 800),  #
                                     nn.ReLU(),  #
                                     nn.Linear(800, 28 * 28),  #
                                     nn.Tanh())

    def forward(self, x):
        x = x.view(-1, self._in_features * 2 * 2)
        x = self._layers(x)
        return x.view(-1, 28, 28)


_models = [(Encoder, Classifier, Reconstructor),  #
           None,  # Model 2 is no longer implemented.
           (Encoder2, Classifier, Reconstructor),  #
           (Encoder3, Classifier, Reconstructor2),  #
           None,  # 4 is no longer implemented, see commit da9d7c9.
           (EncoderFC, Classifier, ReconstructorFC),  #
           (Encoder3, Classifier2, Reconstructor2)]


class MultitaskMnistModel(nn.Module):
    def __init__(self, initial_ses: [float], model_version: int):
        super().__init__()

        encoder_con, classifier_con, reconstructor_con = _models[model_version]
        self._encoder = encoder_con()
        self._classifier1 = classifier_con(num_classes=3, in_features=self._encoder.get_out_features())
        self._classifier2 = classifier_con(num_classes=10, in_features=self._encoder.get_out_features())
        self._reconstructor = reconstructor_con(in_features=self._encoder.get_out_features())

        assert len(initial_ses) == 3
        self._weight1 = nn.Parameter(torch.tensor([initial_ses[0]]))
        self._weight2 = nn.Parameter(torch.tensor([initial_ses[1]]))
        self._weight3 = nn.Parameter(torch.tensor([initial_ses[2]]))

    def forward(self, x):
        assert_shape(x, (28, 28))

        x = self._encoder(x)
        x1 = self._classifier1(x)
        x2 = self._classifier2(x)
        x3 = self._reconstructor(x)
        return x1, x2, x3

    def get_loss_weights(self) -> (nn.Parameter, nn.Parameter, nn.Parameter):
        """Returns the loss weight parameters (s in the paper)."""
        return self._weight1, self._weight2, self._weight3
