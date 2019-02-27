from torch import nn
from torch.nn import Module
import torch.nn.functional as F


def _build_base_decoder():
    """Builds the base decoder shared by all three decoder types."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels=1280,
            out_channels=256,
            kernel_size=(1, 1),
            stride=1),
        nn.BatchNorm2d(num_features=256),
        nn.ReLU()
    )


class Decoders(Module):
    """Module which contains all three decoders."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.base_semseg = _build_base_decoder()
        self.base_insseg = _build_base_decoder()
        self.base_depth = _build_base_decoder()

        kernel_size = (1, 1)
        self.semsegcls = nn.Conv2d(256, self.num_classes, kernel_size)
        self.inssegcls = nn.Conv2d(256, 2, kernel_size)
        self.depthcls = nn.Conv2d(256, 1, kernel_size)

    def forward(self, x):
        """Returns (sem seg, instance seg, depth)."""
        # x: [batch x 1280 x H x W]
        batch_size = x.shape[0]
        x1 = self.base_semseg(x)
        x1 = self.semsegcls(x1)
        x1 = F.interpolate(x1, size=(128, 256), mode='bilinear', align_corners=False)

        x2 = self.base_insseg(x)
        x2 = self.inssegcls(x2)
        x2 = F.interpolate(x2, size=(128, 256), mode='bilinear', align_corners=False)

        x3 = self.base_depth(x)
        x3 = self.depthcls(x3)
        x3 = F.interpolate(x3, size=(128, 256), mode='bilinear', align_corners=False)

        return x1, x2, x3
