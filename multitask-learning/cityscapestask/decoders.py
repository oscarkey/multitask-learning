"""Decoder portion of the model."""

import torch
import torch.nn.functional as F
from torch import nn


def _build_base_decoder():
    """Builds the base decoder shared by all three decoder types."""
    return nn.Sequential(nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
                         nn.BatchNorm2d(num_features=256), nn.ReLU())


class Decoders(nn.Module):
    """Module which contains all three decoders."""

    def __init__(self, num_classes: int, enabled_tasks: (bool, bool, bool), output_size=(128, 256)):
        super().__init__()
        self._output_size = output_size
        self._num_classes = num_classes
        self._enabled_tasks = enabled_tasks

        self._base_semseg = _build_base_decoder()
        self._base_insseg = _build_base_decoder()
        self._base_depth = _build_base_decoder()

        kernel_size = (1, 1)
        self._semsegcls = nn.Conv2d(256, self._num_classes, kernel_size)
        self._inssegcls = nn.Conv2d(256, 2, kernel_size)
        self._depthcls = nn.Conv2d(256, 1, kernel_size)

    def set_output_size(self, size):
        self._output_size = size

    def forward(self, x):
        """Returns (sem seg, instance seg, depth)."""
        # x: [batch x 1280 x H/8 x W/8]

        sem_seg_enabled, inst_seg_enabled, depth_enabled = self._enabled_tasks

        if sem_seg_enabled:
            x1 = self._base_semseg(x)
            x1 = self._semsegcls(x1)
            x1 = F.interpolate(x1, size=self._output_size, mode='bilinear', align_corners=True)
        else:
            x1 = None

        if inst_seg_enabled:
            x2 = self._base_insseg(x)
            x2 = self._inssegcls(x2)
            x2 = F.interpolate(x2, size=self._output_size, mode='bilinear', align_corners=True)
        else:
            x2 = None

        if depth_enabled:
            x3 = self._base_depth(x)
            x3 = self._depthcls(x3)
            x3 = F.interpolate(x3, size=self._output_size, mode='bilinear', align_corners=True)
        else:
            x3 = None

        return x1, x2, x3


if __name__ == '__main__':
    # ### Shape test
    output_size = (123, 432)
    model = Decoders(num_classes=20, output_size=output_size)
    test = torch.zeros(size=(2, 1280, 256, 256))
    result = model.forward(test)
    assert result[0].shape == (2, 20, *output_size), "output shape is {}".format(result[0].shape)
