# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


# code in this cell mostly from torchvision/models/resnet.py

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                     bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AtrousBottleneck(nn.Module):
    """Bottleneck ResNet Module, with the added option to use atrous (dilated) convolution
    for the 3x3 convolution, given by the dilation parameter.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(AtrousBottleneck, self).__init__()
        self.dilation = dilation
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module as described for DeeplabV3 with output_stride = 8

    It applies, in parallel: one 1x1 convolution, three 3x3 convolutions with dilation = 12,24,36. All have
    out_channels=256. These are concatenated with the feature map convolved down to 256 channels by a 1x1 convolution.
    """

    def __init__(self, dilations: (int, int, int)):
        super().__init__()

        assert len(dilations) == 3
        assert all([dilation > 0 for dilation in dilations])

        self.conv1 = conv1x1(2048, 256)
        self.conv2 = conv3x3(2048, 256, dilation=dilations[0])
        self.conv3 = conv3x3(2048, 256, dilation=dilations[1])
        self.conv4 = conv3x3(2048, 256, dilation=dilations[2])

        # Operations for last feature map
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = conv1x1(2048, 256)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)

    def forward(self, x):
        # x is feature map
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(x)))
        out3 = F.relu(self.bn3(self.conv3(x)))
        out4 = F.relu(self.bn4(self.conv4(x)))

        out5 = F.relu(self.bn5(self.conv(self.gap(x))))
        out5 = F.interpolate(out5, size=x.shape[-2:], mode="bilinear", align_corners=True)

        out = torch.cat((out1, out2, out3, out4, out5), dim=1)
        return out


class Encoder(nn.Module):
    """
        https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html
        https://mc.ai/resnet-torchvision-bottlenecks-and-layers-not-as-they-seem/
        https://github.com/fregu856/deeplabv3

    """

    def __init__(self, aspp_dilations: (int, int, int)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(AtrousBottleneck, 64, 3)
        self.layer2 = self._make_layer(AtrousBottleneck, 128, 4, stride=2)

        # Dilation choices of 2 and 4
        self.layer3 = self._make_layer(AtrousBottleneck, 256, 23, stride=1, dilation=2)
        self.layer4 = self._make_layer(AtrousBottleneck, 512, 3, stride=1, dilation=4)
        self.aspp = ASPP(aspp_dilations)

    # from torchvision.models.resnet.ResNet
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None

        # If a stride=2 is passed to the block, input doesn't match the output
        # We need to downsample so we can add them together
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        return x


if __name__ == '__main__':
    # ### Shape test
    model = Encoder()
    test = torch.zeros(size=(2, 3, 256, 256))
    result = model.forward(test)
    size = test.shape
    assert result.shape == (2, 1280, test.shape[-2] // 8, test.shape[-1] // 8), "output shape is {} {} {} {}".format()
