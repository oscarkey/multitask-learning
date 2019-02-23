# coding: utf-8
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# code in this cell mostly from torchvision/models/resnet.py

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                 dilation=dilation, padding=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AtrousBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  downsample=None, dilation=1):
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
    


# ### ASPP

class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()
        self.conv1 = conv1x1(2048, 256)
        self.conv2 = conv3x3(2048, 256, dilation=12)
        self.conv3 = conv3x3(2048, 256, dilation=24)
        self.conv4 = conv3x3(2048, 256, dilation=36)
        
        # Operations for last feature map
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = conv1x1(2048, 256)
        self.bn = nn.BatchNorm2d(256)
    
    def forward(self, x):
        # x is feature map
        out1 = F.relu(self.bn(self.conv1(x)))
        out2 = F.relu(self.bn(self.conv2(x)))
        out3 = F.relu(self.bn(self.conv3(x)))
        out4 = F.relu(self.bn(self.conv4(x)))

        out5 = F.relu(self.bn(self.conv(self.gap(x))))
        out5 = F.interpolate(out5, size=x.shape[-2:], mode="bilinear", align_corners = False)                
                       
        out = torch.cat([out1,out2,out3,out4, out5], 1)
        return out
        



class Encoder(nn.Module):
    """
        Bottleneck(nn.Module)

        def __init__(self, )

        1x1, 2
        batch norm, ReLU
        3x3,
        batch norm, ReLU
        1x1
        batch norm(?)
        add input to output
        batch norm(?), relu

        MODEL

        BLOCK 0, OS = 4
        7x7, 64, stride 2
        Batch norm, ReLU
        3x3 max pool, stride 2
        Batch norm, ReLU


        BLOCK 1, OS = 4
        3 x bottlenecks:
        1x1, 64 stride = 1
        3x3, 64 stride = 1
        1x1, 256 stride = 1

        BLOCK 2, OS = 8
        4 x bottlenecks:
        1x1, 128 stride = 1
        3x3, 128 stride = 1
        1x1, 512 stride = 1 (* stride = 2 for the last block)

        BLOCK 3, OS = 8
        23 x bottlenecks:
        1x1, 256 stride = 1
        3x3, 256 stride = 1 dilation = 2 padding = dilation
        1x1, 1024 stride = 1

        BLOCK 4, OS = 8
        3 x bottlenecks:
        1x1, 512 stride = 1
        3x3, 512 stride = 1 dilation = 2 padding = dilation
        1x1, 2048 stride = 1

        ASPP, OS = 8
        4 x PARALLEL convolutional layers
        1x1, 256 stride = 1 dilation = 1 padding = dilation
        3x3, 256 dilation = 12
        3x3, 256 dilation = 24
        3x3, 256 dilation = 36
            FOR EACH LAYER
            Global average pooling
            1x1, 256
            Batch normalisation

        Concatenation

        https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html
        https://mc.ai/resnet-torchvision-bottlenecks-and-layers-not-as-they-seem/
        https://github.com/fregu856/deeplabv3

    """

    def __init__(self):
        super(Encoder, self).__init__()
        rn101 = models.resnet101()
        self.truncated_rn101 = nn.Sequential(*rn101.children())[:-4]
        
        self.inplanes = 512
        
        # Dilation choices of 2 and 4
        self.layer3 = self._make_layer(AtrousBottleneck, 256, 23, stride=1, dilation=2)
        self.layer4 = self._make_layer(AtrousBottleneck, 512, 3, stride=1, dilation=4)
        self.aspp = ASPP()
        
    # from torchvision.models.resnet.ResNet
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        
        # If a stride=2 is passed to the block, input doesn't match the output
        # We need to downsample so we can add them together
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.truncated_rn101(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        return x



if __name__ == '__main__':
# ### Shape test
    model = Encoder()
    test = torch.zeros(size=(2,3,256,512))
    result = model.forward(test)
    assert(result.shape == (2,1280,32,64))



