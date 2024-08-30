import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None, *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class ResNet(nn.Module):
    
    def __init__(self, number_element, block, layers, num_classes=121):
        super(ResNet, self).__init__()
        self.inplanes = 72
        self.M = number_element

        self.conv1 = nn.Conv1d(self.M * 2, 72, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(72)
        self.conv2 = nn.Conv1d(self.M * 2, 72, kernel_size=5, stride=1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(72)
        self.conv3 = nn.Conv1d(self.M * 2, 72, kernel_size=7, stride=1, padding='same', bias=False)
        self.bn3 = nn.BatchNorm1d(72)
        self.conv1_1 = nn.Conv1d(72, 24, kernel_size=1, bias=False)
        self.se = SELayer(24 * 3)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(1024 * block.expansion, num_classes),
            Hswish(inplace=True),
            nn.Linear(num_classes, num_classes)
            )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input):
        
        bts, M, L = input.shape
        input = torch.view_as_real(input).float()  # (btz, M, L) -> (btz, M, L, 2)
        input = input.permute(0, 1, 3, 2)  # (btz, M, L, 2) -> (btz, M, 2, L)
        x = input.contiguous().view(bts, M * 2, L )  # (btz, M, 2, L) -> (btz, M * 2, L)

        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv1_1(x1)

        x2 = self.conv1(x)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.conv1_1(x2)

        x3 = self.conv1(x)
        x3 = self.bn1(x3)
        x3 = self.relu(x3)
        x3 = self.conv1_1(x3)

        x = torch.cat((x1, x2, x3), 1)
        x = self.se(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def MSEDOANet(number_element, num_classes,**kwargs):
    model = ResNet(number_element, SEBasicBlock, [3, 4, 6, 3], num_classes, **kwargs)
    return model

