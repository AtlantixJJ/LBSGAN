import math, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import math

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    """
    Leaky ReLU
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None, activation="lrelu", expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        if activation == "lrelu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.act(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.act(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class Bottleneck2(nn.Module):
    """
    Leaky ReLU
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None, activation="lrelu", expansion=4):
        super(Bottleneck2, self).__init__()
        self.expansion = expansion
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes // 2)
        self.conv2 = nn.Conv2d(planes // 2, planes // 2, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes // 2)
        self.conv3 = nn.Conv2d(planes // 2, planes, kernel_size=1, bias=False)
        if activation == "lrelu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.act(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.act(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class ResNetBase(nn.Module):
    def __init__(self, **kwargs):
        super(ResNetBase, self).__init__()
        self.inplanes = 64

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, act="lrelu", expansion=4):
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, act, expansion))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, act, expansion))

        return nn.Sequential(*layers)

class ResNetGen(ResNetBase):
    def __init__(self, block=Bottleneck2, layers=[3, 3, 3, 3], **kwargs):
        super(ResNetGen, self).__init__()

        self.feature_divergence = 0

        self.fc = nn.Linear(128, 4 * 4 * 1024)
        self.bn = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.inplanes = 512
        self.layer1 = self._make_layer(block, 256, layers[0], stride=1, act="relu", expansion=1)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=1, act="relu", expansion=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, act="relu", expansion=1)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1, act="relu", expansion=1)
        
        self.conv2 = nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        self.init_weight()
    
    def forward(self, x):
        x = self.fc(x).view(x.size(0), 1024, 4, 4)
        x = self.relu(self.bn(x))

        x = self.relu(self.bn1(self.conv1(x))) # 64x64
        x = F.upsample_nearest((self.layer1(x)), scale_factor=2)
        x = F.upsample_nearest((self.layer2(x)), scale_factor=2)
        x = F.upsample_nearest((self.layer3(x)), scale_factor=2)
        x = F.upsample_nearest((self.layer4(x)), scale_factor=2)
        
        x = self.tanh(self.conv2(x))
        return x


class ResNetDisc(ResNetBase):
    def __init__(self, block=Bottleneck, layers=[3, 3, 3, 3], **kwargs):
        super(ResNetDisc, self).__init__()
        self.inplanes = 64
        self.feature_divergence = 0
        self.dims = [64, 128, 256, 512, 1024]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=2, act="lrelu")
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, act="lrelu")
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, act="lrelu")
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, act="lrelu")
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(2048, 1, 1)

        self.init_weight()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc(x)
        return x




