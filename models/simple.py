import math, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class ConvolutionGenerator(nn.Module):
    def __init__(self, upsample=3):
        """
        Start from 4x4, upsample=3 -> 32
        """
        super(ConvolutionGenerator, self).__init__()
        dims = [64 * (2**i) for i in range(upsample+1)][::-1]
        self.dims = dims

        self.fc = nn.Linear(128, 4 * 4 * dims[0])
        self.relu = nn.ReLU(True)
        self.deconvs = nn.ModuleList()
        for prevDim, curDim in zip(dims[:-1], dims[1:]):
            conv = nn.Sequential()
            conv.deconv = nn.ConvTranspose2d(prevDim, curDim, 4, stride=2, padding=1)
            conv.bn = nn.BatchNorm2d(curDim)
            conv.relu = nn.ReLU(True)
            self.deconvs.append(conv)
        self.visualize = nn.Conv2d(dims[-1], 3, 3, padding=1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.relu(self.fc(x)).view(-1, self.dims[0], 4, 4)
        for layers in self.deconvs: x = layers(x)
        x = self.tanh(self.visualize(x))
        return x

class ConvolutionDiscriminator(nn.Module):
    def __init__(self, upsample=3):
        super(ConvolutionDiscriminator, self).__init__()
        dims = [64 * (2**i) for i in range(upsample+1)]
        self.dims = dims

        self.conv = nn.Conv2d(3, dims[0], 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.convs = nn.ModuleList()
        for prevDim, curDim in zip(dims[:-1], dims[1:]):
            conv = nn.Sequential()
            conv.conv = nn.Conv2d(prevDim, curDim, 4, stride=2, padding=1)
            conv.lrelu = nn.LeakyReLU(0.2, True)
            self.convs.append(conv)
        self.fc = nn.Linear(4 * 4 * dims[-1], 1)

    def forward(self, x):
        x = self.lrelu(self.conv(x))
        for layers in self.convs: x = layers(x)
        x = self.fc(x.view(-1, 4 * 4 * self.dims[-1]))
        return x

class UpsampleGenerator(nn.Module):
    def __init__(self, upsample=3):
        """
        Start from 4x4, upsample=3 -> 32
        """
        super(UpsampleGenerator, self).__init__()
        dims = [64 * (2**i) for i in range(upsample+1)][::-1]
        self.dims = dims

        self.fc = nn.Linear(128, 4 * 4 * dims[0])
        self.fc_bn = nn.BatchNorm2d(dims[0])
        self.relu = nn.ReLU(True)
        self.deconvs = nn.ModuleList()
        for prevDim, curDim in zip(dims[:-1], dims[1:]):
            conv = nn.Sequential()
            conv.upsample = nn.Upsample(scale_factor=2)
            conv.conv = nn.Conv2d(prevDim, curDim, 3, padding=1)
            conv.bn = nn.BatchNorm2d(curDim)
            conv.relu = nn.ReLU(True)
            self.deconvs.append(conv)
        self.visualize = nn.Conv2d(dims[-1], 3, 3, padding=1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.fc(x).view(-1, self.dims[0], 4, 4)
        x = self.relu(self.fc_bn(x))
        for layers in self.deconvs: x = layers(x)
        x = self.tanh(self.visualize(x))
        return x

class DownsampleDiscriminator(nn.Module):
    def __init__(self, downsample=3):
        super(DownsampleDiscriminator, self).__init__()
        dims = [64 * (2**i) for i in range(downsample+1)]
        self.dims = dims

        self.conv = nn.Conv2d(3, dims[0], 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.convs = nn.ModuleList()
        for prevDim, curDim in zip(dims[:-1], dims[1:]):
            conv = nn.Sequential()
            conv.conv = nn.Conv2d(prevDim, curDim, 3, padding=1)
            conv.downsample = nn.AvgPool2d(2, 2)
            conv.lrelu = nn.LeakyReLU(0.2, True)
            self.convs.append(conv)
        self.fc = nn.Linear(4 * 4 * dims[-1], 1)

    def forward(self, x):
        x = self.lrelu(self.conv(x))
        for layers in self.convs: x = layers(x)
        x = self.fc(x.view(-1, 4 * 4 * self.dims[-1]))
        return x