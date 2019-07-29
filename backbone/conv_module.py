import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def dwconv33(in_channels, stride=1, padding=1,dilation=1):
    return nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, groups=in_channels, bias=False)

def stdconv33(in_channels, out_channels, stride=1, padding=1,dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, bias=False)

def pwconv11(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                     padding=0, dilation=1, bias=False)

def stdconv(in_channels, out_channels, ksize, stride=1, padding=1,dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride,
                     padding=padding, dilation=dilation, bias=False)



class StdConvBR(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, padding, dilation=1,
                  bn_eps=1e-5, inplace=True, bn=True, relu=True, bias=False):
        super(StdConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=1, bias=bias)
        self.BN = bn
        self.RELU = relu
        if bn:
           self.bn = nn.BatchNorm2d(out_channels, eps=bn_eps)
        if relu:
           self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.BN:
          x = self.bn(x)
        if self.RELU:
          x = self.relu(x)

        return x


class SepConvBR(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, padding=1, dilation=1,
                  bn_eps=1e-5, inplace=True, bn=True, relu=True, bias=False):
        super(SepConvBR, self).__init__()

        self.dwconv = dwconv33(in_channels, stride=stride, padding=padding, dilation=dilation)
        self.pwconv = pwconv11(in_channels, out_channels)

        self.BN = bn
        self.RELU = relu

        if bn:
          self.bn = nn.BatchNorm2d(out_channels, eps=bn_eps)
        if relu:
          self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        if self.BN:
           x = self.bn(x)
        if self.RELU:
           x = self.relu(x)

        return x




"""
Custom ERFNet building blocks
Adapted from: https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
"""




class non_bottleneck_1d(nn.Module):
    def __init__(self, n_channel, drop_rate, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(n_channel, n_channel, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(n_channel, n_channel, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.conv3x1_2 = nn.Conv2d(n_channel, n_channel, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(n_channel, n_channel, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn1 = nn.BatchNorm2d(n_channel, eps=1e-03)
        self.bn2 = nn.BatchNorm2d(n_channel, eps=1e-03)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv3x1_2(output)
        output = self.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return self.relu(output + input)  # +input = identity (residual connection)


class DownsamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel - in_channel, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return self.relu(output)


class UpsamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return self.relu(output)

