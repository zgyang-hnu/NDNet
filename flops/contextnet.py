#author zhengeng yang


import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

from backbone import StdConvBR, SepConvBR

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class context(nn.Module):
    def __init__(self,  width_mult=1.):
        super(context, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 128
        interverted_residual_setting = [
            # t, c, n, s
            [1, 32, 1, 1],
            [6, 32, 1, 1],
            [6, 48, 3, 2],
            [6, 64, 3, 2],
            [6, 96, 2, 1],
            [6, 128, 2, 1]
        ]

        # building first layer
        #assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last  layer
        self.features.append(conv_1x1_bn(128, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)


        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



class spatial(nn.Module):
      def __init__(self):
        super(spatial, self).__init__()
        self.first_layer = conv_bn(3, 32, 2)
        self.sec_layer = SepConvBR(32,64,stride=2)
        self.third_layer = SepConvBR(64,128,stride=2)
        self.fourth_layer = SepConvBR(128,128,stride=1)

      def forward(self, x):
          x= self.first_layer(x)
          x= self.sec_layer(x)
          x= self.third_layer(x)
          x= self.fourth_layer(x)
          return x


class contextnet(nn.Module):
      def __init__(self, n_classes=19, out_stride=8):
        super(contextnet, self).__init__()
        self.context = context()
        self.context_trans = nn.ModuleList([SepConvBR(128,128,stride=1,padding=4, dilation=4),conv_1x1_bn(128,128)])
        self.spatial = spatial()
        self.spatial_trans = conv_1x1_bn(128,128)
        self.scores = nn.Conv2d(128, n_classes, 1, 1, 0, bias=True)
        self.classes = n_classes
        self.out_stride = out_stride
     
      def forward(self, x):
          #the full resolution spatial branch
          spatial = self.spatial(x)
          spatial = self.spatial_trans(spatial)

          #the 1/4 resolution context branch
          target_size = np.array(list(x.size())[-2:])//4
          #print(target_size)
         
          x14 = F.interpolate(x, size=tuple(target_size),
                                    mode='bilinear', align_corners=True)
          context= self.context(x14)
          #print(context.size())
          #print(spatial.size())
          context = F.interpolate(context, size=(spatial.size()[2:]),
                                    mode='bilinear', align_corners=True)
          context = self.context_trans[0](context)
          context = self.context_trans[1](context)
          
          #fusion
          lastf = spatial+context
          
          scores = self.scores(lastf)
          scores = F.interpolate(scores, scale_factor=self.out_stride,
                                   mode='bilinear',
                                   align_corners=True)
          
          return F.log_softmax(scores, dim=1)          



    

