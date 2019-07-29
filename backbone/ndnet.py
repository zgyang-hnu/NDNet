#this model is developed from the resnet code in pytorch
import numpy as np

import torch
import torch.nn as nn

from .conv_module import StdConvBR, SepConvBR

class Narrow_BottleNeck(nn.Module):
      expansion = 4
      def __init__(self, in_channels, mid_channels, stride=1, padding=1, dilation=1, iden_mapping=True):
         super(Narrow_BottleNeck, self).__init__()
         self.iden_mapping = iden_mapping
         self.conv1 = SepConvBR(in_channels, mid_channels, stride=stride, 
                                    padding=dilation, dilation=dilation)
         self.conv2 = SepConvBR(mid_channels, mid_channels*self.expansion, stride=1, 
                                    padding=dilation, dilation=dilation, relu=False)
         if not self.iden_mapping:
            self.shortcut = SepConvBR(in_channels, mid_channels*self.expansion, stride=stride, 
                                      padding=dilation, dilation=dilation, relu=False)
         self.relu_inplace = nn.ReLU(inplace=True)

      def forward(self, x):
          shortcut = x
          res=self.conv1(x)
          res=self.conv2(res)
          if not self.iden_mapping:
             shortcut = self.shortcut(x)
          return self.relu_inplace(res + shortcut)
          
             
 
class NDNet(nn.Module):
     expansion = 4
     def __init__(self, bottleneck, 
                        in_channels, 
                        conv1_outchannels, 
                        midchannel_comb, 
                        depth_comb, 
                        out_stride=32):
        super(NDNet, self).__init__()

        self.out_stride = out_stride
        assert self.out_stride in [8,16,32]
        if self.out_stride==8:
           self.last_two_stride = [1,1]
           self.last_two_dilation = [2,4]
        elif self.out_stride==16:
           self.last_two_stride = [2,1]
           self.last_two_dilation = [1,2]
        else:
           self.last_two_stride = [2,2]
           self.last_two_dilation = [1,1]

        self.conv1 = StdConvBR(in_channels, conv1_outchannels, ksize=3, padding=1, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # contrcut block using ''bottleneck'' class
        self.block3 = self._make_block(bottleneck, conv1_outchannels, 
                                       depth_comb[0], 
                                       midchannel_comb[0], 
                                       stride=2)

        self.block4 = self._make_block(bottleneck, midchannel_comb[0]*self.expansion, 
                                       depth_comb[1], 
                                       midchannel_comb[1], 
                                       stride=self.last_two_stride[0],
                                       dilation=self.last_two_dilation[0])

        self.block5 = self._make_block(bottleneck, midchannel_comb[1]*self.expansion, 
                                       depth_comb[2], 
                                       midchannel_comb[2], 
                                       stride=self.last_two_stride[1],
                                       dilation=self.last_two_dilation[1])


     def _make_block(self, 
                     bottleneck, 
                     in_channels, 
                     depth, 
                     mid_channels, 
                     stride=1, 
                     dilation=1):
        layers = []
        iden_mapping = False if stride > 1 or dilation>1 else True
        layers.append(bottleneck(in_channels, mid_channels, stride=stride,  dilation=dilation, 
                                 iden_mapping=iden_mapping))

        in_channels = mid_channels * self.expansion
        for i in range(1, depth):#
            layers.append(bottleneck(in_channels, mid_channels, stride=1, dilation=dilation,
                                     iden_mapping=True))

        return nn.Sequential(*layers)

     def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        blocks = []
        x = self.block3(x);
        blocks.append(x)
        x = self.block4(x);
        blocks.append(x)
        x = self.block5(x);
        blocks.append(x)

        return blocks


def NDNet45(out_stride=32, **kwargs):
    #if out_stride==32:
  
    model = NDNet(Narrow_BottleNeck, 3, 8, [16, 32, 64], [4, 12, 6], out_stride=out_stride, **kwargs)


    return model


def NDNet61(out_stride=32, **kwargs):
    #if out_stride==32:
  
    model = NDNet(Narrow_BottleNeck, 3, 8, [12, 24, 48], [6, 16, 8], out_stride=out_stride, **kwargs)


    return model

def NDNet29(out_stride=32, **kwargs):
    #if out_stride==32:
  
    model = NDNet(Narrow_BottleNeck, 3, 8, [24, 48, 96], [3, 8, 3], out_stride=out_stride, **kwargs)


    return model

def NDNet29w(out_stride=32, **kwargs):
    #if out_stride==32:
  
    model = NDNet(Narrow_BottleNeck, 3, 8, [64, 128, 256], [3, 8, 3], out_stride=out_stride, **kwargs)


    return model

#if __name__ == "__main__":
#   model=NDNet45()
#   print(model)
