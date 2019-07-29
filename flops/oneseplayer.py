import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np

import sys
sys.path.append('..')

from backbone import StdConvBR, SepConvBR


class oneseplayer(nn.Module):
    def __init__(self, out_channels, criterion=None, 
                 aux_loss=True, out_stride=32):
        super(oneseplayer, self).__init__()
        self.features = SepConvBR(12, 24, 3, 1, 1,
                                       bn=False, relu=False)

    def forward(self, x, label=None):
        x = self.features(x)
        return x 
