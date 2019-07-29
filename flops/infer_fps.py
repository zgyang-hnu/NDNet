"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import scipy.io as matio

from PIL import Image

import torch
import numpy as np


import torch.nn as nn


from erfnet import *

from enet import *
from NDNet_ende import *
from NDNet_fcn32 import *
from espnetv2 import *

from icnet import *
from segnet import *

from contextnet import *

from MobileNetV2_fcn import *




               
    
def normalize(img, mean, std):
    # pytorch pretrained model need the input range: 0-1
    #img = img - mean
    img = img.astype(np.float32) / 255.0
    img = img - mean
    # img = img / std

    return img








def main():
    """Create the model and start the evaluation process."""
    #args = get_arguments()
    
    #model = ERFNet( 19)#

    model = NDNet_ende(19)
    #model = NDNet_fcn32(19)
    #model = MobileNetV2(19)
    #model = contextnet(19)
    #model = ENet(19)
    #model = icnet(19)#icenet use 513 1025 or 1025 2049 for test

    #model = segnet(19)#icenet use 513 1025 or 1025 2049 for test
    #model = EESPNet_Seg(19).cuda()
    #

    print(model)
    model.eval()#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    duration=0

    i=0
  
    with torch.no_grad():
     for i in range(101):

       _3d_image=np.zeros([1024,2048,3])#icenet use 513 1025 or 1025 2049 for test


       _4d_image_array=np.expand_dims(_3d_image,0) 
       #_4d_image_array=normalize(_4d_image_array,)
       img = torch.from_numpy(_4d_image_array).float().permute(0,3,1,2).contiguous()
       img = img.cuda(non_blocking=True)
       start_time = time.time()
       scores=model(img)
       prediction=torch.max(scores[0].permute(1,2,0),dim=2)[1]#.cpu().detach().numpy()
       torch.cuda.synchronize()
       print('time elapsed%f' % (time.time()-start_time))
       if i==0:
          #pass
          print('shape of prediction:', prediction.shape)
       if i>0:
         duration=duration+(time.time()-start_time)
     print('average time:', duration/100)

       
       

       

     
    
if __name__ == '__main__':
    main()


#.....................................NDNET
#1024*2018 average time: 0.02080930122221359  48.1
#512*1024 average time: 0.009457776040741892  111.1

#


#....................................ERFNET
#512*1024  average time: 0.07564253036421958  13.33
#1024*2048 average time: 0.33082881599965724  3.0







#....................................segnet
#512*1024 average time: 0.05974130437831686 16.9
#640*360 average time: 0.029695481965036102 34.5



#....................................ENET
#512*1024  average time: 0.02056735934633197 48.8
#640*360   average time: 0.013622320059574011 76.9



#...................................ESPNetV2
#512*1024  average time: 0.012102721917508828 83.3






#.......................................ICnet
#1024*2048

