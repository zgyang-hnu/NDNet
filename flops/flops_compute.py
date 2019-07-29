import torchvision.models as models
import torch
from flops_counter import get_model_complexity_info

from erfnet import *

from enet import *
from NDNet_ende import *
from NDNet_fcn32 import *

from espnetv2 import * 
from icnet import *
from segnet import *
from contextnet import *
from MobileNetV2_fcn import *

with torch.cuda.device(0):
  #model = models.resnet18()
  #model = ERFNet( 19)
  
 
  #model = NDNet_ende(19)
  #model = NDNet_fcn32(19)
 
  #model = contextnet(19)
  #model = MobileNetV2(19)
  #model = ENet(19)
  #model = icnet(19)#icenet use 513 1025 or 1025 2049 for test
  #model = segnet(19)#
  model = EESPNet_Seg(19).cuda()
  #net = models.densenet161()
  model.eval()
  flops, params = get_model_complexity_info(model, (3,1024, 2048), as_strings=True, print_per_layer_stat=True)
  print('Flops:  ' + flops)
  print('Params: ' + params)







#.....................................NDNET
#
#Flops:  6.92 GMac ndnet ende-1024*2018
#Params: 502.62 k

#Flops:  1.73 GMac  ndnet ende-512*1024
#Params: 502.62 k

#Flops:  2.09 GMac   ndnet45-32 1024*2048
#Params: 420.19 k

#Flops:  1.72 GMac   ndnet61-32 1024*2048 
#Params: 323.33 k

#Flops:  2.84 GMac   ndnet29-32 1024*2048
#Params: 550.42 k 

#Flops:  17.41 GMac  ndnet29w-32 1024*2048
#Params: 3.46 M


#.....................................contextnet
#Flops:  6.42 GMac
#Params: 871.7 k

#Flops:  1.61 GMac
#Params: 871.7 k


#....................................ERFNET
#512*1024
#Flops:  26.86 GMac
#Params: 2.07 M



#....................................segnet
#640*360
#Flops:  143.24 GMac *2 =286G consistent with the FLOPS reported in the paper
#Params: 29.45 M


#....................................ENET
#640*360
#Flops:  1.91 GMac
#Params: 357.99 k


#512*1024
#Flops:  4.35 GMac
#Params: 357.99 k

#...................................ESPNetV2
#512*1024
#Flops:  1.78 GMac
#Params: 340.47 k



#.......................................ICnet
#1024*2048
#Flops:  30.86 GMac
#Params: 7.76 M



