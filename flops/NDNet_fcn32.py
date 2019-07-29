import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np

import sys
sys.path.append('..')
from backbone import NDNet45, NDNet29, NDNet29w, NDNet61
from backbone import StdConvBR, SepConvBR

label_colours_cityscape = [(0,0,0), (0,  0,  0),(  0,  0,  0),(  0,  0,  0),( 0,  0,  0),
               
                (111, 74,  0),( 81,  0, 81) ,(128, 64,128),(244, 35,232) ,(250,170,160),
                
                (230,150,140),( 70, 70, 70),(102,102,156),(190,153,153),(180,165,180),
               
                (150,100,100),(150,120, 90),(153,153,153),(153,153,153),(250,170, 30),
                
                (220,220,  0),(107,142, 35) ,(152,251,152) ,( 70,130,180),(220, 20, 60),
                (255,  0,  0),(  0,  0,142),(  0,  0, 70),(  0, 60,100),(  0,  0, 90),
                (  0,  0,110),(  0, 80,100),(  0,  0,230),(119, 11, 32)]
               
label_color_camvid=[
[64,128,64],[192,0,128],[0,128, 192],[0,128,  64],
[128,0,  0],[64,0, 128],[64,0,  192],[192,128,64],
[192,192,128],[64,64,128],[128,0,192],[192,0,64], 
[128,128,64],[192,0,192],[128,64,64],[64,192,128],
[64,64,0],   [128,64,128],[128,128,192],[0,0,192],
[192,128,128],[128,128,128],[64,128,192],[0,0,64],
[0,64,64],    [192,64,128], [128,128,0],[192,128,192],
[64,0,64],    [192,192,0],  [64,192,0],[0,0,0]
]    

label_color_camvid11=[
[128, 128, 128],    # sky
[128, 0, 0],        # building
[192, 192, 128],# column_pole
[128, 64, 128],# road
[0, 0, 192],# sidewalk
[128, 128, 0], # Tree
[192, 128, 128], # SignSymbol
[64, 64, 128],# Fence
[64, 0, 128],# Car
[64, 64, 0],# Pedestrian
[0, 128, 192],# Bicyclist
[0, 0, 0] # Void
] 

def decode_labels_camvid11(mask):
    """Decode segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    h, w = mask.shape
    #assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((h, w, 3), dtype=np.uint8)
    #for i in range(num_images):
    #img = Image.new('RGB', (len(mask[0]), len(mask)))
    #pixels = img.load()
    for i in range(h):
        for j in range(w):
          if mask[i,j]< 255:
            outputs[i,j,:] = label_color_camvid11[mask[i,j]]
    #outputs = np.array(img)
    return outputs


def decode_labels_camvid(mask):
    """Decode segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    h, w = mask.shape
    #assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((h, w, 3), dtype=np.uint8)
    #for i in range(num_images):
    #img = Image.new('RGB', (len(mask[0]), len(mask)))
    #pixels = img.load()
    for i in range(h):
        for j in range(w):
          if mask[i,j]< 255:
            outputs[i,j,:] = label_color_camvid[mask[i,j]]
    #outputs = np.array(img)
    return outputs



def decode_labels_cityscape(mask):
    """Decode segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    h, w = mask.shape
    #assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((h, w, 3), dtype=np.uint8)
    #for i in range(num_images):
    #img = Image.new('RGB', (len(mask[0]), len(mask)))
    #pixels = img.load()
    for i in range(h):
        for j in range(w):
            if mask[i,j]< 255:
               outputs[i,j,:] = label_colours_cityscape[mask[i,j]]
    #outputs = np.array(img)
    return outputs


def trainid2labelid_efficient(prediction):
    shape=np.shape(prediction)
    prediction=prediction+13
    prediction=prediction.reshape(-1)
    #color=np.zeros(prediction.shape,3)
    index=np.where(prediction==13)
    #print(index[0].shape)
    index1=np.where(prediction==14)
    index=np.concatenate((index[0],index1[0]))
    #print(index.shape)
    
    prediction[index]=prediction[index]-6
    index=np.where(prediction==15) 
    index1=np.where(prediction==16) 
    index2=np.where(prediction==17) 
    index=np.concatenate((index[0],index1[0],index2[0]))
    prediction[index]=prediction[index]-4

    index=np.where(prediction==18)
    prediction[index[0]]=prediction[index[0]]-1

    index=np.where(prediction==29) 
    index1=np.where(prediction==30) 
    index2=np.where(prediction==31) 
    index=np.concatenate((index[0],index1[0],index2[0]))
    
    prediction[index]=prediction[index]+2
    prediction=prediction.reshape(shape[0],shape[1])
    return prediction



class NDNet_fcn32(nn.Module):
    def __init__(self, out_channels, criterion=None, 
                 aux_loss=True, out_stride=32):
        super(NDNet_fcn32, self).__init__()
        self.features = NDNet61(out_stride=out_stride)# 
        

  
        block4_channel = 24*4
        block5_channel = 48*4

        self.seg=DensePrediction(block5_channel, 64, out_channels, 32)#

        if aux_loss:
          print('.....auxloss is introduced..............')
          self.aux_seg =DensePrediction(block4_channel, 128, out_channels, 16)#
     
        self.criterion = criterion


    def forward(self, data, label=None,step=None):

        
        feature_blocks = self.features(data)
        lastf=feature_blocks[2]

     
         
        
        if self.criterion:
            scores=self.seg(lastf)
            
            #pred=pred.view()
            
            main_loss = self.criterion(scores, label)
            if self.aux_loss:
               aux_loss  = self.criterion(self.aux_seg(feature_blocks[1]), label)
               loss=main_loss + 0.4*aux_loss
            else:
               loss = main_loss
            if step % 10000==0:
               print('........size of feature maps........',lastf.shape)
               pred=F.softmax(scores, dim=1)
               print('........size of score maps...........',pred.shape)
               im=torch.max(pred[0].permute(1,2,0),dim=2)[1].cpu().detach().numpy()
               print(im.shape)
               im=Image.fromarray(decode_labels_cityscape(trainid2labelid_efficient(np.uint8(im))))
               #im=Image.fromarray(decode_labels_camvid(np.uint8(im)))
               #im.show()
               im.save('./mid_train_results/'+str(step)+'_pred'+'.jpg')
               im=label[0].cpu().numpy()
               #print(im.shape)
               im=Image.fromarray(decode_labels_cityscape(trainid2labelid_efficient(np.uint8(im))))
               #im=Image.fromarray(decode_labels_camvid(np.uint8(im)))
               #im.show()
               im.save('./mid_train_results/'+str(step)+'_pred'+'.jpg')
            #print(loss)
            #print('....lossshape.....',loss.shape)
            return loss

        return F.log_softmax(self.seg(lastf), dim=1)



class DensePrediction(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, up_rate,
                   norm_layer=nn.BatchNorm2d):
        super(DensePrediction, self).__init__()
        
        self.transform = SepConvBR( in_channels, mid_channels, 3, 1, 1,
                                       bn=True)
        
        
        self.score = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.up_rate = up_rate

    def forward(self, x):
        feature_maps = self.transform(x)
        
        score_maps = self.score(feature_maps)
        if self.up_rate > 1:
            output = F.interpolate(score_maps, scale_factor=self.up_rate,
                                   mode='bilinear',
                                   align_corners=True)

        return output



if __name__ == "__main__":
    model = NDNet_ende(19, criterion=nn.CrossEntropyLoss(reduction='mean'))
    print(model.parameters)
