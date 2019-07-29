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
import torch.nn.functional as F
import glob


from NDNet_ende import NDNet_ende
#from NDNet_fcn8 import NDNet_fcn8
#from fcn_res import FCN
BatchNorm2d=nn.BatchNorm2d



#specify the path accordingly

VAL_IMAGE_DIR='/opt/Cityscapes/leftImg8bit/val'

TEST_IMAGE_DIR='/opt/Cityscapes/leftImg8bit/test'

VAL=True # if test on val set
SAVE_COLOR_PREDICTION = True # if need to save color prediction
if VAL:
   IMAGE_PATH=VAL_IMAGE_DIR
   PREDICTION_SAVED_DIR='/opt/yzg_matlab_tool/Cityscape/citys_val_deep/'
   COLOR_PREDICTION_SAVED_DIR='./saved_color/'  
else:
   IMAGE_PATH=TEST_IMAGE_DIR
   PREDICTION_SAVED_DIR='/opt/yzg_matlab_tool/Cityscape/citys_test_ndnetv2'
   COLOR_PREDICTION_SAVED_DIR='./saved_color/'

#


RESTORE_FROM ='./ndnet.pth' #'/opt/temp/citys_ndnet_itsfinal/100000.pth'



if not os.path.exists(PREDICTION_SAVED_DIR):
   os.mkdir(PREDICTION_SAVED_DIR)

if not os.path.exists(COLOR_PREDICTION_SAVED_DIR):
   os.mkdir(COLOR_PREDICTION_SAVED_DIR)



#labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
#]

label_colours_cityscape = [(0,0,0), (0,  0,  0),(  0,  0,  0),(  0,  0,  0),( 0,  0,  0),
               
                (111, 74,  0),( 81,  0, 81) ,(128, 64,128),(244, 35,232) ,(250,170,160),
                
                (230,150,140),( 70, 70, 70),(102,102,156),(190,153,153),(180,165,180),
               
                (150,100,100),(150,120, 90),(153,153,153),(153,153,153),(250,170, 30),
                
                (220,220,  0),(107,142, 35) ,(152,251,152) ,( 70,130,180),(220, 20, 60),
                (255,  0,  0),(  0,  0,142),(  0,  0, 70),(  0, 60,100),(  0,  0, 90),
                (  0,  0,110),(  0, 80,100),(  0,  0,230),(119, 11, 32)]
               
    
def normalize(img, mean=np.array([0.5, 0.5, 0.5])):
    #
    #
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img * 2.0
   

    return img


def load_model(model, model_file, is_restore=False):
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict, strict=False)

    del state_dict 
    return model





def decode_labels_cityscape(mask):
    """Decode segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.

    
    Returns:
      A  RGB image of the same size as the input. 
    """
    h, w = mask.shape
    #assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((h, w, 3), dtype=np.uint8)
    #for i in range(num_images):
    #img = Image.new('RGB', (len(mask[0]), len(mask)))
    #pixels = img.load()
    for i in range(h):
        for j in range(w):
            outputs[i,j,:] = label_colours_cityscape[mask[i,j]]
    #outputs = np.array(img)
    return outputs





def trainid2labelid(prediction):
    shape=np.shape(prediction)
    for i in range (shape[0]):
       for j in range(shape[1]):
         if prediction[i][j]==0 or prediction[i][j]==1:
            prediction[i][j]=prediction[i][j]+7
         elif prediction[i][j]==2 or prediction[i][j]==3 or prediction[i][j]==4:
            prediction[i][j]=prediction[i][j]+9
         elif prediction[i][j]==5:
            prediction[i][j]=prediction[i][j]+12
         elif prediction[i][j]==16 or prediction[i][j]==17 or prediction[i][j]==18:
            prediction[i][j]=prediction[i][j]+15
         else:
            prediction[i][j]=prediction[i][j]+13
    return prediction




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



def get_test_list_from_path_cityscape(image_dir):
    if not os.path.exists(image_dir):
        raise IOError("No such image direcotry exist!")
    match_str=os.path.join(image_dir, '*', '*' "*.png")#
    print(match_str)
    image_path_list=[]#
    
    image_path_list.extend(glob.glob(match_str))#
    
    print('there are ', len(image_path_list), 'test images')
    #print(image_path_list[0])
    return image_path_list


def main():
    
    #args = get_arguments()
    
    # Prepare image.



    
    network =NDNet_ende(19, criterion=None)#
   
    model=load_model(network, RESTORE_FROM)


    print(model)
    model.eval()#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
    model.to(device)
    duration=0
    test_list=get_test_list_from_path_cityscape(IMAGE_PATH)
    i=0
    #for i in model.named_parameters():
    #   print(i)
    print(len(test_list))
    with torch.no_grad():
     for image in test_list:
       image_name=os.path.splitext(image.split('/')[-1])[0]
       saved_path_pred=os.path.join(PREDICTION_SAVED_DIR,image_name+'.png')
       saved_path_pred_color=os.path.join(COLOR_PREDICTION_SAVED_DIR,image_name+'.jpeg')
       im=Image.open(image)
       #o_shape=im.size
       #dst_h=int(o_shape[0]*0.5)
       #dst_w=int(o_shape[1]*0.5)
       #im=im.resize([dst_h,dst_w],1)#Image.BILINEAR  
       _3d_image=np.array(im)

       if _3d_image.ndim<3:
           _3d_image=np.stack([_3d_image,_3d_image,_3d_image],axis=-1)
           print('gray image used %s' % image_name)



       _4d_image_array=np.expand_dims(_3d_image,0) 
       _4d_image_array=normalize(_4d_image_array)
       img = torch.from_numpy(_4d_image_array).float().permute(0,3,1,2).contiguous()
       img = img.cuda(non_blocking=True)
       
       start_time = time.time()
       scores=model(img)
       #print( scores.shape)
       prediction=torch.max(scores[0].permute(1,2,0),dim=2)[1].cpu().detach().numpy()
       print('time elapsed %f' % (time.time()-start_time))

       duration=duration+(time.time()-start_time)
       prediction=np.squeeze(prediction)
       print(i+1)
       prediction=np.uint8(prediction)

       prediction=trainid2labelid_efficient(prediction)

       i+=1
       shape=np.shape(prediction)



       im=Image.frombytes('L', (shape[1],shape[0]),prediction)
       im.save(saved_path_pred,'png') #the saved png can be used to compute the miou value with the code provided by the official  
       if SAVE_COLOR_PREDICTION:
         color_pred=decode_labels_cityscape(prediction)
         im=Image.fromarray(color_pred)
         im.save(saved_path_pred_color,'jpeg')

     print(duration)
    #print('The output file has been saved to {}'.format(args.save_dir + 'mask.png'))

    
if __name__ == '__main__':
    main()
