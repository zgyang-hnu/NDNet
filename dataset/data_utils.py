import glob
import os

from PIL import Image


import numpy as np
import scipy.misc as misc #
import scipy.io  

import random
import math
import time



def read_image_label_with_dataaug(image_path,label_path, scale_candi=[0.75,1,1.5,1.75,2], ran_scale=True, mirror=True,rotation=False):
    if not os.path.exists(image_path):
        raise IOError("image not found!")

    im=Image.open(image_path)
    im_label=Image.open(label_path)
   
    #if you want to trained on 512×1024 images
    #o_shape=im.size
    #dst_h=int(o_shape[0]*0.5)
    #dst_w=int(o_shape[1]*0.5)
    #im=im.resize([dst_h,dst_w],1)#Image.BILINEAR
    #im_label=im_label.resize([dst_h,dst_w])

    assert(im.size[0]==im_label.size[0])
    assert(im.size[1]==im_label.size[1])

    if mirror:
       p=random.uniform(0,1)
       if p>0.5:
         im=im.transpose(0)
         im_label=im_label.transpose(0)

    if ran_scale:
       scale=random.sample(scale_candi,1)[0]#random.shuffle(scale_candi)[0]
       o_shape=im.size
       dst_h=int(o_shape[0]*scale)
       dst_w=int(o_shape[1]*scale)
       im=im.resize([dst_h,dst_w],1)#Image.BILINEAR
       im_label=im_label.resize([dst_h,dst_w])

    if rotation:
       angle=random.uniform(-10,10)
       im=im.rotate(angle,resample=Image.BILINEAR)
       im_label=im_label.rotate(angle)


    
    return np.array(im),np.array(im_label)


def array_resize_with_pad(array,height,width):
    

   orig_shape=array.shape
   array_list=[]
   if len(orig_shape) > 2:#
      if len(orig_shape) > 3:
         raise ValueError("cannot process image with size bigger than 3")
      else:
         for i in range(3):
             _2d_array=array[:,:,i]
             _2d_array=_2d_array_resize_with_pad(_2d_array,height,width)
             array_list.append(_2d_array)
      array=np.stack([array_list[i] for i in range(3)], axis=2)#  
     
   else:     
      array=_2d_array_resize_with_pad(array,height,width)
   
   return array


def _2d_array_resize_with_pad(array,height,width,pad_value=0):
   orig_shape=array.shape
   
  
   pad_0_before=math.ceil((height-orig_shape[0]) / 2)#
   pad_0_after=height-orig_shape[0]-pad_0_before
     
   pad_1_before=math.ceil((width-orig_shape[1]) / 2)#
   pad_1_after=width-orig_shape[1]-pad_1_before
  
   

   array=np.lib.pad(array,((pad_0_before,pad_0_after),(pad_1_before,pad_1_after)),'constant', constant_values=((pad_value,pad_value),(pad_value,pad_value)))
   #                
   return array


def _2d_label_resize_with_pad(label,height,width):

   label=np.int32(label)  
   label=_2d_array_resize_with_pad(label,height,width,pad_value=255)
  
   return label


def random_crop_rgb_image_and_label(rgb_image,label,size):
  
    crop_h = size[0]
    crop_w = size[1]
    final_h=np.maximum(rgb_image.shape[0],crop_h)#if pad needed? since the crop size may larger than the original size
    final_w=np.maximum(rgb_image.shape[1],crop_w)

    if final_h != rgb_image.shape[0] or final_w != rgb_image.shape[1]:
      rgb_image = array_resize_with_pad(rgb_image,final_h,final_w)
      label = _2d_label_resize_with_pad(label,final_h,final_w)

    assert(rgb_image.shape[0]==label.shape[0])

    start_h=random.randint(0,final_h-crop_h) 
   
    start_w=random.randint(0,final_w-crop_w) 
   
    rgb_image=rgb_image[start_h:start_h+crop_h,start_w:start_w+crop_w,:]
    label=label[start_h:start_h+crop_h,start_w:start_w+crop_w]

  
    return rgb_image,label










      
      


