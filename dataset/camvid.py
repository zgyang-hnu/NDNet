import torch
import numpy as np
import glob
import os
import time
import cv2
import random
import math

import torch.utils.data as data
import torchvision.transforms as transforms
import time

from PIL import Image


from .data_utils import *

class Camvid(data.Dataset):
      
      def __init__(self, train_path, val_path, train_size=None, use_val=False):
          super(Camvid, self).__init__()
          self.train_path = train_path
          self.train_size = train_size
          self.val_path = val_path
          self.use_val = use_val

          self.train_list = self.get_train_list(train_path, val_path)
          #self.transform = transforms.Compose(transforms_)
          self.file_length = len(self.train_list)

      def __len__(self):
         return self.file_length


      def __getitem__(self, index):
          image_path = self.train_list[index]['image_path']
          label_path = self.train_list[index]['label_path']
          image_name = self.train_list[index]['image_name']


          im, im_label = read_image_label_with_dataaug(image_path,label_path)
          im, im_label = random_crop_rgb_image_and_label(im, im_label, self.train_size)
          im = self.normalize(im)
          im = im.transpose(2, 0, 1)# move the rgb channel to dimension 1 for pytorch data format NCHW

          image = torch.from_numpy(np.ascontiguousarray(im)).float()
          label = torch.from_numpy(np.ascontiguousarray(im_label)).long()

          return dict(image=image, label=label, image_name=image_name)
     
      def normalize(self, img, mean=np.array([0.5, 0.5, 0.5])):
    #
    #
           img = img.astype(np.float32) / 255.0
           img = img - mean #
           img = img * 2.0 #0-1

           return img

      def get_train_list(self, train_path, val_path):

         image_dir = train_path[0]
         label_dir = train_path[1]

         image_dir1 = val_path[0]
         label_dir1 = val_path[1]

         if not os.path.exists(label_dir):
              raise IOError("No such training direcotry exist!")
         match_str=os.path.join(label_dir, "*.png")
         print(match_str)
         image_label_path_list=[]
         image_record_list=[]
         image_label_path_list.extend(glob.glob(match_str))
         print(len(image_label_path_list))
    
         for f in image_label_path_list:                   
             image_name=os.path.splitext(f.split('/')[-1])[0]
             
             #print(image_name)
             image_path=os.path.join(image_dir, image_name+'.png')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                image_record_list.append(image_record)


         match_str=os.path.join(label_dir1,  "*.png")
         print(match_str)
         image_label_path_list=[]
         image_label_path_list.extend(glob.glob(match_str))

         if self.use_val:
           for f in image_label_path_list:                   
             image_name=os.path.splitext(f.split('/')[-1])[0]

             
             image_path=os.path.join(image_dir1, image_name+'.png')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                image_record_list.append(image_record)


         return image_record_list






