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

class Cityscapes(data.Dataset):
      
      def __init__(self, train_path, val_path, train_size=None, use_val=False):
          super(Cityscapes, self).__init__()
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
         match_str=os.path.join(label_dir, '*', "*labelTrainIds.png")
         print(match_str)
         image_label_path_list=[]
         image_record_list=[]
         image_label_path_list.extend(glob.glob(match_str))
         #print(len(image_label_path_list))
    
         for f in image_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
             image=image.split('_')
             city=image[0]
             #print(city)
             image_name=image[0]+'_'+image[1]+'_'+image[2]
             #print(image_name)
             image_path=os.path.join(image_dir, city, image_name+'_leftImg8bit.png')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                                                                                                
                image_record_list.append(image_record)


         match_str=os.path.join(label_dir1, '*', "*labelTrainIds.png")
         print(match_str)
         image_label_path_list=[]
         image_label_path_list.extend(glob.glob(match_str))

         if self.use_val:
           for f in image_label_path_list:                   
             image=os.path.splitext(f.split('/')[-1])[0]
             image=image.split('_')
             city=image[0]
             
             image_name=image[0]+'_'+image[1]+'_'+image[2]
             
             image_path=os.path.join(image_dir1, city, image_name+'_leftImg8bit.png')
        
             if not os.path.exists(image_path):
                print("Image %s is not exist %s" % (image_name,image_dir))
                raise IOError("Please Check")

             else:
                image_record={'label_path':f,'image_path':image_path,'image_name':image_name}
                image_record_list.append(image_record)


         return image_record_list



#the following code is used to test the Class of Cityscape, you can igore these codes if you do not want to test it by yourself

label_colours_cityscape = [(0,0,0), (0,  0,  0),(  0,  0,  0),(  0,  0,  0),( 0,  0,  0),
               
                (111, 74,  0),( 81,  0, 81) ,(128, 64,128),(244, 35,232) ,(250,170,160),
                
                (230,150,140),( 70, 70, 70),(102,102,156),(190,153,153),(180,165,180),
               
                (150,100,100),(150,120, 90),(153,153,153),(153,153,153),(250,170, 30),
                
                (220,220,  0),(107,142, 35) ,(152,251,152) ,( 70,130,180),(220, 20, 60),
                (255,  0,  0),(  0,  0,142),(  0,  0, 70),(  0, 60,100),(  0,  0, 90),
                (  0,  0,110),(  0, 80,100),(  0,  0,230),(119, 11, 32)]

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

if __name__ == "__main__":
    train_path = ['/opt/Cityscapes/leftImg8bit/train',
                  '/opt/Cityscapes/gtFine_trainvaltest/gtFine/train']
    val_path = ['/opt/Cityscapes/leftImg8bit/val',
                '/opt/Cityscapes/gtFine_trainvaltest/gtFine/val']
    transforms_ = [transforms.RandomCrop((1024, 1024), pad_if_needed=True)]
    bd = Cityscapes(train_path, val_path, train_size=[1024,1024])
    for i in range(30):
      start_time=time.time()
      out=bd.__getitem__(i)
      print(time.time()-start_time)
      print(bd.file_length)
      print(out['image'].shape)
      #k=transforms.ToPILImage()k(out['data'])#
      im=out['image'].numpy()
      print(im.shape,out['image_name'])
      im=Image.fromarray(np.uint8(im))
      im.show()
      im=Image.fromarray(decode_labels_cityscape(trainid2labelid_efficient(out['label'].numpy())))
      im.show()
      time.sleep(10)

