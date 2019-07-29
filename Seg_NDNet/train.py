from __future__ import division
import os.path as osp
import sys

import time
import os
import scipy.io as matio

from tqdm import tqdm
import numpy as np


import torch
from torch.utils import data
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from PIL import Image

sys.path.append('..')

from NDNet_ende import *
from dataset import Cityscapes
from train_utils import build_para_list_from_model, init_model, Moving_average_computer

BN2d=nn.BatchNorm2d
max_steps = 100000

#specify the parameters used to save model
snapshot_dir = '/opt/temp/citys_ndnet_itsfinal'
start_save_step = max_steps/2
save_interval = 10000
if not os.path.exists(snapshot_dir):
      os.makedirs(snapshot_dir)


#specify the training parameters
batch_size = 11
train_size = [1024, 1024]

base_lr = 0.1
lr_momentum =0.9


l2_weight_decay = 5e-4

bn_eps = 1e-5
bn_momentum = 0.1


#specify the path of the Cityscape dataset, and construct the dataset and dataloader for training 
train_path = ['/opt/Cityscapes/leftImg8bit/train',
              '/opt/Cityscapes/gtFine_trainvaltest/gtFine/train']
val_path =   ['/opt/Cityscapes/leftImg8bit/val',
              '/opt/Cityscapes/gtFine_trainvaltest/gtFine/val']
use_val = False

DATASET = Cityscapes(train_path, val_path, train_size=train_size,use_val=use_val)
print('the length of training set is:' , len(DATASET), ',please check if consistent with your setting')
time.sleep(10)
num_classes = 19
num_workers = 4

train_loader = data.DataLoader(DATASET,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               drop_last=True,
                               shuffle=True,
                               pin_memory=True,
                               sampler=None)


#specify the loss function
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)


#build and init model 
model = NDNet_ende( num_classes, criterion=criterion)

init_model(model, nn.init.kaiming_normal_,
            bn_eps, bn_momentum,
            mode='fan_in', nonlinearity='relu')

#get trainable params and set hyperparas 
para_list = build_para_list_from_model(model, base_lr)
print(len(para_list[0]['params']),len(para_list[1]['params']))

#specify the optimizer
optimizer = torch.optim.SGD(para_list,
                            lr=base_lr,
                            momentum=lr_momentum,
                            weight_decay=l2_weight_decay)


#judge the device the training performed on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
model.to(device)
#model.cuda()

#set the "is_training" status in some modules(e.g., batch normalization, drop out etc) to true
model.train()
print('....model_params....',model.parameters)

#record loss to mat
loss_moving_averager=Moving_average_computer(20)
loss_list=[]



one_epoch_iter=int(len(DATASET)/batch_size)
print(one_epoch_iter)
stop_flag=False
for epoch in range(10000):
   dataloader = iter(train_loader)
   dbar = tqdm(range(one_epoch_iter))
   for i in dbar:
     step=epoch*one_epoch_iter+i
     #s_time = time.time()
     
     optimizer.zero_grad()
     trainbatch = dataloader.next()#
     ims = trainbatch['image']
     labels = trainbatch['label']
     #print(time.time()-s_time)
     #you can used these commented code to test the data loading process
     #print(ims.shape)  
     #print(labels.shape)
     #im=ims[0].permute(1,2,0).numpy()
     #print(im)
     #im=Image.fromarray(np.uint8(im*255))
     #im.show()
     #im=labels[0].numpy()
     #print(im.shape)
     #im=Image.fromarray(decode_labels_cityscape(trainid2labelid_efficient(np.uint8(im))))
     #im.show()
     #print(trainbatch['image_name'])
     #time.sleep(20)

     ims = ims.cuda(non_blocking=True)
     labels = labels.cuda(non_blocking=True)

     loss = model(ims, labels, step)


     #..........record loss to mat......................
     loss_moving_averager.add_new_sample(loss.cpu().detach().numpy())
     if (step+1)%100==0:
        loss_list.append(loss_moving_averager.compute_average())
        loss_record=np.array(loss_list)
        if not os.path.exists(snapshot_dir):
         os.makedirs(snapshot_dir)
        matio.savemat(os.path.join(snapshot_dir,'loss_record.mat'),{'loss':loss_record})


     #..........scrath training policy......................
            
     if (step+1) <35000:# camvid set to 7K city to 35K
               lr= base_lr
     elif (step+1)<60000:# camvid set to 15K 60K
               lr= base_lr/10
     elif (step+1)<80000:# camvid set to 20K 80k
               lr= base_lr/100
     else:
               lr=base_lr/200

     # update the learning rate
     optimizer.param_groups[0]['lr'] = lr
     optimizer.param_groups[1]['lr'] = lr

     loss.backward()  # compute gradients
     optimizer.step() # update params


     if step < 21:
        disp_info = 'Steps {}/{}'.format(step, max_steps) + ' lr=%.3e' % lr  + ' loss=%.3f' % loss.item()
        #print(print_str)

    #..........display mean loss rathar than real-time loss......................
     else:
        disp_info = 'Steps {}/{}'.format(step, max_steps) + ' lr=%.3e' % lr  + ' meanloss_over_last20=%.3f' % loss_moving_averager.compute_average()
        #print(print_str)
     dbar.set_description(disp_info, refresh=False) 
     if step > max_steps:
        stop_flag= True
        break

     if step > start_save_step and step % save_interval ==0:
        torch.save(model.state_dict(), snapshot_dir+ '/%s.pth' %  step) 

   #pause every two epochs, if your GPU is colded with water machine, you can comment this to speed the training
   if (epoch+1)%2==0:   
     print('..........................cooling GPU for 50s.................')
     time.sleep(50) 

   if stop_flag:
      torch.save(model.state_dict(), snapshot_dir+ '/%s.pth' %  step) 
      break





