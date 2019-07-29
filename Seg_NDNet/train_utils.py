import torch
import torch.nn as nn
import numpy as np




def build_para_list_from_model(model, lr):
    para_list=[]
    para_with_l2decay = []
    para_without_l2decay  = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            para_with_l2decay.append(m.weight)
            if m.bias is not None:
                para_without_l2decay.append(m.bias)
        elif isinstance(m, nn.BatchNorm2d): #all the params of BN without l2decay
            if m.weight is not None:
                para_without_l2decay.append(m.weight)
            if m.bias is not None:
                para_without_l2decay.append(m.bias)


    assert len(list(model.parameters())) == len(para_with_l2decay) + len(para_without_l2decay)
    para_list.append(dict(params=para_with_l2decay, lr=lr))
    para_list.append(dict(params=para_without_l2decay, weight_decay=.0, lr=lr))
    return para_list


def init_model(model, cnn_init_func, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            cnn_init_func(m.weight, **kwargs)
        elif isinstance(m, nn.BatchNorm2d):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)



class Moving_average_computer:
   #'''compute moving average'''

    
    def __init__(self,capacity):
        
        self.sample_array=np.zeros(capacity)
        self.capacity=capacity
        self.new_sample_entry=0

    def add_new_sample(self,sample):
        if self.new_sample_entry>=self.capacity:
           self.new_sample_entry=0
  
        self.sample_array[self.new_sample_entry]=sample
        self.new_sample_entry+=1#
        #print(self.sample_array)

    def compute_average(self):
        #print(np.average(self.sample_array))
        return np.average(self.sample_array)
