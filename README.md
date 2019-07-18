# NDNet
This repository is the implementation of "Narrow while deepnet work for real-time semantic segmenation"

We have tested our code on Ubuntu 16.04, Pytorch 1.0


Requirements:

Pytorch 1.0

Usage

1 Prepare the data

You need first to download the Cityscapes dataset by yourself, since the Cityscape use 19 class label for semantic segmentation task, you also need to convert the original 33 class label image to 19 class image using the code provided by the Cityscapes team

2 set the parameters

All the parameters are manally set in the Seg_NDnet/train.py

3 Train

After all the parameters is set, you can train the model just with: 

python train.py


If there are any other error message about "python package can not find", use "pip intall xx" to setup 

4 Test

After the training completed, set the "RESTORE_FROM" in "Seg_NDnet/infer_citys.py" with the path at which your model saved, and then  


python infer_citys.py

this command will make prediction over the val set one by one and save the prediction to the path you specified


