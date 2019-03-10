hyperlpr-train_e2e
======
A simple code for creating licence plate images and train e2e network based on [HyperLPR](https://github.com/zeusees/HyperLPR)   

****
	
|Author|LCorleone|
|---|---
|E-mail|lcorleone@foxmail.com


****
## Requirements
* tensorflow 1.5
* keras 2.2.0
* some common packages like numpy and so on.

## Quick start
* run create_train_data.py to create plate image and corresponding labels. This repository also contains the plate generator and can generate thousands of plates.
* reset the train data path and run train_nn.py to train your model.

## Attention
* The image size created automatically is 120 * 30, fix the input size when you use the e2e network. You can create and train your own e2e network if you want.  
* Generate at least 50000 images for training, less may degrade the performance.
* Also, when tested in real scene, the e2e network performs not very well due to that the images' quality created automatically are still poor. If you have real image dataset and labels, it may be perfect.  

