# 3D Structure-invariant Transformation and Weighted Intermediate Feature Divergence
This repository contains code to reproduce results from the paper:

[Boosting Adversarial Transferability for Hyperspectral Image Classification Using 3D Structure-invariant Transformation and Weighted Intermediate Feature Divergence](https://arxiv.org/abs/2506.10459)  
Chun Liu, Bingqian Zhu, Tao Xu, Zheng Zheng, Zheng Li, Wei Yang, Zhigang Han, Jiayao Wang

# Requirements  
Pytorch 2.4.0  
Python  3.10.8  
Numpy  1.26.4    
Scipy  1.14.1  
 
# Usage   
1. The hyperspectral dataset is sourced from the link below, you can download the dataset and place it under the 900(1000)\_PaviaU01 folder. You can use your own dataset by matlab, and the "SelectSample.m" file can help you split the training set and the test set.  
   [The Hyperspectral dataset link.](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_University_scene)

2. To train the "train_model.py" with dataset PvaiaU, which will generate checkpoint: '/net_resnet18.pkl'. You can try other targetmodel, such as VGG, Inc-v3.  
                      <pre> ```                  $ python train_model.py --dataset PaviaU --train ``` </pre>   

3.  Run the "main.py" to generate adversarial examples.  
                   <pre> ```                  $ python "main.py" --dataset PaviaU  ``` </pre>  


# Related works
[SS_FGSM_Hyperspectral Adversarial Attack](https://github.com/AAAA-CS/SS_FGSM_HyperspectralAdversarialAttack)  
[Other comparative attack methods](https://github.com/Trustworthy-AI-Group/TransferAttack)  

 

