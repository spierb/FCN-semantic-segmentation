'''
Author: bg
Date: 2020-11-11 21:54:56
LastEditTime: 2020-11-13 09:55:25
LastEditors: bg
Description: build up the model.
FilePath: /FCN-semantic-segmentation/src/model.py
'''
import torch
import os
from vgg import VGG
from torch.nn import Module

class SegNet(Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.feature_net = VGG()

    def forward(self, x):
        x = self.feature_net(x)
        return x

device = torch.device('cuda')
net = SegNet().to(device)
from torchsummary import summary
print(summary(net, (3, 512, 512)))