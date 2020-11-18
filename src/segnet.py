'''
Author: bg
Date: 2020-11-11 21:54:56
LastEditTime: 2020-11-18 12:13:17
LastEditors: bg
Description: build up the model.
FilePath: /FCN-semantic-segmentation/src/segnet.py
'''
import torch
import os
from vgg import VGG
from torch import nn
from torch.nn import Module

class SegNet(Module):
    def __init__(self, num_classes=20, batch_norm=False):
        super(SegNet, self).__init__()
        self.num_classes = num_classes
        self.batch_norm = batch_norm

        self.feature_net = VGG()
        self.deconv_net = self.seg_layer()

        self.features = []
        self.feature_net_outputchannel = self.feature_net.CNNlayer_outchannels

    def seg_layer(self):
        channels = [512, 256, 128, 64, 32]
        seg_layer = []
        for i in range(4):
            seg_layer.append(nn.ConvTranspose2d(channels[i], channels[i+1], 3, 2, 1, 1))
            if self.batch_norm:
                seg_layer.append(nn.BatchNorm2d(channels[i+1]))
        seg_layer.append(nn.ConvTranspose2d(channels[-1], self.num_classes, 3, 2, 1, 1))
        # seg_layer.append(nn.Sigmoid())
        return nn.Sequential(*seg_layer)

    def forward(self, x):
        x = self.feature_net(x)
        self.features = self.feature_net.middle_feature
        x = self.deconv_net(x)
        return x