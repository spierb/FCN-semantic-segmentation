'''
Author: bg
Date: 2020-11-11 21:55:29
LastEditTime: 2020-11-13 09:54:34
LastEditors: bg
Description: basic vgg network.
FilePath: /FCN-semantic-segmentation/src/vgg.py
'''
import torch
from torch.nn import *

layermap = {
    # conv-layer output channels 
    8: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    10: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(torch.nn.Module):
    def __init__(self, num_classes=None, batch_norm=False, pretrained='', vgg=16, dense_clf=True):
        super(VGG, self).__init__()
        self.batch_norm = batch_norm
        self.dense_clf = dense_clf
        self.num_classes = num_classes
        self.layer_args = layermap[vgg]
        # loading the pretrained model
        if pretrained:
            self.pretrained = pretrained
            self.init_weights = False
        # initial the network structure
        self.conv_net, self.CNNlayer_outchannels = self.layers()
        if self.num_classes:
            if self.dense_clf:
                self.clf_net = self.dense_classifier()
            else:
                self.clf_net = self.CNN_classifier()

    def layers(self):
        layers = []
        input_channels = 3
        for output_channels in self.layer_args:
            if output_channels == 'M':
                layers += [MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
                if self.batch_norm:
                    layers += [conv2d, BatchNorm2d(output_channels), ReLU(inplace=True)]
                else:
                    layers += [conv2d, ReLU(inplace=True)]
                input_channels = output_channels
        return Sequential(*layers), output_channels

    def dense_classifier(self):
        classifier = Sequential(
            AdaptiveAvgPool2d((7, 7)),
            Flatten(),
            Linear(512 * 7 * 7, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, self.num_classes),
            # Softmax()
        )
        return classifier

    def CNN_classifier(self):
        CNN_classifier = Sequential(
            AdaptiveAvgPool2d((7, 7)),
            Conv2d(512, 4096, kernel_size=7),
            ReLU(inplace=True),
            Dropout2d(),
            
            Conv2d(4096, 4096, 1),
            ReLU(inplace=True),
            Dropout2d(),

            Conv2d(4096, self.num_classes, 1),
            Flatten()
            # ConvTranspose2d(self.num_classes, self.num_classes, 512, stride=256, bias=False)
        )
        return CNN_classifier

    def forward(self, x):
        x_middle = []
        for i, net in enumerate(self.conv_net):
            x_middle.append(x)
            x = net(x)
        x_middle.append(x)
        if self.num_classes:
            x = self.clf_net(x)
            return x, x_middle
        return x, x_middle

device = torch.device('cuda')
# x = torch.rand(20, 3, 512, 512).to(device)
vgg = VGG().to(device)
# print(vgg.dense_clf)
# y, y_middle = vgg(x)
# print(y.shape)
from torchsummary import summary
summary(vgg, (3, 512, 512))
# # # for i in y_middle:
# # #     print(i.shape)
# # import os
# # os.system("nvidia-smi")
# print(y.shape)
# # # y = input()