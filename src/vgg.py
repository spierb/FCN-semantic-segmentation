'''
Author: bg
Date: 2020-11-11 21:55:29
LastEditTime: 2020-11-16 11:06:31
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
    #feature maps scale: /2^5
}

class VGG(torch.nn.Module):
    '''
    description: VGG class.\n
    param {*} num_classes: Numbers of classes to classify. The VGG net will output the feature map of the input image.\n
    param {*} batch_norm: Using batch-norm layer between 3x3 conv layers, default False.\n
    param {*} vgg: Depth of VGG net. 8/10/13/16.\n
    param {*} classifier: Using Dense classifier/NiN classifier.\n
    return {*} feature map: [batch_size, 512, input_size/32, input_size/32], classifier:[batch_size, num_classes]\n
    '''
    def __init__(self, num_classes=None, batch_norm=False, vgg=16, classifier='dense'):
        super(VGG, self).__init__()
        self.batch_norm = batch_norm
        self.num_classes = num_classes
        self.layer_args = layermap[vgg]
        self.middle_feature = []
            # self.init_weights = False
        # initial the network structure
        self.features, self.CNNlayer_outchannels = self.layers()
        if self.num_classes:
            if classifier=='dense':
                self.classifier = self.dense_classifier()
            elif classifier=='CNN':
                self.classifier = self.CNN_classifier()

    def layers(self):
        layers = []
        input_channels = 3
        for output_channels in self.layer_args:
            if output_channels == 'M':
                layers += [MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
                if self.batch_norm:
                    layers += conv2d, BatchNorm2d(output_channels)
                    layers += ReLU(inplace=True)
                else:
                    layers += conv2d, ReLU(inplace=True)
                input_channels = output_channels
        return Sequential(*layers), input_channels

    def dense_classifier(self):
        classifier = Sequential(
            # AdaptiveAvgPool2d((7, 7)),
            Flatten(),
            Linear(512 * 16 * 16, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, self.num_classes),
            # Softmax()
        )
        return classifier

    def CNN_classifier(self): # based on NiN block
        CNN_classifier = Sequential(
            # AdaptiveAvgPool2d((7, 7)),
            #the first NiN block
            Conv2d(512, 4096, kernel_size=16),
            ReLU(inplace=True),
            Dropout2d(),
            
            Conv2d(4096, 4096, kernel_size=1),
            ReLU(inplace=True),
            Dropout2d(),

            Conv2d(4096, 4096, kernel_size=1),
            ReLU(inplace=True),

            Dropout2d(),
            # the second NiN block
            Conv2d(4096, self.num_classes, kernel_size=1),
            ReLU(inplace=True),
            Dropout2d(),
            
            Conv2d(self.num_classes, self.num_classes, kernel_size=1),
            ReLU(inplace=True),
            Dropout2d(),

            Conv2d(self.num_classes, self.num_classes, kernel_size=1),
            ReLU(inplace=True),

            Flatten()
        )
        return CNN_classifier

    def forward(self, x):
        self.middle_feature = []
        for i, net in enumerate(self.features):          
            x = net(x)
            if str(type(net))=="<class 'torch.nn.modules.pooling.MaxPool2d'>":
                self.middle_feature.append(x)
        if self.num_classes:
            x = self.classifier(x)
            return x
        return x