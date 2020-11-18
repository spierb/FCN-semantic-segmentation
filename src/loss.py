'''
Author: bg
Date: 2020-11-18 12:31:38
LastEditTime: 2020-11-18 15:14:20
LastEditors: bg
Description: loss functions
FilePath: /FCN-semantic-segmentation/src/loss.py
'''
import torch
import torch.nn.functional as F
import torch.nn as nn

loss_index = {
    'CrossEntropyLoss' : nn.CrossEntropyLoss(),
    'MSELoss': nn.BCELoss()    
}