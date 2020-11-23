'''
Author: bg
Date: 2020-11-11 21:55:39
LastEditTime: 2020-11-23 15:03:14
LastEditors: bg
Description: 
FilePath: /FCN-semantic-segmentation/src/resnet.py
'''
import torch
from torch import nn, optim
import torch.nn.functional as F
import time


class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


if __name__ == "__main__":
    device = torch.device('cuda')
    resblock = Residual(3,3)
    x = torch.rand((4, 3, 6, 6))
    y = resblock(x)
    print(x.shape, y.shape)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('/home/spierb/workspace/FCN-semantic-segmentation/summary')
    for i in range(10000):
        writer.add_scalar('loss', i, i)
    