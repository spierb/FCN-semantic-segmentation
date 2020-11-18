'''
Author: 
Date: 2020-11-17 17:22:13
LastEditTime: 2020-11-17 19:32:21
LastEditors: bg
Description: 
FilePath: /FCN-semantic-segmentation/src/test.py
'''
import torch

loss = torch.nn.BCELoss()

a = torch.randint(0,5,(3,3))
print(a)

a_onehot = torch.nn.functional.one_hot(a).permute(2, 0, 1)
a_onehot = torch.argmax(a_onehot, 0)
print(a_onehot)