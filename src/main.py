'''
Author: bg
Date: 2020-11-16 10:24:56
LastEditTime: 2020-11-23 15:15:03
LastEditors: bg
Description: main
FilePath: /FCN-semantic-segmentation/src/main.py
'''
from options import Options
import random
from model import Model

if __name__ == '__main__':
    opt = Options().parse()
    random.seed(opt.random_seed)
    model = Model(opt)
    model.train()