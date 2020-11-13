'''
Author: bg
Date: 2020-11-11 22:05:27
LastEditTime: 2020-11-11 22:08:29
LastEditors: bg
Description: 
FilePath: /FCN-semantic-segmentation/src/params.py
'''
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--nun_classes', default=20, help='the num of the network segmentation')