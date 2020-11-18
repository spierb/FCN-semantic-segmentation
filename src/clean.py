'''
Author: bg
Date: 2020-11-17 16:05:44
LastEditTime: 2020-11-17 16:30:00
LastEditors: bg
Description: clean up the logs/results
FilePath: /FCN-semantic-segmentation/src/clean.py
'''
import os

root = os.getcwd()

def cleanWS(root=root):
    logs_cmd = 'rm -r ' + root + '/logs/*'
    img_cmd = 'rm -r ' + root + '/results/image_results/*'
    model_cmd = 'rm -r ' + root + '/results/models/*'
    os.system(logs_cmd)
    os.system(img_cmd)
    os.system(model_cmd)

cleanWS()