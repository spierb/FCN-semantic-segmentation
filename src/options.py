'''
Author: bg
Date: 2020-11-11 22:05:27
LastEditTime: 2020-11-23 15:33:05
LastEditors: bg
Description: 
FilePath: /FCN-semantic-segmentation/src/options.py
'''
import argparse
import os

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        #sys options
        self.parser.add_argument('--train', default=True, help='Train/test')
        self.parser.add_argument('--dataset', default='CityscapesDataset', help='Dataset used.')
        self.parser.add_argument('--num_classes', default=20, help='the num of the network segmentation')
        self.parser.add_argument('--crop', default=512)
        self.parser.add_argument('--device', default='cuda')
        # self.parser.add_argument('--pretrained', default='/home/spierb/workspace/FCN-semantic-segmentation/results/models/Nov_17_16_32/net_epoch_3.pth')
        self.parser.add_argument('--pretrained', default='')
        self.parser.add_argument('--random_seed', default=42)
        
        #train options
        self.parser.add_argument('--batch_size', default=20)
        self.parser.add_argument('--epoch', default=10)
        self.parser.add_argument('--loss', default='CrossEntropyLoss')
        self.parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
        self.parser.add_argument('--momentum', type=float, default=0, help='Momentum')
        self.parser.add_argument('--weight_decay', type=float, default=2e-4, help='Weight decay')
        self.parser.add_argument('--data_dir', default='data/')
        self.parser.add_argument('--logs_dir', default='logs/')
        self.parser.add_argument('--results_dir', default='results/')
        self.parser.add_argument('--summary_dir', default='summary/')

        self.opt = None

    def parse(self):
        self.opt = self.parser.parse_args()
        self.check_dirs(self.opt)
        return self.opt

    def check_dirs(self, opt):
        print('checking workspaces...')
        if not os.getcwd().endswith('FCN-semantic-segmentation'):
            print('Running in the wrong dir!')
        if not os.path.exists(opt.data_dir):
            print("data file doesn't exist")
        if not os.path.exists(opt.logs_dir):
            print('creating logs/ dir...')
            os.mkdir('logs')
        if not os.path.exists(opt.results_dir):
            print('creating results/ dir...')
            os.mkdir('logs')

