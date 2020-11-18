'''
Author: bg
Date: 2020-11-18 12:39:18
LastEditTime: 2020-11-18 15:17:18
LastEditors: bg
Description: trainer
FilePath: /FCN-semantic-segmentation/src/trainer.py
'''
import torch
from torch import nn

loss_index = {
    'CrossEntropyLoss' : nn.CrossEntropyLoss(),
    'MSELoss': nn.BCELoss()    
}

class Trainer():
    def __init__(self, opt):
        self.device = opt.device
        self.batch_size = opt.batch_size
        self.lr = opt.lr
        self.momentum = opt.momentum
        self.weight_decay = opt.weight_decay
        self.dirs = {
            'data' : opt.data_dir,
            'logs' : opt.logs_dir,
            'results' : opt.results_dir
        }
        self.criterion = loss_index[opt.loss].to(self.device)
        

    def train(self):
        print('train')

    def loss(self, output, target):
        loss = self.criterion(output, target)
        return loss



from data import build_dataloader
from options import Options

opt = Options().parse()

train_loader, _, _ = build_dataloader(opt)
trainer = Trainer(opt)

for i, (raw, target_onehot, target_raw) in enumerate(train_loader):
    x = torch.randn((20,20,512,512), requires_grad=True).cuda()
    print(torch.max(target_raw), torch.min(target_raw))
    loss = trainer.loss(x, target_raw)
    print(loss)
    break


        
