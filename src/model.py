'''
Author: bg
Date: 2020-11-16 20:04:14
LastEditTime: 2020-11-18 14:58:14
LastEditors: bg
Description: 
FilePath: /FCN-semantic-segmentation/src/model.py
'''
import torch
from torch import nn
import torch.nn.functional as F
from segnet import SegNet
from data import build_dataloader, show_onehot_img
import os
import time
import re
import sys
from PIL import Image
import json

class Model():
    def __init__(self, opt):
        self.opt = opt
        self.device = self.opt.device
        self.net = SegNet(opt.num_classes, batch_norm=True).to(self.device)
        if self.opt.dataset == 'CityscapesDataset':
            train_dst, val_dst, test_dst = build_dataloader(opt)
            self.dataloader = {'train_dst':train_dst, 'val_dst':val_dst, 'test_dst':test_dst}
        # self.loss = nn.BCELoss().to(self.device)
        self.loss = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.opt.lr, momentum=self.opt.momentum, weight_decay=self.opt.weight_decay)
        if self.opt.pretrained:
            self.load_model()
        self.print_summary()

    def save_model(self, epoch, t):
        file_dir = os.path.join(self.opt.results_dir, 'models', t)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        torch.save(self.net.state_dict(), os.path.join(file_dir, "net_epoch_%d.pth" % epoch))
    
    def load_model(self):
        self.net.load_state_dict(torch.load(self.opt.pretrained))

    def train_epoch(self, epoch, t):
        start = time.time()
        self.net.train()
        loss_list = []
        batch_num = 0
        for i, (raw, target_onehot, target_raw) in enumerate(self.dataloader['train_dst']):
            self.optimizer.zero_grad()
            output = self.net(raw)
            loss = self.loss(output, target_raw)
            loss_list.append(float(loss))
            print('\r Training epoch %d, batch %d, loss: %f, time: %fs' % (epoch, i, float(loss), time.time()-start), end='')
            sys.stdout.flush()
            loss.backward()
            self.optimizer.step()
            batch_num = i+1
        self.save_model(epoch, t)
        print('\r Train epoch %d finished! Totally %d batches, using time %f s.                                    ' % (epoch, batch_num, time.time()-start))
        return loss_list


    def train(self):
        log_time_list = re.findall('\w+', time.asctime(time.localtime(time.time())))
        t = log_time_list[1] + '_' + log_time_list[2] + '_' + log_time_list[3] + '_' + log_time_list[4]
        for i in range(self.opt.epoch):
            train_loss_list = self.train_epoch(i+1, t)
            val_loss_mean = self.val(i+1, t)
            self.write_log(i+1, t, train_loss_list, val_loss_mean)
    
    def val(self, epoch, t):
        self.net.eval()
        loss_list = []
        for i, (raw, target_onehot, target_raw) in enumerate(self.dataloader['val_dst']):
            output = self.net(raw)
            loss = self.loss(output, target_raw)
            loss_list.append(float(loss))
            print('\r evaluating epoch %d, batch %d, loss: %f' % (epoch, i, float(loss)), end='')
            sys.stdout.flush()
        loss_mean = sum(loss_list) / len(loss_list)
        print('\r mean val loss of epoch %d is %f                          ' % (epoch, loss_mean))
        return loss_mean

    def test(self):
        self.net.eval()
        loss_list = []
        for i, (raw, target_onehot, target_raw) in enumerate(self.dataloader['test_dst']):
            output = self.net(raw)
            loss = self.loss(output, target_raw)
            # loss = self.loss(F.sigmoid(output.to(torch.float)), target.to(torch.float))
            loss_list.append(float(loss))
            print('\r testing epoch %d, batch %d, loss: %f' % (epoch, i, float(loss)), end='')
            sys.stdout.flush()
        loss_mean = sum(loss_list) / len(loss_list)
        print('\r mean test loss of epoch %d is %f                          ' % (epoch, loss_mean))
        return loss_mean



    def write_log(self, epoch, t, train_loss_list, val_loss_mean):
        self.get_result_img(epoch, t)
        log_file_dir = os.path.join(self.opt.logs_dir , t)
        log = {
            'time' : t,
            'epoch' : epoch,
            'val_loss_mean' : val_loss_mean,
            'train_loss_list' : train_loss_list
            }
        if not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir)
        jsobj = json.dumps(log)
        fileobj = open(os.path.join(log_file_dir, "net_epoch_%d_log.json" % epoch), 'w')
        fileobj.write(jsobj)
        fileobj.close()
        

    def print_summary(self):
        from torchsummary import summary
        smry = summary(self.net, (3,self.opt.crop,self.opt.crop))
        print(smry)
        return smry
    
    def get_result_img(self, epoch=0, t=0, show=False):
        self.net.train()
        for i, (raw, target_onehot, target_raw) in enumerate(self.dataloader['test_dst']):
            output = self.net(raw)
            # print(output)
            raw_img = show_onehot_img(raw[0], show=show)
            target_img = show_onehot_img(target_onehot[0], show=show)
            output_img = show_onehot_img(output[0], show=show)
            result = Image.new(raw_img.mode, (self.opt.crop*3, self.opt.crop))
            result.paste(raw_img, box=(0, 0))
            result.paste(target_img, box=(self.opt.crop, 0))
            result.paste(output_img, box=(self.opt.crop*2, 0))
            if show:
                result.show()
            if not (epoch == 0) & (t == 0):
                file_dir = os.path.join('results/image_results', str(t), str(epoch))
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                result.save(os.path.join(file_dir, str(i) + '.jpg'))
            if i == 9:
                break



