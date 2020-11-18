'''
Author: bg
Date: 2020-11-11 16:25:06
LastEditTime: 2020-11-18 14:54:27
LastEditors: bg
Description: data process/datasets prepare.
FilePath: /FCN-semantic-segmentation/src/data.py
'''
import os
import time
import random
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from options import Options
from torch.utils.data import DataLoader, RandomSampler

CityscapesDataset_default_root = os.path.join(os.getcwd(), 'data', 'cityscapes')
PascalVocDataset_default_root = os.path.join(os.getcwd(), 'data', 'pascal_voc', 'VOCdevkit', 'VOC2012')

class CityscapesDataset(Dataset):
    def __init__(self, root=CityscapesDataset_default_root, split='train', crop=512, flip=False, device='cuda'):
        super(CityscapesDataset, self).__init__()
        self.root = root # /home/spierb/workspace/FCN-semantic-segmentation/data/cityscapes
        self.split = split # train, test, val
        self.crop = crop # 512
        self.flip = flip # False
        self.device = device
        self.raw = []
        self.target = []

        self.raw_dir = os.path.join(self.root, 'leftImg8bit_trainvaltest', split)
        self.target_dir = os.path.join(self.root, 'gtFine_trainvaltest', split)
    
        for file_dir, _, files in os.walk(self.raw_dir):
            for file in files:
                if file.endswith('_leftImg8bit.png'):
                    self.raw.append(os.path.join(file_dir, file))
                    self.target.append(os.path.join(file_dir.replace('leftImg8bit_trainvaltest', 'gtFine_trainvaltest'), file[:-15]) + 'gtFine_labelIds.png')

        self.size = len(self.raw) # 2975
        self.num_classes = 20 
        self.all2labelIds = {-1: 19, 0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19, 7: 0, 8: 1, 9: 19, 10: 19, 11: 2, 12: 3, 13: 4, 14: 19, 15: 19, 16: 19, 17: 5, 18: 19, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 19, 30: 19, 31: 16, 32: 17, 33: 18}
        self.labelIds2all = {0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33, 19: 0}
        self.colormap = {0: (0, 0, 0), 7: (128, 64, 128), 8: (244, 35, 232), 11: (70, 70, 70), 12: (102, 102, 156), 13: (190, 153, 153), 17: (153, 153, 153), 19: (250, 170, 30), 20: (220, 220, 0), 21: (107, 142, 35), 22: (152, 251, 152), 23: (70, 130, 180), 24: (220, 20, 60), 25: (255, 0, 0), 26: (0, 0, 142), 27: (0, 0, 70), 28: (0, 60,100), 31: (0, 80, 100), 32: (0, 0, 230), 33: (119, 11, 32)}
        
        self.raw_transformer = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomCrop(512),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5) if self.flip else transforms.RandomHorizontalFlip(p=0)
        ])
        self.target_transformer = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomCrop(512),
            transforms.RandomHorizontalFlip(p=0.5) if self.flip else transforms.RandomHorizontalFlip(p=0)
        ])

    def __getitem__(self, i):
        raw, target= Image.open(self.raw[i]), Image.open(self.target[i])
        # print(target.getcolors())
        left = int(random.random()*(raw.size[0]-self.crop))
        upper = int(random.random()*(raw.size[1]-self.crop))
        crop_num = [left, upper, left+self.crop, upper+self.crop]
        raw, target= raw.crop(crop_num), target.crop(crop_num)
        raw = self.raw_transformer(raw)
        if target.mode == 'RGB':
            target = target.convert('L')
        target = (self.target_transformer(target)*255).to(torch.long)
        target = target.squeeze()
        target_onehot = target
        for k, v in self.all2labelIds.items():
            target = torch.where(target==k, v, target)
        target_onehot = torch.nn.functional.one_hot(target, 20).permute(2, 0, 1)
        return raw.to(self.device), target_onehot.to(self.device), target.to(self.device)

    def __len__(self):
        return self.size

class PascalVocDataset(Dataset):
    def __init__(self, root=PascalVocDataset_default_root, split='train', crop=256, flip=False, device='cuda'):
        super(PascalVocDataset, self).__init__()
        self.classes_index = {
            0: 'background', 
            1: 'aeroplane', 
            2: 'bicycle', 
            3: 'bird', 
            4: 'boat', 
            5: 'bottle', 
            6: 'bus', 
            7: 'car', 
            8: 'cat', 
            9: 'chair', 
            10: 'cow', 
            11: 'diningtable', 
            12: 'dog', 
            13: 'horse', 
            14: 'motorbike', 
            15: 'person', 
            16: 'potted plant', 
            17: 'sheep', 
            18: 'sofa', 
            19: 'train', 
            20: 'tv/monitor'}
        self.colormap = {
            0:[0,0,0],
            1:[128,0,0],
            2:[0,128,0],
            3:[128,128,0],
            4:[0,0,128],
            5:[128,0,128],
            6:[0,128,128],
            7:[128,128,128],
            8:[64,0,0],
            9:[192,0,0],
            10:[64,128,0],
            11:[192,128,0],
            12:[64,0,128],
            13:[192,0,128],
            14:[64,128,128],
            15:[192,128,128],
            16:[0,64,0],
            17:[128,64,0],
            18:[0,192,0],
            19:[128,192,0],
            20:[0,64,128]}   
        self.crop = crop
        self.split = split
        self.filp = flip
        self.raw, self.target = [], []

    def __filter__(self, file_list):

        return file_list

    def __transforms__(self, raw_img, target_img):

        return raw_img, target_img

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self):
        return "1 item"

def build_dataloader(opt):
    if opt.dataset == 'CityscapesDataset':
        train_dst = CityscapesDataset(split='train', device=opt.device)
        val_dst = CityscapesDataset(split='val', device=opt.device)
        test_dst = CityscapesDataset(split='test', device=opt.device)
        train_loader, val_loader, test_loader = DataLoader(train_dst, batch_size=opt.batch_size, shuffle=True), DataLoader(val_dst, batch_size=opt.batch_size, shuffle=True), DataLoader(test_dst, batch_size=opt.batch_size, shuffle=True)
        return train_loader, val_loader, test_loader
    if opt.dataset == '':
        print('not write yet')

def show_onehot_img(img, show=True):
    if img.shape[0] == 3:
        std= [0.229, 0.224, 0.225]
        mean=[0.485, 0.456, 0.406]
        img[0]=img[0]*std[0]+mean[0]
        img[1]=img[1]*std[1]+mean[1]
        img[2]=img[2]*std[2]+mean[2]
        img = np.array((img.cpu()*255).permute(1,2,0)).astype(np.uint8)       
    else:
        img = np.array((torch.argmax(img.cpu(), 0)*10).to(torch.float)).astype(np.uint8)
    img = Image.fromarray(img)
    if show:
        img.show()
    return img