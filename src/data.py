'''
Author: bg
Date: 2020-11-11 16:25:06
LastEditTime: 2020-11-11 21:19:16
LastEditors: bg
Description: data process/datasets prepare.
FilePath: /FCN-semantic-segmentation/src/data.py
'''
import os
import time
import random
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

CityscapesDataset_default_root = os.path.join(os.getcwd(), 'data', 'cityscapes')
PascalVocDataset_default_root = os.path.join(os.getcwd(), 'data', 'pascal_voc')

def PIL2Tensor(img):
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transformer(img)
    return img, img.shape

class CityscapesDataset(Dataset):
    def __init__(self, root=CityscapesDataset_default_root, split='train', crop=512, flip=False):
        super(CityscapesDataset, self).__init__()
        self.root = root # /home/spierb/workspace/FCN-semantic-segmentation/data/cityscapes
        self.split = split # train, test, val
        self.crop = crop # 512
        self.flip = flip # False
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
        self.num_classes = 20 # TODO
        self.all2labelIds = {-1: 19, 0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19, 7: 0, 8: 1, 9: 19, 10: 19, 11: 2, 12: 3, 13: 4, 14: 19, 15: 19, 16: 19, 17: 5, 18: 19, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 19, 30: 19, 31: 16, 32: 17, 33: 18}
        self.labelIds2all = {0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33, 19: 0}
        self.colormap = {0: (0, 0, 0), 7: (128, 64, 128), 8: (244, 35, 232), 11: (70, 70, 70), 12: (102, 102, 156), 13: (190, 153, 153), 17: (153, 153, 153), 19: (250, 170, 30), 20: (220, 220, 0), 21: (107, 142, 35), 22: (152, 251, 152), 23: (70, 130, 180), 24: (220, 20, 60), 25: (255, 0, 0), 26: (0, 0, 142), 27: (0, 0, 70), 28: (0, 60,100), 31: (0, 80, 100), 32: (0, 0, 230), 33: (119, 11, 32)}

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        raw, target= Image.open(self.raw[i]), Image.open(self.target[i])
        if self.crop:
            x_crop, y_crop = random.randint(0, raw.size[0] - self.crop), random.randint(0, raw.size[1] - self.crop)
            raw, target = raw.crop((x_crop, y_crop, x_crop + self.crop, y_crop + self.crop)), target.crop((x_crop, y_crop, x_crop + self.crop, y_crop + self.crop))
        if (self.flip & bool(random.random() < 0.5)):
            raw, target = raw.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT)
        # raw.show()
        h, w = raw.size[0], raw.size[1]
        raw, _ = PIL2Tensor(raw)
        #############################################################################################################################
        target = torch.ByteTensor(torch.ByteStorage.from_buffer(target.tobytes())).view(target.size[0], target.size[1]).long()
        remapped_target = target.clone()
        for k, v in self.all2labelIds.items():
            remapped_target[target == k] = v
            # Create one-hot encoding
        target = torch.zeros(self.num_classes, h, w)
        for c in range(self.num_classes):
            target[c][remapped_target == c] = 1
        # return input, target, remapped_target  # Return x, y (one-hot), y (index)
        ##############################################################################################################################TODO
        return raw, target

class PascalVocDataset(Dataset):
    def __init__(self):
        super(PascalVocDataset, self).__init__()

        self.file_list = []

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self):
        return "1 item"

data = CityscapesDataset()
i, t = data.__getitem__(1)

import numpy as np
t = np.array(t)
print(t.shape)
# import numpy as np
# print(t)
# t = np.array(t).astype(np.uint8)*(255/33) 
# print(t.shape)
# t = Image.fromarray(t)
# t.show()
