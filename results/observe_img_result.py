import torch
import numpy as np
from PIL import Image

def cut_result(img_path):
    img = Image.open(img_path)
    img_arr = np.array(img)
    raw = img_arr[:, :512, :].transpose(2,1,0)
    target = img_arr[:, 512:(512*2), :].transpose(2,1,0)
    output = img_arr[:, (512*2):, :].transpose(2,1,0)
    return raw, target, output
