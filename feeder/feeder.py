import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from pathlib import Path
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from utils.cityscape import colors, encode_mask

class Dataset(Dataset):
    def __init__(self, path_data, transform = None):
        self.path_data = path_data
        self.img_list = glob(path_data + "/*.jpg")
        self.transform = transform
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        self.img_and_label = cv2.imread(self.img_list[idx])
        self.img_and_label = cv2.cvtColor(self.img_and_label, cv2.COLOR_BGR2RGB)
        self.img = self.img_and_label[:, :256, :]
        self.label = self.img_and_label[:, 256:, :]
        # self.label = b 
        return self.img, self.label

a = Dataset(path_data = "data/cityscapes_data/train")
img, label = a.__getitem__(0)
# print(img.shape)
# print(label.shape)
label_reshape = label.reshape(-1, 3)
colors = np.asarray(colors)
print(label_reshape.shape)
print(colors.shape)
print(label_reshape[:10])
print(colors)
# a = np.sum(abs(label_reshape - colors[2]), axis = 1).reshape(-1, 1)
# print(a[:10])
classes = encode_mask(label_reshape, colors)
print(np.unique(classes))

plt.figure(figsize=(10, 7))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.subplot(1, 3, 2)
plt.imshow(label)
plt.subplot(1, 3, 3)
plt.imshow(classes)
plt.show()