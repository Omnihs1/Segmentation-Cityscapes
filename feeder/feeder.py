import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from utils.cityscape import colors, encode_mask
import random

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RGBShift(),
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.8),
    ToTensorV2(),
])
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
        self.mask = encode_mask(self.label)
        if self.transform is not None:
            transformed = self.transform(image = self.img, mask = self.mask)
            self.img = transformed["image"]
            self.mask = transformed["mask"]
        return self.img, self.label, self.mask

if __name__ == "__main__":
    a = Dataset(path_data = "data/cityscapes_data/train", transform = transform)
    random_number = random.randint(0, len(a.img_list))
    print(random_number)
    print(a.img_list[random_number])
    img, label, mask = a.__getitem__(810)
    print(np.unique(mask))
    print(type(img))
    print(type(mask))
    plt.figure(figsize=(10, 7))
    plt.subplot(1, 3, 1)
    plt.imshow(img.permute(1,2,0).numpy())
    plt.title("Image") 
    plt.subplot(1, 3, 2)
    plt.imshow(label)
    plt.title("Label") 
    plt.subplot(1, 3, 3)
    plt.imshow(mask)
    plt.title("Mask") 
    plt.show()