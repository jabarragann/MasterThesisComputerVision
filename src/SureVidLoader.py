# -*- coding: utf-8 -*-

from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from PIL import Image


import cv2
import matplotlib.pyplot as plt
import imgaug as ia
from pathlib import Path
import numpy as np
from imgaug import augmenters as iaa

root_dir = "customDataset/"
train_file = os.path.join(root_dir, "train.csv")
val_file = os.path.join(root_dir, "val.csv")

num_class = 5
means = np.array([103.939, 116.779, 123.68]) / 255.  # mean of three channels in the order of BGR

train_h = 384  # 12*32
train_w = 480 # 15*32
val_h = 480  # 704
val_w = 640 # 960


class SureVidDataset(Dataset):

    def __init__(self, csv_file, phase, n_class=num_class, crop=True, flip_rate=0.5, augment_img=False):
        self.data = pd.read_csv(csv_file)
        self.means = means
        self.n_class = n_class

        self.flip_rate = flip_rate
        self.crop = crop
        if phase == 'train':
            self.new_h = train_h
            self.new_w = train_w
        elif phase == 'val':
            self.flip_rate = 0.
            # debug
            # self.crop = False
            self.new_h = val_h
            self.new_w = val_w

        #Data augmentation
        self.augment_img = augment_img

        self.augmentationPipeline = iaa.Sequential([
            iaa.ChannelShuffle(0.50),
            iaa.Add((-30, 30)),
            iaa.AdditiveGaussianNoise(scale=(2, 50)),
            iaa.SaltAndPepper(0.051),
            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)
        ], random_order=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]

        img = Image.open(img_name)
        img = np.array(img)

        label_name = self.data.iloc[idx, 1]
        label = np.load(label_name)

        ##Change color and add noise to augment data
        if self.augment_img:
            img = self.augmentationPipeline(image=img)

        if self.crop:
            h, w, _ = img.shape
            top = random.randint(0, h - self.new_h)
            left = random.randint(0, w - self.new_w)
            img = img[top:top + self.new_h, left:left + self.new_w]
            label = label[top:top + self.new_h, left:left + self.new_w]

        if random.random() < self.flip_rate:
            img = np.fliplr(img)
            label = np.fliplr(label)

        # reduce mean
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1

        sample = {'X': img, 'Y': target, 'l': label}

        return sample


def show_batch(batch):
    img_batch = batch['X']
    img_batch[:, 0, ...].add_(means[0])
    img_batch[:, 1, ...].add_(means[1])
    img_batch[:, 2, ...].add_(means[2])
    batch_size = len(img_batch)

    grid = utils.make_grid(img_batch)
    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))

    plt.title('Batch from dataloader')


if __name__ == "__main__":
    train_data = SureVidDataset(csv_file=train_file, phase='train', augment_img=False, crop=False)

    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['X'].size(), sample['Y'].size())

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(i,"X", batch['X'].size(),"Y", batch['Y'].size())

        # observe 4th batch
        if i == 15:
            plt.figure()
            show_batch(batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
