from torch.utils.data import Dataset, DataLoader
import torch
import math
import random
import numpy as np
import cv2
import albumentations

classListArray = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

classCount = len(classListArray)

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

color_map_c = color_map()



class PascalVOCSegmentation(Dataset):
    def composeAugmentation(self):
        if self.source == 'train':
            self.augment = albumentations.Compose(
            [
                albumentations.LongestMaxSize(256, always_apply=True),
                albumentations.PadIfNeeded(256, 256, cv2.BORDER_CONSTANT, 0),
                albumentations.HorizontalFlip(),
            ])
        else:
            self.augment = albumentations.Compose(
            [
                albumentations.LongestMaxSize(256, always_apply=True),
                albumentations.PadIfNeeded(256, 256, cv2.BORDER_CONSTANT, 0),
            ])

    def __init__(self, source='train', random=False):
        self.classList = classListArray
        self.classCount = len(self.classList)

        f = open('data/seg_' + source + '.txt', 'r')
        self.labels = f.readlines()
        self.total = len(self.labels)
        self.source = source
        self.random = random

        self.sizeX = 256
        self.sizeY = 256

        self.composeAugmentation()
    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if (self.random == True):
            sample = random.randint(0, self.total-1)
        else:
            sample = idx

        # Images
        imageName = self.labels[sample].replace('\n', '')

        image = cv2.imread('../VOC2012/JPEGImages/' + imageName + '.jpg') / 255.0
        label = cv2.imread('../VOC2012/SegmentationClass/' + imageName + '.png') / 255.0
        transform = self.augment(image=image, mask=label)

        image = transform['image']
        label = transform['mask']

        gray = cv2.cvtColor(label.astype(np.float32), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0.01, 1, cv2.THRESH_BINARY)

        label_image_32 = cv2.resize(thresh, (16, 16))
        label_image_32_r = cv2.resize(label_image_32, (256, 256), None, 0, 0, cv2.INTER_LINEAR)
        label_image_16 = cv2.resize(thresh, (8, 8))
        label_image_16_r = cv2.resize(label_image_16, (256, 256), None, 0, 0, cv2.INTER_LINEAR)

        #diff = label_image_16_r - label_image_32_r
        diff = label_image_32_r
        diff = diff[:, :, np.newaxis]

        inputs = np.zeros((self.sizeX, self.sizeY, 4))
        inputs[:, :, :3] = image
        inputs[:, :, 3] = label_image_16_r

        #cv2.imshow('inputs_1_3', inputs[:, :, :3])
        #cv2.imshow('inputs_3_3', inputs[:, :, 3])
        #cv2.imshow('label', label_image_32_r)
        #cv2.imshow('diff', diff)
        #cv2.waitKey(0)

        return (inputs, diff)
