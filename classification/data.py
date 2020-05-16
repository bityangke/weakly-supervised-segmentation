from torch.utils.data import Dataset, DataLoader
import torch
import math
import random
import numpy as np
import cv2
import albumentations

classListArray = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class PascalVOCClassification(Dataset):
    def composeAugmentation(self):
        if self.source == 'train':
            self.augment = albumentations.Compose(
            [
                # albumentations.Rotate(5, always_apply=True),
                # albumentations.GridDistortion(distort_limit=0.05, always_apply=True),
                # albumentations.RandomBrightnessContrast(),
                albumentations.LongestMaxSize(256, always_apply=True),
                albumentations.PadIfNeeded(256, 256, cv2.BORDER_CONSTANT, 0),
                # albumentations.HorizontalFlip(),
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

        f = open('./data/' + source + '.txt', 'r')
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

        parts = self.labels[sample].split(' ')

        # Image
        imageName = parts[0]
        image = cv2.imread('../VOC2012/JPEGImages/' + imageName + '.jpg')

        et = self.augment(image=image)
        # cv2.imwrite('data/img' + str(idx + b) + '.jpg', et['image'])
        image = et['image'] / 255.0

        # Label
        label = np.zeros(shape=(self.classCount))
        labelParts = parts[1].split('|')
        for lpart in labelParts:
            lpart = lpart.replace('\n', '')
            label[self.classList.index(lpart)] = 1

        return (image, label)