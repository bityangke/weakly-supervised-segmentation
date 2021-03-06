#
# This file generates train and val text files in the data folder which is used for training
#

import cv2
import numpy as np
from data import classListArray

labels = {}

def buildTextFile(source):
    textFile = open('./data/' + source + '.txt', 'w')
    for className in classListArray:
        f = open('../VOC2012/ImageSets/Main/' +className+ '_' + source + '.txt', 'r')
        lines = f.readlines()

        for line in lines:
            line = line.replace('\n', '')
            parts = line.split(' ')

            imageName = parts[0]

            if (len(parts) > 2):
                if (parts[2] == '1'):
                    if imageName in labels:
                        labels[imageName] += '|' + className
                    else:
                        labels[imageName] = className

    for key, value in labels.items():
        textFile.write(key + ' ' + value +'\n')

    textFile.close()

buildTextFile('train')
buildTextFile('val')