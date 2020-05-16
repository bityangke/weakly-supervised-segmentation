from torch.utils.data import DataLoader
from data import PascalVOCSegmentation
from data import classListArray
from data import color_map_c
from unet import model

import numpy as np
import cv2
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pascal_val = PascalVOCSegmentation(source='val')
data = DataLoader(pascal_val, batch_size=4, shuffle=False, num_workers=0)

model.eval()
count = 0
for images, labels in data:
    inputs = images.permute(0, 3, 1, 2)

    inputs = inputs.float()
    labels = labels.float()

    # Predict
    outputs = model(inputs)

    # Convert to numpy
    images_np = images.numpy()
    labels_np = labels.numpy()
    outputs_np = outputs.data.numpy()

    
    for sample in range(0, outputs_np.shape[0]):
        inputs_1 = images_np[sample, :, :, :3]
        inputs_2 = images_np[sample, :, :, 3]
        inputs_2 = inputs_2[:, :, np.newaxis]
        output = outputs_np[sample, 0]
        label = labels_np[sample]

        output = output[:, :, np.newaxis]

        final = np.zeros((256, 256*4, 3))
        final[:, 0:256, :] = inputs_1
        
        final[:, 256:512, :] = inputs_2
        final[:, 512:768, :] = output
        final[:, 768:1024, :] = label

        final[:, 254:256, :] = (0, 0, 1)
        final[:, 510:512, :] = (0, 0, 1)
        final[:, 766:768, :] = (0, 0, 1)

        cv2.imwrite('output/' + str(count) + '.jpg', final * 255)
        count += 1
        cv2.imshow('final', final)
        cv2.waitKey(10)
        if (count >= 40):
            exit()
