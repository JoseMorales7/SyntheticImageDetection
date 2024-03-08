# %%
import torch
from torch.utils.data import Dataset, DataLoader

from skimage import io

import numpy as np
import torch
import os
from PIL import Image, ImageOps, ImageFilter
import random
import time
import pandas as pd



# %%
class ImageData(Dataset):
    test_generators = ['ddpm', 'gau_gan', 'pallet', 'cips']
    real_data = ['afhq', 'celebahq', 'coco', 'imagenet', 'landscape', 'lsun', 'metfaces', '']

    def __init__(self, dataParentFolder: str, dataIdxs: list, transform = None, batch_size=128):
        self.dataParentFolder = dataParentFolder
        self.transform = transform

        imagePaths = []
        for dataset in os.listdir(dataParentFolder):
            path = os.path.join(dataParentFolder, dataset)
            df = pd.read_csv(os.path.join(path, 'metadata.csv'))
            imageFiles = df['image_path'].tolist()
            imagePaths.extend([os.path.join(path, imageFile).replace("\\", "/") for imageFile in imageFiles])
            
        self.imagePaths = np.array(imagePaths)[dataIdxs]

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imagePath = self.imagePaths[idx]
        pathSplits = imagePath.split(self.dataParentFolder)[1].split("/")
        dataset = pathSplits[0]
        dataset =self.image_to_vector(dataset)
        image = io.imread(imagePath)
        if self.transform:
            image = self.transform(image)
    
        return image, dataset
    
    # TODO: 
    def image_to_vector(self, dataset):
        output = np.zeros(10)
        # for i in range(len(city)):
        #     if city == 'Atlanta':
        #         output[0] = 1
        #     elif city == 'Austin':
        #         output[1] = 1
        #     elif city == 'Boston':
        #         output[2] = 1
        #     elif city == 'Chicago':
        #         output[3] = 1
        #     elif city == 'LosAngeles':
        #         output[4] = 1
        #     elif city == 'Miami':
        #         output[5] = 1
        #     elif city == 'NewYork':
        #         output[6] = 1
        #     elif city == 'Phoenix':
        #         output[7] = 1
        #     elif city == 'SanFrancisco':
        #         output[8] = 1
        #     elif city == 'Seattle':
        #         output[9] = 1
        return torch.tensor(output)


# %%
def getImagesDataLoader(dataParentFolder: str, batchSize: int = 128, transforms = None):
    datasetIdxs = [0]
    totalPoints = 0
    test_generators = ['ddpm', 'gau_gan', 'pallet', 'cips']
    testGeneratorIdxs = [0]
    for dataset in os.listdir(dataParentFolder):
        if dataset in test_generators:
            testGeneratorIdxs.append(totalPoints)
        df = pd.read_csv(os.path.join(dataParentFolder, dataset, 'metadata.csv'))
        totalPoints += len(df['image_path'].tolist())
        datasetIdxs.append(totalPoints)

    trainIdxs = []
    validIdxs = []
    testIdxs = []
    for i in range(len(datasetIdxs) - 1):
        start = datasetIdxs[i]
        stop = datasetIdxs[i + 1]
        idxs = np.arange(start, stop)

        if start in testGeneratorIdxs:
            testIdxs.extend(idxs)
        else:
            num_train = int(np.round((stop - start) / 100 * 80))
            num_valid = int(round(num_train * 0.05))

            num_train -= num_valid
            num_valid += num_train

            # Shuffle all training stimulus image
            np.random.shuffle(idxs)

            # Assign 80% of the shuffled stimulus images for each city to the training partition,
            # and 20% to the test partition
            trainIdxs.extend(idxs[:num_train])
            validIdxs.extend(idxs[num_train:num_valid])
            testIdxs.extend(idxs[num_valid:])

    trainData = ImageData(dataParentFolder, trainIdxs, transform=transforms)
    validData = ImageData(dataParentFolder, validIdxs, transform=transforms)
    testData = ImageData(dataParentFolder, testIdxs, transform=transforms)

    trainDataLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True)
    validDataLoader = DataLoader(validData, batch_size=batchSize, shuffle=True)
    testDataLoader = DataLoader(testData, batch_size=batchSize, shuffle=True)

    return trainDataLoader, validDataLoader, testDataLoader
    
