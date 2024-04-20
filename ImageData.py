# %%
import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import numpy as np
import torch
import os
from PIL import Image, ImageOps, ImageFilter
import random
import time
import pandas as pd

from sklearn.model_selection import train_test_split



# %%
class ImageData(Dataset):
    # test_generators = ['ddpm', 'gau_gan', 'pallet', 'cips']
    # real_data = ['afhq', 'celebahq', 'coco', 'imagenet', 'landscape', 'lsun', 'metfaces', '']

    def __init__(self, dataParentFolder: str, dataIdxs: list = None, transform = None, batch_size=128):
        self.dataParentFolder = dataParentFolder
        self.transform = transform

        imagePaths = []
        imageDataset = []
        fake = []
        for dataset in sorted(os.listdir(dataParentFolder)):
            path = os.path.join(dataParentFolder, dataset)
            df = pd.read_csv(os.path.join(path, 'metadata.csv'))
            imageFiles = df['image_path'].tolist()
            imagePaths.extend([os.path.join(path, imageFile).replace("\\", "/") for imageFile in imageFiles])
            imageDataset.extend([dataset] * len(imageFiles))
            fake.extend(df["target"])

        self.imagePaths = np.array(imagePaths)
        self.imageDataset = np.array(imageDataset)
        self.fake = np.array(fake)

        if dataIdxs:
            self.imagePaths = self.imagePaths[dataIdxs]
            self.imageDataset = self.imageDataset[dataIdxs]
            self.fake = self.fake[dataIdxs]

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imagePath = self.imagePaths[idx]
        image = Image.open(imagePath)
        if self.transform:
            image = self.transform(image)

        label = 0
        dataset = self.imageDataset[idx]
        if self.fake[idx] > 0:
            label = self.image_to_vector(dataset)

        return image, torch.tensor(label)
    
    def image_to_vector(self, dataset):
        #1 is gan, 2 is diffusion, 3 is other
        labels = {
            'big_gan': 1,
            'cips' : 1,
            'cycle_gan' : 1,
            'ddpm' : 2,
            'denoising_diffusion_gan' : 2,
            'diffusion_gan' : 2,
            'face_synthetics' : 3,
            'gansformer' : 1,
            'gau_gan' : 1,
            'generative_inpainting' : 1,
            'glide' : 2,
            'lama' : 3,
            'latent_diffusion' : 2,
            'mat' : 3,
            'palette' : 2,
            'projected_gan' : 1,
            'pro_gan' : 1,
            'sfhq' : 2,
            'stable_diffusion' : 2,
            'star_gan' : 1,
            'stylegan1' : 1,
            'stylegan2' : 1,
            'stylegan3' : 1,
            'taming_transformer' : 3,
            'vq_diffusion' : 2
        }

        return labels.get(dataset)



# %%
def getImagesDataloaders(dataParentFolder: str, batchSize: int = 128, transforms = None):
    datasetIdxs = [0]
    totalPoints = 0
    test_generators = ['ddpm', 'gau_gan', 'palette', 'cips']
    testGeneratorIdxs = []
    for dataset in sorted(os.listdir(dataParentFolder)):
        if dataset in test_generators:
            testGeneratorIdxs.append(totalPoints)
        df = pd.read_csv(os.path.join(dataParentFolder, dataset, 'metadata.csv'))
        totalPoints += len(df['image_path'].tolist())
        datasetIdxs.append(totalPoints)

    trainIdxs = []
    validIdxs = []
    testIdxs = []
    unseenModelsIdxs = []
    for i in range(len(datasetIdxs) - 1):
        start = datasetIdxs[i]
        stop = datasetIdxs[i + 1]
        idxs = np.arange(start, stop)
        if start in testGeneratorIdxs:
            testIdxs.extend(idxs)
            unseenModelsIdxs.extend(idxs)
        else:
            # Assign 80% of the shuffled stimulus images for each city to the training partition,
            # and 20% to the test partition
            trIdxs, tIdxs = train_test_split(idxs, train_size=0.8, random_state=42)
            trIdxs, vIdxs = train_test_split(trIdxs, train_size=0.95, random_state=42)
            
            trainIdxs.extend(trIdxs)
            validIdxs.extend(vIdxs)
            testIdxs.extend(tIdxs)

    trainData = ImageData(dataParentFolder, trainIdxs, transform=transforms)
    validData = ImageData(dataParentFolder, validIdxs, transform=transforms)
    testData = ImageData(dataParentFolder, testIdxs, transform=transforms)
    unseenData = ImageData(dataParentFolder, unseenModelsIdxs, transform=transforms)

    trainDataLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True)
    validDataLoader = DataLoader(validData, batch_size=batchSize, shuffle=True)
    testDataLoader = DataLoader(testData, batch_size=batchSize, shuffle=True)
    unseenDataLoader = DataLoader(unseenData, batch_size=batchSize, shuffle=True)

    return trainDataLoader, validDataLoader, testDataLoader, unseenDataLoader
    
