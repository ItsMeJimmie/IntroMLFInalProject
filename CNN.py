#This is using a CNN with U-Net
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize,ToTensor,Compose
import torchvision.transforms.functional as TF
from torchinfo import summary

import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import matplotlib.pyplot as plt

#Directory that holds all the images
path = r'\\wsl.localhost\Ubuntu\home\itsmejimmie\IntroMLFInalProject\datasets'

#Preprocessing transformations, basically we are ressizing the images to 224x224 and converting to tensor
transforms = Compose([Resize((224, 224)), ToTensor()])

#Class used to hold functions
class TSegDataset(Dataset):

    #This is basically our constructor method
    def __init__(self, img_path: str, mask_path: str, transforms=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transforms = transforms

        #Get the list of all terrain images
        self.img_files = sorted(os.listdir(img_path))

        # We expect the terrain image to have a suffix '_t' and the corresponding mask to have suffix '_i2'
        self.img_names = [f.split('_')[0] for f in self.img_files if '_t' in f]  #All unique terrain image prefixes

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        #Get the base name (without suffix)
        base_name = self.img_names[idx]

        #Load the terrain image (_t suffix)
        img = Image.open(os.path.join(self.img_path, f'{base_name}_t.png'))

        #Load the corresponding mask image (_i2 suffix)
        mask = Image.open(os.path.join(self.mask_path, f'{base_name}_i2.png'))

        #Apply transformations
        if self.transforms:
            img = self.transforms(img)
            mask = self.transforms(mask)

        #Convert mask to class labels (if the mask is one-hot encoded)
        mask = torch.argmax(mask, dim=0)  #Assuming the mask has shape (C, H, W), where C is the number of classes

        #Returning as a frame without specifying as a frame
        return {'img': img, 'mask': mask}

dataset = TSegDataset(img_path=path, mask_path=path, transforms=transforms)

# Access the data by index
sample = dataset[2]

img = sample['img'].permute(1, 2, 0)  # Convert to HxWxC for plotting
mask = sample['mask'].numpy()

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Image')

plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.title('Mask')

plt.show()

#Then we need to setup the training and testing size
#train_size =
#test_size =

#Then we do the actual split using in this case random_split but we can do something
#different and then showing the size of the train and test data
train_data, test_data = random_split(dataset, [train_size, test_size])
len(train_data), len(test_data)



