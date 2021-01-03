import torch
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
import pandas as pd
import os
from torchvision import transforms, utils

from skimage import io, transform


# Dataset class to preprocess your data and labels
class bird_dataset(Dataset):
    # You can read the train_list.txt and test_list.txt files here.
    def __init__(self,root,file_path):
        self.data = pd.read_csv(
            root + file_path, sep=" ", names=["name", "class"], header = None)
        self.root = root + '/images/'


    def __len__(self):
        self.length = len(self.data)
        return self.length

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        
        img_name = os.path.join(self.root,
                                self.data.iloc[item, 0])
        
        image = Image.open(img_name)
        image = image.convert('RGB')

        classification = self.data.iloc[item, 1:].values[0]
#         data_transform = transforms.Compose([transforms.RandomResizedCrop(224),  transforms.RandomHorizontalFlip(), transforms.ToTensor(),
#                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
#         image = data_transform(image)

        sample = (image, classification)


        return sample
