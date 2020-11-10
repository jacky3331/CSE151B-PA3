import torch
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
import pandas as pd
import os
from torchvision import transforms, utils

from skimage import io, transform


# Dataset class to preprocess your data and labels
# You can do all types of transformation on the images in this class


class bird_dataset(Dataset):
    # You can read the train_list.txt and test_list.txt files here.
    def __init__(self,root,file_path):
        # pass
        # f = open(root + file_path)
        # Lines = f.readlines()
        # my_data = [line.strip("\n").split(" ") for line in Lines]
        # my_data = [[my_data[index][0], int(my_data[index][1])] for index in range(len(my_data))]
        
        # self.data = [[torch.tensor(np.array(Image.open(root + '/images/' + my_data[index][0]))), \
        #     int(my_data[index][1])] for index in range(len(my_data))]
        self.data = pd.read_csv(
            root + file_path, sep=" ", names=["name", "class"], header = None)
        self.root = root + '/images/'


    def __len__(self):
        self.length = len(self.data)
        return self.length

    # Reshape image to (224,224).
    # Try normalizing with imagenet mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] or
    # any standard normalization
    # You can other image transformation techniques too
    def __getitem__(self, item):
        # return self.data[item]
        # if torch.is_tensor(item):
        #     item = item.tolist()
        if torch.is_tensor(item):
            item = item.tolist()
        
        img_name = os.path.join(self.root,
                                self.data.iloc[item, 0])

        image = io.imread(img_name)

        classification = self.data.iloc[item, 1:].values[0]
        # classification = np.array([classification])
        #reshape rgb image to 1 dimensional?
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        data_transform = transforms.Compose([transforms.ToTensor(), transforms.RandomSizedCrop(224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = data_transform(image)

        # sample = {'image': image, 'classification': classification}

        sample = (image, classification)


        return sample
    
    # def normalization(self):


    # Testing if the code works
    # def printstuff(self):
    #     return self.data[0][0]
