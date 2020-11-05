import torch
from torch.utils.data import Dataset

from PIL import Image
import numpy as np

# Dataset class to preprocess your data and labels
# You can do all types of transformation on the images in this class


class bird_dataset(Dataset):
    # You can read the train_list.txt and test_list.txt files here.
    def __init__(self,root,file_path):
        # pass
        f = open(root + file_path)
        Lines = f.readlines()
        my_data = [line.strip("\n").split(" ") for line in Lines]
        my_data = [[my_data[index][0], int(my_data[index][1])] for index in range(len(my_data))]
        # self.data = my_data 
        
        self.data = [[np.array(Image.open(root + '/images/' + my_data[index][0])), \
            int(my_data[index][1])] for index in range(len(my_data))]
        
        # np.array(Image.open(root + '/images/' + '044.Frigatebird/Frigatebird_0057_43016.jpg'))
    # print(np.array(image).shape)

    def __len__(self):
        self.length = len(self.data)
        return self.length

    # Reshape image to (224,224).
    # Try normalizing with imagenet mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] or
    # any standard normalization
    # You can other image transformation techniques too
    def __getitem__(self, item):
        return self.data[item]

    def normalization


    # Testing if the code works
    # def printstuff(self):
    #     return self.data[0][0]
    """
    Need to implement an accuracy method.
    def evalute_accuracy():
    """