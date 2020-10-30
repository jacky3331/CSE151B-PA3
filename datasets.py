import torch
from torch.utils.data import Dataset

# Dataset class to preprocess your data and labels
# You can do all types of transformation on the images in this class


class bird_dataset(Dataset):
    # You can read the train_list.txt and test_list.txt files here.
    def __init__(self,root,file_path):
        pass

    def __len__(self):
        raise ("Not implemented")

    # Reshape image to (224,224).
    # Try normalizing with imagenet mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] or
    # any standard normalization
    # You can other image transformation techniques too
    def __getitem__(self, item):
        raise ("Not implemented")