import torch
from torch.utils.data import Dataset

# Dataset class to preprocess your data and labels
# You can do all types of transformation on the images in this class


class bird_dataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        raise ("Not implemented")

    def __getitem__(self, item):
        raise ("Not implemented")