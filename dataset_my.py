import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import pickle
import random
class DatasetPorcessing_h5(Dataset):
    def __init__(self, train_data, train_y, train_label, transform=None, is_train=False):
        self.train_data = train_data
        self.is_train = is_train
        self.transform = transform
        self.train_y = train_y
        self.labels = torch.tensor(train_label).float()

    def __getitem__(self, index):
        img = Image.fromarray(self.train_data[index])
        if self.transform is not None:
            img1 = self.transform(img)
        label = self.labels[index]
        y_vector = torch.Tensor(self.train_y[index]).float()
        if self.is_train:
            return img1, y_vector, label, index
        else:
            return img1, y_vector, label, index
    def __len__(self):
        return self.labels.shape[0]

if __name__=='__main__':
    dd = DatasetPorcessing('/apdcephfs/share_1367250/rongchengtu/coco', 'database')
    dd.get_labels()
    print(dd.num_class)