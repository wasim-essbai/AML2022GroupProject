from torch.utils.data import Dataset
from skimage import io
import cv2
import pandas as pd
import torch
import os
import numpy as np


class FunctionsDataset(Dataset):
    def __init__(self, csv_file, root_dir, label_dim, transform=None):
        self.function_labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_dim = label_dim

    def __len__(self):
        return len(self.function_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.function_labels.iloc[idx, 0])
        image = io.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=2)
        labels = self.function_labels.iloc[idx, 1:]
        labels = np.array([labels])
        labels = labels.astype('int').reshape(-1, self.label_dim)
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample
