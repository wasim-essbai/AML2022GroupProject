import matplotlib
import torchvision
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

# matplotlib.use("Agg")
from frC_net import FrCNet
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch import nn
from function_generation.functions_dataset import FunctionsDataset
import torch
import time
import utils

# define training hyperparameters
INIT_LR = 1e-3
EPOCHS = 10
# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT
# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## TEst

function_dataset = FunctionsDataset(csv_file='./function_generation/generated_dataset/function_plot_labels.csv',
                                    root_dir='./function_generation/generated_dataset/data/')

print("Dataset length: ", len(function_dataset))

fig = plt.figure()

for i in range(3):
    sample = function_dataset[i]

    print(i, sample['image'].shape, sample['labels'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample['image'])

plt.show()
