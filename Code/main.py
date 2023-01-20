import matplotlib

matplotlib.use("Agg")

from frC_net import FrCNet
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
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

# load the dataset
print("[INFO] loading the dataset...")
full_dataset = utils.get_data()
trainData, testData, valData = utils.get_data_split(full_dataset)

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True,
                             batch_size=utils.BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=utils.BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=utils.BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // utils.BATCH_SIZE
valSteps = len(valDataLoader.dataset) // utils.BATCH_SIZE

dataiter = iter(trainDataLoader)
images, labels = dataiter.__next__()
