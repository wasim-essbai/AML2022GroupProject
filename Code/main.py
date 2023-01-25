import matplotlib
import torchvision
from torchvision.transforms import ToTensor
import numpy as np

matplotlib.use("Agg")
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
TRAIN_SPLIT = 0.70
TEST_SPLIT = (1 - TRAIN_SPLIT) * 0.5
VAL_SPLIT = (1 - TRAIN_SPLIT) * 0.5

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] loading the full dataset...")
full_function_dataset = FunctionsDataset(csv_file='./function_generation/generated_dataset/function_plot_labels.csv',
                                         root_dir='./function_generation/generated_dataset/data/')
print("Dataset length: ", len(full_function_dataset))

# calculate the train/validation split
print("[INFO] generating the train/validation split...")
numTrainSamples = int(len(full_function_dataset) * TRAIN_SPLIT)
numTestSamples = int(len(full_function_dataset) * TEST_SPLIT)
numValSamples = int(len(full_function_dataset) * VAL_SPLIT)
(trainData, testData, valData) = random_split(full_function_dataset,
                                              [numTrainSamples, numTestSamples, numValSamples],
                                              generator=torch.Generator().manual_seed(42))

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True,
                             batch_size=utils.BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=utils.BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=utils.BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // utils.BATCH_SIZE
testSteps = len(testDataLoader.dataset) // utils.BATCH_SIZE
valSteps = len(valDataLoader.dataset) // utils.BATCH_SIZE

# initialize the FrCNet model
print("[INFO] initializing the FrCNet model...")
model = FrCNet(
    numChannels=4,
    classes=15).to(device)

# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.MSELoss()

# initialize a dictionary to store training history
H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()

# loop over our epochs
for e in range(0, EPOCHS):
    print("Start of epoch: ", e)
    # set the model in training mode
    model.train()

    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0

    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    valCorrect = 0

    # loop over the training set
    for sample in trainDataLoader:
        x = sample['image']
        y = sample['labels']

        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        x = x.float()
        y = y.float()
        y = y.squeeze(1)

        # From: [batch_size, height, width, channels]
        # To: [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2)

        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFn(pred, y)

        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()

        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        pred = torch.round(pred)
        totalTrainLoss += loss
        for j in range(len(y)):
            trainCorrect += 1 if (pred[j] == y[j]).sum().item() == 15 else 0

# switch off autograd for evaluation
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()

    # loop over the validation set
    for sample in valDataLoader:
        x = sample['image']
        y = sample['labels']

        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        x = x.float()
        y = y.float()
        y = y.squeeze(1)

        x = x.permute(0, 3, 1, 2)

        # make the predictions and calculate the validation loss
        pred = model(x)
        totalValLoss += lossFn(pred, y)
        # calculate the number of correct predictions
        for j in range(len(y)):
            valCorrect += 1 if (pred[j] == y[j]).sum().item() == 15 else 0

    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    # calculate the training and validation accuracy
    trainCorrect = trainCorrect / len(trainDataLoader.dataset)
    valCorrect = valCorrect / len(valDataLoader.dataset)

    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)

    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
        avgValLoss, valCorrect))

    # finish measuring how long training took
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))
