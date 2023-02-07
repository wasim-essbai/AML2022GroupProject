from frC_net import FrCNet
from torch.autograd.grad_mode import F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch import nn
from torch.utils.data import Subset
from function_generation.functions_dataset import FunctionsDataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import utils
import matplotlib
matplotlib.use("Agg")

import os
os.chdir('/content/drive/MyDrive/AML2022GroupProject/Code')
# dataset choice
full = True

# define training hyperparameters
INIT_LR = 0.001
EPOCHS = 10

# define the train and val splits
TRAIN_SPLIT = 0.70
VAL_SPLIT = (1 - TRAIN_SPLIT) * 0.5
TEST_SPLIT = (1 - TRAIN_SPLIT) * 0.5

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] loading the full dataset...")
if full:
  output_dim_file = open('./function_generation/generated_dataset/output_dim.txt', 'r')
  output_dimension = int(output_dim_file.read())
  full_function_dataset = FunctionsDataset(csv_file='./function_generation/generated_dataset/function_plot_labels.csv',
                                         root_dir='./function_generation/generated_dataset/data/',
                                         label_dim = output_dimension)
else:
  output_dim_file = open('./function_generation/reduced_generated_dataset/output_dim.txt', 'r')
  output_dimension = int(output_dim_file.read())
  full_function_dataset = FunctionsDataset(csv_file='./function_generation/reduced_generated_dataset/function_plot_labels.csv',
                                         root_dir='./function_generation/reduced_generated_dataset/data/',
                                         label_dim = output_dimension)

print("Dataset length: ", len(full_function_dataset))

N = len(full_function_dataset)

# generate & shuffle indices
indices = np.arange(N)
indices = np.random.permutation(indices)
# there are many ways to do the above two operation. (Example, using np.random.choice can be used here too

# select train/test/val, for demo I am using 70,15,15
train_indices = indices [:int(N * TRAIN_SPLIT)]
val_indices = indices[int(N * TRAIN_SPLIT):int(N * (TRAIN_SPLIT + VAL_SPLIT))]
test_indices = indices[int(N * (1 - TEST_SPLIT)):]

# calculate the train/validation split
print("[INFO] generating the train/validation split...")
numTrainSamples = int(len(full_function_dataset) * TRAIN_SPLIT)
numTestSamples = int(len(full_function_dataset) * TEST_SPLIT)
numValSamples = int(len(full_function_dataset) * VAL_SPLIT)
(trainData, testData, valData) = random_split(full_function_dataset,
                                              [numTrainSamples, numTestSamples, numValSamples],
                                              generator=torch.Generator().manual_seed(42))

trainData = Subset(full_function_dataset, train_indices)
valData = Subset(full_function_dataset, val_indices)
testData = Subset(full_function_dataset, test_indices)

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
    numChannels=1,
    output_size=output_dimension).to(device)

# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.L1Loss()

# initialize a dictionary to store training history
H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}
train_cost_history = torch.ones(EPOCHS, len(trainDataLoader))
validation_cost_history = torch.ones(EPOCHS, len(valDataLoader))

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()

label_scaling = 1000000
label_shift = 0

print("Start training:")
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

    trainLevel = 0
    trainLenght = len(trainDataLoader)

    train_epoch_cost_history = torch.ones(trainLenght)
    # loop over the training set
    for sample in trainDataLoader:
        x = sample['image']
        y = sample['labels']

        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        x = x.float()
        y = y.float()
        y = y.squeeze(1)
        y_ex = (y + label_shift) * label_scaling

        # From: [batch_size, height, width, channels]
        # To: [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2)

        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFn(pred, y_ex)
        train_epoch_cost_history[trainLevel] = loss.item()
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        pred = torch.round((pred / label_scaling) - label_shift)
        
        totalTrainLoss += loss
        for j in range(len(y)):
            trainCorrect += 1 if utils.target_close(pred[j], y[j]) else 0
        trainLevel += 1
        print('\r' + ' Training: ' + str(round(100 * trainLevel/trainLenght, 2)) + '% complete..', end ="")
    train_cost_history[e] = train_epoch_cost_history

    # switch off autograd for evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        validationLevel = 0
        validationLenght = len(valDataLoader)

        validation_epoch_cost_history = torch.ones(validationLenght)
        print("")
        # loop over the validation set
        for sample in valDataLoader:
            x = sample['image']
            y = sample['labels']

            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            x = x.float()
            y = y.float()
            y = y.squeeze(1)
            y_ex = (y + label_shift) * label_scaling

            x = x.permute(0, 3, 1, 2)

            # make the predictions and calculate the validation loss
            pred = model(x)
            loss = lossFn(pred, y_ex)
            validation_epoch_cost_history[validationLevel] = loss.item()
            totalValLoss += loss
            
            pred = torch.round((pred / label_scaling) - label_shift)
            
            # calculate the number of correct predictions
            for j in range(len(y)):
                valCorrect += 1 if utils.target_close(pred[j], y[j]) else 0
            validationLevel += 1
            print('\r' + ' Validation: ' + str(round(100 * validationLevel/validationLenght, 2)) + '% complete..', end ="")
        validation_cost_history[e] = validation_epoch_cost_history
            
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
        print("")
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avgValLoss, valCorrect))

# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
    endTime - startTime))

# we can now evaluate the network on the test set
print("[INFO] evaluating network...")
# turn off autograd for testing evaluation
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()

    # initialize a list to store our predictions
    test_preds = torch.ones(len(testDataLoader) * utils.BATCH_SIZE, output_dimension)
    test_targets = torch.ones(len(testDataLoader) * utils.BATCH_SIZE, output_dimension)

    testLevel = 0
    # loop over the test set
    for sample in testDataLoader:
        x = sample['image']
        y = sample['labels']

        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        x = x.float()
        y = y.float()
        y = y.squeeze(1)

        x = x.permute(0, 3, 1, 2)

        # make the predictions and add them to the list
        pred = model(x)

        pred = torch.round((pred / label_scaling) - label_shift)
        for i in range(len(pred)):
            test_preds[testLevel + i] = pred[i]
            test_targets[testLevel + i] = y[i]
        testLevel = testLevel + 1


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="Train Loss")
plt.plot(H["val_loss"], label="Val Loss")
plt.plot(H["train_acc"], label="Train Acc")
plt.plot(H["val_acc"], label="Val Acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig("./output.png")

print("Model ready!")

torch.save(model.state_dict(), './model/model.pth')
torch.save(train_cost_history, 'train_cost_history.pt')
torch.save(validation_cost_history, 'validation_cost_history.pt')
torch.save(test_preds, 'test_preds.pt')
torch.save(test_targets, 'test_targets.pt')