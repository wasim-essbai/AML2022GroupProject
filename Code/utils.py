import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader

# define training hyperparameters
BATCH_SIZE = 100
# define the train and val splits
TRAIN_SPLIT = 0.70
TEST_SPLIT = 0.15
VAL_SPLIT = 1 - TRAIN_SPLIT - TEST_SPLIT


def get_data():
    data_dir = 'function_generation/generated_dataset'

    transform = transforms.Compose([
        #transforms.RandomRotation(20),
        #transforms.RandomResizedCrop(128),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    full_dataset = datasets.ImageFolder(data_dir, transform=transform)

    full_data = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return full_data


def get_data_split(full_dataset):
    # calculate the train/validation split
    print("[INFO] generating the train/test/validation split...")
    print("Total samples number: ", len(full_dataset))
    numTrainSamples = int(len(full_dataset) * TRAIN_SPLIT)
    numTestSamples = int(len(full_dataset) * TEST_SPLIT)
    numValSamples = int(len(full_dataset) * VAL_SPLIT)
    (trainData, testData, valData) = random_split(full_dataset,
                                                  [numTrainSamples, numTestSamples, numValSamples],
                                                  generator=torch.Generator().manual_seed(42))

    return trainData, testData, valData


def target_close(pred, y, shift = 0):
    close = True
    for i in range(len(y) - shift):
        close = close and pred[i + shift] == y[i + shift]
    return close
