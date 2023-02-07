from torch.utils.data import DataLoader
from frC_net import FrCNet
from function_generation.functions_dataset import FunctionsDataset
import torch
import utils
import pandas

import os
os.chdir('/content/drive/MyDrive/AML2022GroupProject/Code')

full = False

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

N = len(full_function_dataset)
print("Dataset length: ", N)

dataLoader = DataLoader(full_function_dataset, batch_size=utils.BATCH_SIZE)

# initialize the FrCNet model
print("[INFO] initializing the FrCNet model...")
model = FrCNet(
    numChannels=1,
    output_size=output_dimension).to(device)
model.load_state_dict(torch.load('./model/model.pth'))

label_scaling = 1000000
label_shift = 0

model.eval()

level = 0
datasetLength = len(dataLoader)

testCorrect = 0
testCorrectShift2 = 0
testCorrectShift4 = 0

print("[INFO] Start evaluation")
with torch.no_grad():
  for sample in dataLoader:
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

      for j in range(len(pred)):
          testCorrect += 1 if utils.target_close(pred[j], y[j]) else 0
          testCorrectShift2 += 1 if utils.target_close(pred[j], y[j], 2) else 0
          testCorrectShift4 += 1 if utils.target_close(pred[j], y[j], 4) else 0
      level += 1
      print('\r' + ' Evaluation: ' + str(round(100 * level/datasetLength, 2)) + '% complete..', end ="")

headers=["Shift", "Accurancy"]
data = [[0, str(round(100 * testCorrect/N, 2)) + '%'],
[2, str(round(100 * testCorrectShift2/N, 2)) + '%'],
[4, str(round(100 * testCorrectShift4/N, 2)) + '%']]
print(pandas.DataFrame(data, columns = headers))