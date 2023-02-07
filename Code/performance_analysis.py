import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import utils
import pandas

#os.chdir('/content/drive/MyDrive/AML2022GroupProject/Code')

train_cost_history = torch.load('./train_cost_history.pt')
total_train_cost_history = torch.ones(len(train_cost_history))
i = 0
for epoch_train_cost_history in train_cost_history:
    plt.figure(figsize=(4, 4))
    x = np.linspace(0, len(epoch_train_cost_history) - 1, num=len(epoch_train_cost_history))
    plt.plot(x, epoch_train_cost_history)
    plt.tight_layout()
    plt.savefig('./plot_costs/train/train-cost-epoch-' + str(i) + '.png')
    plt.close()
    total_train_cost_history[i] = torch.sum(epoch_train_cost_history)
    i += 1

# saving train cost history per epoch
plt.figure(figsize=(4, 4))
x = np.linspace(0, len(total_train_cost_history) - 1, num=len(total_train_cost_history))
plt.plot(x, total_train_cost_history)
plt.tight_layout()
plt.savefig('./plot_costs/train/train-cost.png')
plt.close()

validation_cost_history = torch.load('./validation_cost_history.pt')
total_validation_cost_history = torch.ones(len(validation_cost_history))
i = 0
for epoch_validation_cost_history in validation_cost_history:
    plt.figure(figsize=(4, 4))
    x = np.linspace(0, len(epoch_validation_cost_history) - 1, num=len(epoch_validation_cost_history))
    plt.plot(x, epoch_validation_cost_history)
    plt.tight_layout()
    plt.savefig('./plot_costs/validation/val-cost-epoch-' + str(i) + '.png')
    plt.close()
    total_validation_cost_history[i] = torch.sum(epoch_validation_cost_history)
    i += 1

print("End train analysis")

# saving validation cost history per epoch
plt.figure(figsize=(4, 4))
x = np.linspace(0, len(total_validation_cost_history) - 1, num=len(total_validation_cost_history))
plt.plot(x, total_validation_cost_history)
plt.tight_layout()
plt.savefig('./plot_costs/validation/val-cost.png')
plt.close()

print("End validation analysis")

# test prediction analysis
test_preds = torch.load('./test_preds.pt')
test_targets = torch.load('./test_targets.pt')

testLength = len(test_preds)
testCorrect = 0
testCorrectShift2 = 0
testCorrectShift4 = 0
for j in range(testLength):
    if utils.target_close(test_preds[j], test_targets[j], 0):
        print("Correct")
        print(test_targets[j])
        print(test_preds[j])
    testCorrect += 1 if utils.target_close(test_preds[j], test_targets[j], 0) else 0
    testCorrectShift2 += 1 if utils.target_close(test_preds[j], test_targets[j], 2) else 0
    testCorrectShift4 += 1 if utils.target_close(test_preds[j], test_targets[j], 4) else 0

headers = ["Shift", "Accurancy"]
data = [[0, str(round(100 * testCorrect / testLength, 2)) + '%'],
        [2, str(round(100 * testCorrectShift2 / testLength, 2)) + '%'],
        [4, str(round(100 * testCorrectShift4 / testLength, 2)) + '%']]
print(pandas.DataFrame(data, columns=headers))

print("End analysis!")
