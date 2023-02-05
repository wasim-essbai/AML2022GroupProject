import torch
import matplotlib.pyplot as plt
import numpy as np

train_cost_history = torch.load('./train_cost_history.pt')
total_train_cost_history = torch.ones(len(train_cost_history))
i = 0
for epoch_train_cost_history in train_cost_history:
    plt.figure(figsize=(4, 4))
    x = np.linspace(0, len(epoch_train_cost_history) - 1, num=len(epoch_train_cost_history))
    plt.plot(x, epoch_train_cost_history)
    plt.tight_layout()
    plt.savefig('./plot_costs/train-cost-epoch-' + str(i) + '.png')
    plt.close()
    total_train_cost_history[i] = torch.sum(epoch_train_cost_history)
    i += 1

# saving train cost history per epoch
plt.figure(figsize=(4, 4))
x = np.linspace(0, len(total_train_cost_history) - 1, num=len(total_train_cost_history))
plt.plot(x, total_train_cost_history)
plt.tight_layout()
plt.savefig('./plot_costs/train-cost-.png')
plt.close()
