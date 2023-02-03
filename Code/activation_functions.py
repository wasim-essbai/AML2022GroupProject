from torch.nn import Module
from torch import sigmoid
from torch import relu
from torch import ones


def scaled_sigmoid(x, alpha):
    return alpha * sigmoid(x)


def limited_relu(x, alpha):
    x = relu(x)

    for j in range(len(x)):
        for i in range(len(x[j])):
            if x[j][i] > alpha:
                x[j][i] = alpha
    return x


class ScSigmoid(Module):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return scaled_sigmoid(x, self.alpha)


class LimitedRelu(Module):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return limited_relu(x, self.alpha)
