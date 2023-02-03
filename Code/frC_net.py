from torch.nn import Module, Flatten, Sequential
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import AvgPool2d
from torch.nn import ReLU


class FrCNet(Module):
    def __init__(self, numChannels, output_size):
        super(FrCNet, self).__init__()
        self.network = Sequential(

            Conv2d(numChannels, 16, kernel_size=2, padding=1),
            ReLU(),
            MaxPool2d(2, 2),

            Conv2d(16, 32, kernel_size=2, padding=1),
            ReLU(),
            MaxPool2d(2, 2),

            Conv2d(32, 64, kernel_size=2, padding=0),
            ReLU(),
            MaxPool2d(2, 2),

            Flatten(),
            Linear(23808, 1012),
            ReLU(),
            Linear(1012, 512),
            ReLU(),
            Linear(512, output_size),
        )

    def forward(self, x):
        return self.network(x)
