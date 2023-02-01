from torch.nn import Module, Flatten, Sequential
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU


class FrCNet(Module):
    def __init__(self, numChannels, output_size):
        super(FrCNet, self).__init__()
        self.network = Sequential(

            Conv2d(numChannels, 32, kernel_size=(3, 5), padding=1),
            ReLU(),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(2, 2),

            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(2, 2),

            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(2, 2),

            Flatten(),
            Linear(95232, 2048),
            ReLU(),
            Linear(2048, 512),
            ReLU(),
            Linear(512, output_size)
        )

    def forward(self, x):
        return self.network(x)
