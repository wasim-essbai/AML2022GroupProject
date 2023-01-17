import numpy as np
import matplotlib.pyplot as plt


def print_plot(f):
    plt.figure(figsize=(20, 8))
    x = np.linspace(-10, 30.0, num=500)
    y = f(x)
    plt.plot(x, y)
    plt.show()


print_plot(np.exp)
