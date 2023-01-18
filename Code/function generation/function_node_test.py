import numpy as np
from function_node import FunctionNode


# Testing function node

def test_function(x):
    a = np.square(x)
    b = np.cos(a)
    c = np.multiply(a, b)
    d = np.exp(c)
    e = np.sin(c)
    f = d + e
    return f


x_test = np.linspace(-10, 20.0, num=500)

f1 = FunctionNode(None, None, None, None, True)
f2 = FunctionNode(np.square, False, f1, None, False)
f3 = FunctionNode(np.cos, False, f2, None, False)
f4 = FunctionNode(np.multiply, True, f3, f2, False)
f5 = FunctionNode(np.exp, False, f4, None, False)
f6 = FunctionNode(np.sin, False, f4, None, False)
f7 = FunctionNode(np.add, True, f5, f6, False)

assert (sum(f7.compute_value(x_test) - test_function(x_test)) == 0)
