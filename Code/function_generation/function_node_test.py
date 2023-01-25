import numpy as np
from function_node import FunctionNode


def identity_function(x):
    return x

def set_leaf_node():
    leaf_node = FunctionNode(None, None, None, None, None, False)
    leaf_node.is_leaf = True
    leaf_node.f = identity_function
    leaf_node.f_code = 7
    leaf_node.is_binary = False
    return leaf_node

# Testing function node

def test_function(x):
    a = np.square(x)
    b = np.sin(a)
    c = np.multiply(a, b)
    d = np.exp(c)
    e = np.sin(c)
    f = d + e
    return f


x_test = np.linspace(-10, 20.0, num=500)

f1 = FunctionNode(identity_function, 7, None, None, None, True)
f2 = FunctionNode(np.square, 10, False, f1, None, False)
f3 = FunctionNode(np.sin, 9, False, f2, None, False)
f4 = FunctionNode(np.multiply, 3, True, f2, f3, False)
f5 = FunctionNode(np.exp, 8, False, f4, None, False)
f6 = FunctionNode(np.sin, 9, False, f4, None, False)
f7 = FunctionNode(np.add, 0, True, f5, f6, False)

assert (sum(f7.compute_value(x_test) - test_function(x_test)) == 0)

print(f4.get_encode())

f8 = set_leaf_node()
f9 = set_leaf_node()
f10 = FunctionNode(np.divide, 2, True, f8, f9, False)
print(f10.print_function())
print(f8.__eq__(f9))
