import numpy as np
import matplotlib.pyplot as plt
from google.api_core.path_template import get_field

from function_node import FunctionNode


# Self defined math functions
def identity_function(x):
    return x


def constant_function(x):
    return round(np.random.rand() * 20, 5)


basic_functions = [identity_function, constant_function, np.exp, np.sin, np.square]
operators = [np.add, np.subtract, np.divide, np.multiply, np.power]
function_compositions = ['unary', 'binary']


def get_function():
    n = np.random.randint(0, len(basic_functions))
    return basic_functions[n]


def get_operator():
    n = np.random.randint(0, len(operators))
    return operators[n]


def get_composition():
    n = np.random.randint(0, len(function_compositions))
    return function_compositions[n]


def print_plot(f):
    plt.figure(figsize=(20, 8))
    x = np.linspace(-10, 10.0, num=500)
    y = f(x)
    plt.plot(x, y)
    plt.show()


def generate_function(composition, level):
    function_node = FunctionNode(None, None, None, None, False)
    if level <= 0 and composition == 'unary':
        function_node.is_leaf = True
        return function_node

    if composition is None:
        composition = get_composition()
    if composition == 'binary':
        operator = get_operator()
        function_node.f = operator
        function_node.is_binary = True
        function_node.child1 = generate_function('unary', level - 1)
        function_node.child2 = generate_function('unary', level - 1)

    if composition == 'unary':
        function = get_function()
        function_node.f = function
        function_node.is_binary = False
        function_node.child1 = generate_function(None, level - 1)

    return function_node


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

assert(sum(f7.compute_value(x_test) - test_function(x_test)) == 0)

random_function = generate_function(None, 5)
print_plot(random_function.compute_value)

