import random

import numpy as np
import matplotlib.pyplot as plt

from function_node import FunctionNode

np.seterr(all='raise')


# Self defined math functions
def identity_function(x):
    return x


class ConstantFunction:
    def __init__(self, value):
        self.value = value
        if value is None:
            self.value = round(random.uniform(-1, 1) * 5, 5)

    def constant(self, x):
        return np.ones(x.shape) * self.value


basic_functions = [identity_function, np.exp, np.sin, np.square, np.tan, np.log, np.absolute]
function_types = ['constant', 'variable']
operators = [np.add, np.subtract, np.divide, np.multiply, np.power]
function_compositions = ['unary', 'binary']


def get_function_type():
    n = np.random.choice(np.arange(0, 2), p=[0.03, 0.97])
    return function_types[n]


def get_function():
    n = np.random.randint(0, len(basic_functions))
    return basic_functions[n]


def get_operator():
    n = np.random.randint(0, len(operators))
    return operators[n]


def get_composition():
    n = np.random.randint(0, len(function_compositions))
    return function_compositions[n]


def save_plot(f, figure_name):
    plt.figure(figsize=(20, 8))
    x = np.linspace(-10, 10.0, num=500)
    y = f(x)
    plt.plot(y)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./function generation/generated_dataset/' + figure_name + '.png')
    plt.close()


def is_in_list(a, list_to_check):
    for elem in list_to_check:
        if a == elem:
            return True
    return False


def set_constant_function():
    constant_node = FunctionNode(None, None, None, None, False)
    constant_node.f = ConstantFunction(None).constant
    constant_node.is_binary = False
    constant_node.child1 = FunctionNode(None, None, None, None, False)
    constant_node.child1.is_leaf = True
    return constant_node


def generate_function(composition, level, only_basic):
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
        if only_basic:
            function_node.child1 = set_constant_function()
        else:
            function_node.child1 = generate_function('unary', level - 1, only_basic)
        function_node.child2 = generate_function('unary', level - 1, only_basic)
        if(function_node.child1.__eq__(function_node.child2)
                and (operator == 'subtract' or operator == 'true_division')):
            function_node = set_constant_function()

    if composition == 'unary':
        function_type = get_function_type();

        if function_type == 'constant':
            function_node = set_constant_function()
            return function_node
        if function_type == 'variable':
            function_node.is_binary = False
            function_node.f = get_function()
            function_node.child1 = generate_function(None, level - 1, only_basic)
            if function_node.child1.f.__name__ == 'constant':
                function_node = set_constant_function()

    return function_node


# Do generation with different depth level.
# 100 basic functions (scaled) a*exp level = 0

# 300 combined functions  level = 1

# 600 more general level = 2
line = np.linspace(-10, 10.0, num=500)

dataset_size = 4000
generated_functions = []

gen_f = generate_function('unary', 1, True)

i = 0
num_constants = 0
while i < int(0.1 * dataset_size):
    try:
        gen_f = generate_function(None, 1, True)
        if num_constants > int(0.01 * dataset_size):
            continue
        if is_in_list(gen_f.print_function, generated_functions):
            continue
        if gen_f.f.__name__ == 'constant':
            num_constants = num_constants + 1
        gen_f.compute_value(line)
    except:
        continue
    else:
        generated_functions.append(gen_f.print_function())
        name = str(len(generated_functions)) + '-' + gen_f.print_function()
        save_plot(gen_f.compute_value, name)
        i = i + 1

print(str(len(generated_functions)) + ' functions generated')
print(generated_functions)

i = 0
num_constants = 0
while i < int(0.3 * dataset_size):
    try:
        gen_f = generate_function(None, 1, False)
        if num_constants > int(0.03 * dataset_size):
            continue
        if is_in_list(gen_f.print_function, generated_functions):
            continue
        if gen_f.f.__name__ == 'constant':
            num_constants = num_constants + 1
        gen_f.compute_value(line)
    except:
        continue
    else:
        generated_functions.append(gen_f.print_function())
        name = str(len(generated_functions)) + '-' + gen_f.print_function()
        save_plot(gen_f.compute_value, name)
        i = i + 1

print(str(len(generated_functions)) + ' functions generated')
print(generated_functions)

i = 0
num_constants = 0
while i < int(0.6 * dataset_size):
    try:
        gen_f = generate_function(None, 1, False)
        if num_constants > int(0.06 * dataset_size):
            continue
        if is_in_list(gen_f.print_function, generated_functions):
            continue
        if gen_f.f.__name__ == 'constant':
            num_constants = num_constants + 1
        gen_f.compute_value(line)
    except:
        continue
    else:
        generated_functions.append(gen_f.print_function())
        name = str(len(generated_functions)) + '-' + gen_f.print_function()
        save_plot(gen_f.compute_value, name)
        i = i + 1

print(str(len(generated_functions)) + ' functions generated')
