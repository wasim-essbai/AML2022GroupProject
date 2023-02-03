import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from function_node import FunctionNode

np.seterr(all='raise')

LABEL_DIM = 0


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
basic_functions_encoding = {
    'identity_function': 7,
    'exp': 8,
    'sin': 9,
    'square': 10,
    'tan': 11,
    'log': 12,
    'absolute': 13
}

function_types = ['constant', 'variable']
function_types_encoding = {
    'constant': 5,
    'variable': 6
}

operators = [np.add, np.subtract, np.divide, np.multiply, np.power]
operators_encoding = {
    'add': 0,
    'subtract': 1,
    'true_divide': 2,
    'multiply': 3,
    'power': 4
}
function_compositions = ['unary', 'binary']


def get_function_type():
    n = np.random.choice(np.arange(0, 2), p=[0.0001, 0.9999])
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
    plt.figure(figsize=(2.5, 1))
    x = np.linspace(-8, 8, num=400)
    y = f(x)
    plt.plot(y)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('./mini_dataset/data/' + figure_name + '.png')
    plt.close()


def add_label(figure_name, label_code):
    global LABEL_DIM
    if len(label_code) > LABEL_DIM:
        LABEL_DIM = len(label_code)

    label = [figure_name + '.png'] + label_code
    labels.append(label)


def add_padding_label(figure_name, label_code):
    identity_f_code = basic_functions_encoding[identity_function.__name__]
    while len(label_code) != LABEL_DIM:
        label_code = [6, identity_f_code] + label_code
    label_code = label_code[::-1]
    label = [figure_name] + label_code
    padded_labels.append(label)
    return label


def is_in_list(a, list_to_check):
    for elem in list_to_check:
        if a == elem:
            return True
    return False


def set_constant_function():
    constant_node = FunctionNode(None, None, None, None, None, False)
    constant_node.f = ConstantFunction(None).constant
    constant_node.f_code = function_types_encoding['constant']
    constant_node.is_binary = False
    constant_node.child1 = set_leaf_node()
    return constant_node


def set_leaf_node():
    leaf_node = FunctionNode(None, None, None, None, None, False)
    leaf_node.is_leaf = True
    leaf_node.f = identity_function
    leaf_node.f_code = basic_functions_encoding[identity_function.__name__]
    leaf_node.is_binary = False
    return leaf_node


def generate_function(composition, level, only_basic):
    function_node = FunctionNode(None, None, None, None, None, False)
    if level <= 0:
        function_node = set_leaf_node()
        return function_node

    if composition is None:
        composition = get_composition()
    if composition == 'binary' and not only_basic:
        operator = get_operator()
        function_node.f = operator
        function_node.f_code = operators_encoding[operator.__name__]
        function_node.is_binary = True

        function_node.child1 = generate_function('unary', level - 1, only_basic)
        function_node.child2 = generate_function('unary', level - 1, only_basic)

        if (function_node.child1.__eq__(function_node.child2)
                and (operator.__name__ == np.subtract.__name__ or operator.__name__ == np.divide.__name__)):
            function_node = set_constant_function()

        if (function_node.child1.__eq__(function_node.child2)
                and (operator.__name__ == np.add.__name__)):
            function_node = function_node.child1

        if (function_node.child1.__eq__(function_node.child2)
                and (operator.__name__ == np.multiply.__name__)):
            function_node.f = np.square
            function_node.f_code = basic_functions_encoding[np.square.__name__]
            function_node.is_binary = False
            function_node.child2 = None

        if function_node.child1.f.__name__ == 'constant' and function_node.child2.f.__name__ != 'constant':
            function_node = function_node.child2
        if function_node.child1.f.__name__ != 'constant' and function_node.child2.f.__name__ == 'constant':
            function_node = function_node.child1

    if composition == 'unary':
        function_type = get_function_type()

        if function_type == 'constant':
            function_node = set_constant_function()
            return function_node
        if function_type == 'variable':
            function_node.is_binary = False
            func = get_function()
            function_node.f = func
            function_node.f_code = basic_functions_encoding[func.__name__]
            function_node.child1 = generate_function(None, level - 1, only_basic)
            if function_node.child1.f.__name__ == 'constant':
                function_node = set_constant_function()

    return function_node


# Do generation with different depth level.
# 10% basic functions (scaled) a*exp level = 1
# 30% combined functions  level = 1
# 50% more general level = 2

line = np.linspace(-8, 8, num=400)

dataset_size = 20
generated_functions = []
labels = []
padded_labels = []
num_constants = 0
i = 0
while i < int(0.1 * dataset_size):
    try:
        gen_f = generate_function(None, 1, True)
        if num_constants > int(0.06 * dataset_size) and gen_f.f.__name__ == 'constant':
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
        add_label(name, gen_f.get_encode())
        i = i + 1

print(str(len(generated_functions)) + ' functions generated')

i = 0
num_constants = 0
while i < int(0.3 * dataset_size):
    try:
        gen_f = generate_function(None, 2, False)
        if num_constants > int(0.06 * dataset_size) and gen_f.f.__name__ == 'constant':
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
        add_label(name, gen_f.get_encode())
        i = i + 1

print(str(len(generated_functions)) + ' functions generated')

i = 0
num_constants = 0
while i < int(0.6 * dataset_size):
    try:
        gen_f = generate_function(None, 3, False)
        if num_constants > int(0.06 * dataset_size) and gen_f.f.__name__ == 'constant':
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
        add_label(name, gen_f.get_encode())
        i = i + 1

print(str(len(generated_functions)) + ' functions generated')

header = ['image_name']
for i in range(LABEL_DIM):
    header.append('f' + str(i))

padded_labels.append(header)
for j in range(len(labels)):
    elem = labels[j]
    elem = add_padding_label(elem[0], elem[1:])
    if len(elem) != LABEL_DIM + 1:
        print('wrong')

# saving label dimension
file = open('./mini_dataset/output_dim.txt', 'w')
file.write(str(LABEL_DIM))
file.close()

with open('./mini_dataset/function_plot_labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(padded_labels)
