class FunctionNode:
    def __init__(self, fun, f_code, is_binary, child1, child2, is_leaf):
        self.child1 = child1
        self.child2 = child2
        self.f = fun
        self.f_code = f_code
        self.is_binary = is_binary
        self.is_leaf = is_leaf

    def compute_value(self, value):
        if self.is_leaf:
            return value

        if self.is_binary:
            a = self.child1.compute_value(value)
            b = self.child2.compute_value(value)
            try:
                x = self.f(a, b)
            except ZeroDivisionError:
                x = None
        else:
            a = self.child1.compute_value(value)
            try:
                x = self.f(a)
            except ZeroDivisionError:
                x = None

        return x

    def print_function(self):
        name = 'x'
        if self.is_leaf:
            return name

        if self.is_binary:
            name = '(' + self.child1.print_function() + ' ' + self.f.__name__ + ' ' + self.child2.print_function() + ')'
        else:
            name = self.f.__name__ + '(' + self.child1.print_function() + ')'

        return name

    def get_encode(self):
        label = []
        if self.is_leaf:
            label.append(self.f_code)
            return label
        if self.is_binary:
            label.append(self.f_code)
            label = label + self.child1.get_encode()
            label = label + self.child2.get_encode()
        else:
            if self.child1.is_leaf:
                label.append(self.f_code)
            else:
                label.append(6)
                label.append(self.f_code)
                label = label + self.child1.get_encode()
        return label

    def __eq__(self, other):
        """Overrides the default implementation"""
        if not isinstance(other, FunctionNode):
            return False

        equal = True
        equal = equal and self.f is not None and other.f is not None and self.f.__name__ == other.f.__name__
        equal = equal and self.is_binary == other.is_binary
        equal = equal and self.is_leaf == other.is_leaf
        equal = equal and self.child1.__eq__(other.child1)
        equal = equal and self.child2.__eq__(other.child2)

        return equal
