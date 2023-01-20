class FunctionNode:
    def __init__(self, fun, is_binary, child1, child2, is_leaf):
        self.child1 = child1
        self.child2 = child2
        self.f = fun
        self.is_binary = is_binary
        self.is_leaf = is_leaf

    def compute_value(self, value):
        if self.is_leaf:
            return value

        x = None
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

    def __eq__(self, other):
        """Overrides the default implementation"""
        if not isinstance(other, FunctionNode):
            return False

        equal = True
        equal = equal and self.f is not None and other.f is not None and self.f.__name__ == other.f.__name__
        equal = equal and other.child1 is not None and self.child1 is None and self.child1.__eq__(other.child1)
        equal = equal and other.child2 is not None and self.child2 is None and self.child2.__eq__(other.child2)
        equal = equal and self.is_binary == other.is_binary
        equal = equal and self.is_leaf == other.is_leaf
        return equal
