
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
            x = self.f(a, b)
        else:
            a = self.child1.compute_value(value)
            x = self.f(a)

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