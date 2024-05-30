import numpy as np
import collections

from ops import Add, Mul, ReLU, Pow


class ConversionException(Exception):
    pass


def tensor(x):
    if isinstance(x, Tensor): return x 
    elif isinstance(x, (tuple, list, int, float)): return Tensor(x)
    else: raise ConversionException(f"Cannot convert {type(x)} to Tensor")


class Tensor:

    def __init__(self, data):
        if isinstance(data, (tuple, list)):
            self.data = np.asarray(data).astype(np.float32)
            self.grad = np.zeros(self.data.shape, dtype=np.float32)
        elif isinstance(data, Tensor):
            self.data = np.asarray(data.data).astype(np.float32)
            self.grad = np.zeros(self.data.shape, dtype=np.float32)
        elif isinstance(data, (int, float)):
            self.data = data
            self.grad = 0
        
        self._parents = None
        self._backward = lambda: None 

    def __repr__(self):
        return f"tensor({str(self.data)})"
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.full(self.data.shape, other))
        return Add.forward(self, other)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.full(self.data.shape, other))
        return Mul.forward(self, other)

    def __pow__(self, other):
        return Pow.forward(self, other)

    def relu(self):
        return ReLU.forward(self)
    
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1
    
    #@staticmethod
    #def topological_graph(node, visited_set, topo_graph):
    #    if node in visited_set: return
    #    visited_set.add(node)
    #    topo_graph.append(node)
    #    for child in node._parents:
    #        Tensor.topological_graph(child)

    #def backward(self):
    #    # topological order all of the children in the graph
    #    topo_graph = collections.deque()
    #    visited_set = set()
    #    Tensor.topological_graph(self, visited_set, topo_graph)
    #    # go one variable at a time and apply the chain rule to get its gradient
    #    self.grad = np.ones(self.data.shape)
    #    for v in topo_graph: v._backward()