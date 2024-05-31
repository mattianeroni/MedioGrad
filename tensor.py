import numpy as np
from functools import partial

import graph 


class ConversionException(Exception):
    pass


def tensor(x):
    if isinstance(x, Tensor): return x 
    elif isinstance(x, (tuple, list, int, float)): return Tensor(x)
    else: raise ConversionException(f"Cannot convert {type(x)} to Tensor")


class Add:

    _label = "+"
    
    @staticmethod
    def forward(t1, t2): 
        out = Tensor(t1.data + t2.data)
        out._parents = (t1, t2)
        out._backward = partial(Add.backward, t1, t2, out)
        return out
    
    @staticmethod
    def backward(t1, t2, out):
        t1.grad += out.grad 
        t2.grad += out.grad 


class Mul:

    _label = "*"
    
    @staticmethod
    def forward(t1, t2): 
        out = Tensor(t1.data * t2.data)
        out._parents = (t1, t2)
        out._backward = partial(Mul.backward, t1, t2, out)
        return out
    
    @staticmethod
    def backward(t1, t2, out):
        t1.grad += t2.data * out.grad
        t2.grad += t1.data * out.grad
        
class Pow:

    _label = "**"

    @staticmethod
    def forward(tensor, scalar):
        assert isinstance(scalar, (int, float)), "Only power to scalar is supported"
        out = Tensor(tensor.data**scalar)
        out._parents = (tensor, )
        out._backward = partial(Pow.backward, tensor, scalar, out)
        return out

    @staticmethod
    def backward(tensor, scalar, out):
        tensor.grad += (scalar * tensor.data**(scalar - 1)) * out.grad


class ReLU:

    _label = "ReLU"

    @staticmethod
    def forward(tensor):
        out = Tensor(np.where(tensor.data < 0.0, 0.0, tensor.data))
        out._parents = (tensor, )
        out._backward = partial(ReLU.backward, tensor, out)
        return out 

    @staticmethod
    def backward(tensor, out):
        tensor.grad += np.where(out.data > 0.0, 1, 0) * out.grad


class Tensor:

    def __init__(self, data, name=""):
        if isinstance(data, (tuple, list, np.ndarray)):
            self.data = np.asarray(data).astype(np.float32)
            self.grad = np.zeros(self.data.shape, dtype=np.float32)
        elif isinstance(data, Tensor):
            self.data = np.asarray(data.data).astype(np.float32)
            self.grad = np.zeros(self.data.shape, dtype=np.float32)
        elif isinstance(data, (int, float)):
            self.data = data
            self.grad = 0

        self.name = name
        self._parents = None
        self._backward = lambda: None 
        self._pointer = id(self)

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
        return self.__mul__(-1)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        return other.__add__(self.__neg__())

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(other.__pow__(-1))

    def __rtruediv__(self, other):
        return other.__mul__(self.__pow__(-1))
    
    def backward(self):
        # topological order all of the children in the graph
        topo_graph = graph.topological_graph(self)
        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones(self.data.shape)
        for v in topo_graph: v._backward()
