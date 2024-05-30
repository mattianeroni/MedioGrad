from tensor import Tensor

import numpy as np 
from functools import partial


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