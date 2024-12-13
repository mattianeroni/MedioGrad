# MedioGrad (WIP)
A ML library in the middle between [Pytorch](https://github.com/pytorch/pytorch) and [Micrograd](https://github.com/karpathy/micrograd).

Not evoluted as [TinyGrad](https://github.com/tinygrad/tinygrad) since objective is to be [numpy](https://numpy.org/)-based only (maybe in future with a bit of [pycuda](https://documen.tician.de/pycuda/)).


### Stupid example

```python
from tensor import Tensor 
import graph

a = Tensor([1,2,3])
b = Tensor([1,1,1])
c = a + b
c.backward()
graph.draw_dot(c, format="png").render(filename="graph")
```

![alt text](https://github.com/mattianeroni/mediograd/blob/main/img/graph.png)
