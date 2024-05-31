from tensor import Tensor 
import graph

a = Tensor([1,2,3])
b = Tensor([1,1,1])
c = a**2
c.backward()
graph.draw_dot(c, format="png").render(filename="img/graph")