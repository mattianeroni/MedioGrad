from tensor import Tensor 
import graph

a = Tensor([1,2,3])
b = Tensor([1,1,1])

c = a + b
print(c._parents)

graph.draw_dot(c, format="png").render(filename="graph")