import svglib 
import enum 

def trace(root, nodes=None, edges=None):
    # only for root node
    if nodes is None and edges is None:
        nodes, edges = set(), set()
    if root not in nodes:
        nodes.add(root)
        for child in root._parents:
            edges.add((child, root))
            trace(child, nodes, edges)
    # only for root node
    return nodes, edges

class GraphDirection(enum.Enum):
    horizontal = "LR"
    vertical = "TB"

def draw_dot(root, format='svg', direction=GraphDirection.horizontal):
    nodes, edges = trace(root)
    dot = svglib.Digraph(format=format, graph_attr={'rankdir': direction}) #, node_attr={'rankdir': 'TB'})
    for n in nodes:
        dot.node(name=str(n._pointer), label = "{ data  | grad %.4f }" % (n, n.grad), shape='record')
        if n.name:
            dot.node(name=str(n._pointer) + n.name, label=n.name)
            dot.edge(str(n._pointer) + n.name, str(n._pointer))
    for n1, n2 in edges:
        dot.edge(str(n1._pointer), str(n2._pointer) + n2.name)
    return dot

