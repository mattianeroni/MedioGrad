import graphviz
import enum 
import collections


def topological_graph(node, visited_set=None, topo_graph=None):
    assert (visited_set is None and topo_graph is None) \
        or (visited_set is not None and topo_graph is not None) \
        , "Both 'visited_set' and 'topo_graph' must be None or not None"
    # only for root node 
    if visited_set is None or topo_graph is None:
        visited_set, topo_graph = set(), collections.deque()
    # already included node
    if node in visited_set: return
    visited_set.add(node)
    topo_graph.append(node)
    # no parents 
    if node._parents is None: return
    # recursion on parents
    for parent in node._parents:
        topological_graph(parent, visited_set=visited_set, topo_graph=topo_graph)
    # only for root 
    return topo_graph


def trace(root, nodes=None, edges=None):
    assert (nodes is None and edges is None) \
        or (nodes is not None and edges is not None) \
        , "Both 'nodes' and 'edges' must be None or not None"
    # only for root node
    if nodes is None or edges is None:
        nodes, edges = set(), set()
    if root not in nodes:
        nodes.add(root)
        if root._parents is None: return set(), set()
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
    dot = graphviz.Digraph(format=format, graph_attr={'rankdir': direction.value}) #, node_attr={'rankdir': 'TB'})
    for n in nodes:
        dot.node(name=str(n._pointer), label=f"{n} | grad({n.grad})", shape='record')
        if n.name:
            dot.node(name=str(n._pointer) + n.name, label=n.name)
            dot.edge(str(n._pointer) + n.name, str(n._pointer))
    for n1, n2 in edges:
        dot.edge(str(n1._pointer), str(n2._pointer) + n2.name)
    return dot

