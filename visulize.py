import graphviz
from tinytorch import *

G = graphviz.Digraph(format="png")
G.clear()


def visit_nodes(G: graphviz.Digraph, node: Tensor):
    uid = str(id(node))
    G.node(uid, f"Tensor: {str(node.data) } ")
    if node._ctx:
        ctx_uid = str(id(node._ctx))
        G.node(ctx_uid, f"Context: {str(node._ctx.op.__name__)}")
        G.edge(uid, ctx_uid)
        for child in node._ctx.args:
            G.edge(ctx_uid, str(id(child)))
            visit_nodes(G, child)


if __name__ == "__main__":
    x = Tensor([8])
    y = Tensor([5])
    z = x + y
    visit_nodes(G, z)
    G.render(directory="vis", view=True)
    print(z)

    print(len(G.body))
