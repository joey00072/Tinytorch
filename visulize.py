import graphviz

# import matplotlib.pyplot as plt
from tinytorch import *

G = graphviz.Digraph(format="png")
G.clear()


def visit_nodes(G: graphviz.Digraph, node: Tensor):
    if not isinstance(node, Tensor):
        return
    uid = str(id(node))
    node_name = f"Tensor: ({str(node.data.shape) }) "
    print(type(node))
    G.node(
        uid,
        node_name,
    )
    if node._ctx:
        ctx_uid = str(id(node._ctx))
        G.node(ctx_uid, f"Context: {str(node._ctx.op.__name__)}")
        G.edge(uid, ctx_uid)
        for child in node._ctx.args:
            G.edge(ctx_uid, str(id(child)))
            visit_nodes(G, child)


def f(x):
    return x * x * x + x


# Defining the function to plot the given function and its derivative using the custom Tensor class
def plot_function_and_derivative():
    # Values for x ranging from -3 to 3
    x_values_custom = np.linspace(-3, 3, 100)
    y_values_custom = []
    derivative_values_custom = []

    # Using the custom Tensor class to calculate the function and its derivative for each x value
    for x_val in x_values_custom:
        x_tensor = Tensor([x_val])
        y_tensor = f(x_tensor)
        y_tensor.backward()
        y_values_custom.append(y_tensor.data[0])
        derivative_values_custom.append(x_tensor.grad.data[0])

    # Plotting the original function and its derivative using the custom implementation
    plt.plot(x_values_custom, y_values_custom, label="f(x) = x^3 + x (custom)")
    plt.plot(
        x_values_custom, derivative_values_custom, label="f'(x) = 3x^2 + 1 (custom)"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Plot of the Function and its Derivative (Custom Implementation)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_function_and_derivative()

    x = Tensor([1.2])
    z = f(x)
    z.backward()
    visit_nodes(G, z)
    G.render(directory="vis", view=True)
    print(f"Z:{x} grad:{x.grad}")
