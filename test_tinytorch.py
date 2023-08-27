import torch
import tinytorch
import numpy as np

np.random.seed(69420)


def test_add():
    x = np.random.rand(3, 3)
    y = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    y_t = torch.tensor(y, requires_grad=True)

    x_tt = tinytorch.tensor(x, requires_grad=True)
    y_tt = tinytorch.tensor(y, requires_grad=True)

    z_t = x_t + y_t
    z_t.sum().backward()

    z_tt = x_tt + y_tt
    z_tt.sum().backward()

    assert np.allclose(z_t.detach().numpy(), z_tt.data), "Addition results do not match between PyTorch and tinytorch."
    assert np.allclose(x_t.grad.numpy(), x_tt.grad.data), "Gradients for x do not match between PyTorch and tinytorch."
    assert np.allclose(y_t.grad.numpy(), y_tt.grad.data), "Gradients for y do not match between PyTorch and tinytorch."



def test_mul():
    x = np.random.rand(3, 3)
    y = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    y_t = torch.tensor(y, requires_grad=True)

    x_tt = tinytorch.tensor(x, requires_grad=True)
    y_tt = tinytorch.tensor(y, requires_grad=True)

    z_t = x_t * y_t
    z_t.sum().backward()

    z_tt = x_tt * y_tt
    z_tt.sum().backward()

    assert np.allclose(z_t.detach().numpy(), z_tt.data), "Multiplication results do not match between PyTorch and tinytorch."
    assert np.allclose(x_t.grad.numpy(), x_tt.grad.data), "Gradients for x do not match between PyTorch and tinytorch."
    assert np.allclose(y_t.grad.numpy(), y_tt.grad.data), "Gradients for y do not match between PyTorch and tinytorch."


def test_div():
    x = np.random.rand(3, 3) + 1  # Adding 1 to avoid division by zero
    y = np.random.rand(3, 3) + 1

    x_t = torch.tensor(x, requires_grad=True)
    y_t = torch.tensor(y, requires_grad=True)

    x_tt = tinytorch.tensor(x, requires_grad=True)
    y_tt = tinytorch.tensor(y, requires_grad=True)

    z_t = x_t / y_t
    z_t.sum().backward()

    z_tt = x_tt / y_tt
    z_tt.sum().backward()

    assert np.allclose(z_t.detach().numpy(), z_tt.data), "Division results do not match between PyTorch and tinytorch."
    assert np.allclose(x_t.grad.numpy(), x_tt.grad.data), "Gradients for x do not match between PyTorch and tinytorch."
    assert np.allclose(y_t.grad.numpy(), y_tt.grad.data), "Gradients for y do not match between PyTorch and tinytorch."


def test_sub():
    x = np.random.rand(3, 3)
    y = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    y_t = torch.tensor(y, requires_grad=True)

    x_tt = tinytorch.tensor(x, requires_grad=True)
    y_tt = tinytorch.tensor(y, requires_grad=True)

    z_t = x_t - y_t
    z_t.sum().backward()

    z_tt = x_tt - y_tt
    z_tt.sum().backward()

    assert np.allclose(z_t.detach().numpy(), z_tt.data), "Subtraction results do not match between PyTorch and tinytorch."
    assert np.allclose(x_t.grad.numpy(), x_tt.grad.data), "Gradients for x do not match between PyTorch and tinytorch."
    assert np.allclose(y_t.grad.numpy(), y_tt.grad.data), "Gradients for y do not match between PyTorch and tinytorch."

def test_sum():
    x = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    x_tt = tinytorch.tensor(x, requires_grad=True)

    z_t = x_t.sum()
    z_t.backward()

    z_tt = x_tt.sum()
    z_tt.backward()

    assert np.allclose(z_t.detach().numpy(), z_tt.data), "Sum results do not match between PyTorch and tinytorch."
    assert np.allclose(x_t.grad.numpy(), x_tt.grad.data), "Gradients do not match between PyTorch and tinytorch."

def test_sum_axis(axis=1):
    x = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    x_tt = tinytorch.tensor(x, requires_grad=True)

    z_t = x_t.sum(dim=axis)
    z_t.sum().backward()  # Summing again to get a scalar for backward()

    z_tt = x_tt.sum(axis=axis)
    z_tt.sum().backward()  # Summing again to get a scalar for backward()

    assert np.allclose(z_t.detach().numpy(), z_tt.data), f"Sum along axis {axis} results do not match between PyTorch and tinytorch."
    assert np.allclose(x_t.grad.numpy(), x_tt.grad.data), f"Gradients along axis {axis} do not match between PyTorch and tinytorch."

def test_reshape():
    x = np.random.rand(2, 3, 4)

    x_t = torch.tensor(x, requires_grad=True)
    x_tt = tinytorch.tensor(x, requires_grad=True)

    new_shape = (3, 2, 4)
    z_t = x_t.reshape(new_shape)
    z_t.sum().backward()

    z_tt = x_tt.reshape(new_shape)
    z_tt.sum().backward()

    assert np.allclose(z_t.detach().numpy(), z_tt.data), "Reshape results do not match between PyTorch and tinytorch."
    assert np.allclose(x_t.grad.numpy(), x_tt.grad.data), "Gradients do not match between PyTorch and tinytorch."

def test_custom_eq():
    # Define custom function f(x, y) within the scope of test_custom_eq
    def f(x, y):
        return x * x + (x * y) / (x + y) + x*(x+y)
        
    x = np.random.rand(3, 3)
    y = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    y_t = torch.tensor(y, requires_grad=True)

    x_tt = tinytorch.tensor(x, requires_grad=True)
    y_tt = tinytorch.tensor(y, requires_grad=True)

    # Compute f(x, y) using PyTorch
    z_t = f(x_t, y_t)
    z_t.sum().backward()

    # Compute f(x, y) using tinytorch
    z_tt = f(x_tt, y_tt)
    z_tt.sum().backward()

    # Assertions to check if both results match
    assert np.allclose(z_t.detach().numpy(), z_tt.data), "Custom eq results do not match between PyTorch and tinytorch."
    assert np.allclose(x_t.grad.numpy(), x_tt.grad.data), "Gradients for x do not match between PyTorch and tinytorch."
    assert np.allclose(y_t.grad.numpy(), y_tt.grad.data), "Gradients for y do not match between PyTorch and tinytorch."


if __name__  == "__main__":
    test_add()