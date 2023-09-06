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

    assert np.allclose(
        z_t.detach().numpy(), z_tt.data
    ), "Addition results do not match between PyTorch and tinytorch."
    assert np.allclose(
        x_t.grad.numpy(), x_tt.grad.data
    ), "Gradients for x do not match between PyTorch and tinytorch."
    assert np.allclose(
        y_t.grad.numpy(), y_tt.grad.data
    ), "Gradients for y do not match between PyTorch and tinytorch."


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

    assert np.allclose(
        z_t.detach().numpy(), z_tt.data
    ), "Multiplication results do not match between PyTorch and tinytorch."
    assert np.allclose(
        x_t.grad.numpy(), x_tt.grad.data
    ), "Gradients for x do not match between PyTorch and tinytorch."
    assert np.allclose(
        y_t.grad.numpy(), y_tt.grad.data
    ), "Gradients for y do not match between PyTorch and tinytorch."


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

    assert np.allclose(
        z_t.detach().numpy(), z_tt.data
    ), "Division results do not match between PyTorch and tinytorch."
    assert np.allclose(
        x_t.grad.numpy(), x_tt.grad.data
    ), "Gradients for x do not match between PyTorch and tinytorch."
    assert np.allclose(
        y_t.grad.numpy(), y_tt.grad.data
    ), "Gradients for y do not match between PyTorch and tinytorch."


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

    assert np.allclose(
        z_t.detach().numpy(), z_tt.data
    ), "Subtraction results do not match between PyTorch and tinytorch."
    assert np.allclose(
        x_t.grad.numpy(), x_tt.grad.data
    ), "Gradients for x do not match between PyTorch and tinytorch."
    assert np.allclose(
        y_t.grad.numpy(), y_tt.grad.data
    ), "Gradients for y do not match between PyTorch and tinytorch."


def test_sum():
    x = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    x_tt = tinytorch.tensor(x, requires_grad=True)

    z_t = x_t.sum()
    z_t.backward()

    z_tt = x_tt.sum()
    z_tt.backward()

    assert np.allclose(
        z_t.detach().numpy(), z_tt.data
    ), "Sum results do not match between PyTorch and tinytorch."
    assert np.allclose(
        x_t.grad.numpy(), x_tt.grad.data
    ), "Gradients do not match between PyTorch and tinytorch."


def test_sum_axis(axis=1):
    x = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    x_tt = tinytorch.tensor(x, requires_grad=True)

    z_t = x_t.sum(dim=axis)
    z_t.sum().backward()  # Summing again to get a scalar for backward()

    z_tt = x_tt.sum(axis=axis)
    z_tt.sum().backward()  # Summing again to get a scalar for backward()

    assert np.allclose(
        z_t.detach().numpy(), z_tt.data
    ), f"Sum along axis {axis} results do not match between PyTorch and tinytorch."
    assert np.allclose(
        x_t.grad.numpy(), x_tt.grad.data
    ), f"Gradients along axis {axis} do not match between PyTorch and tinytorch."


def test_reshape():
    x = np.random.rand(2, 3, 4)

    x_t = torch.tensor(x, requires_grad=True)
    x_tt = tinytorch.tensor(x, requires_grad=True)

    new_shape = (3, 2, 4)
    z_t = x_t.reshape(new_shape)
    z_t.sum().backward()

    z_tt = x_tt.reshape(new_shape)
    z_tt.sum().backward()

    assert np.allclose(
        z_t.detach().numpy(), z_tt.data
    ), "Reshape results do not match between PyTorch and tinytorch."
    assert np.allclose(
        x_t.grad.numpy(), x_tt.grad.data
    ), "Gradients do not match between PyTorch and tinytorch."


def test_custom_eq():
    # Define custom function f(x, y) within the scope of test_custom_eq
    def f(x, y):
        return x * x + (x * y) / (x + y) + x * (x + y)

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
    assert np.allclose(
        z_t.detach().numpy(), z_tt.data
    ), "Custom eq results do not match between PyTorch and tinytorch."
    assert np.allclose(
        x_t.grad.numpy(), x_tt.grad.data
    ), "Gradients for x do not match between PyTorch and tinytorch."
    assert np.allclose(
        y_t.grad.numpy(), y_tt.grad.data
    ), "Gradients for y do not match between PyTorch and tinytorch."


def test_matmul():
    x = np.random.rand(3, 4)
    y = np.random.rand(4, 3)

    x_t = torch.tensor(x, requires_grad=True)
    y_t = torch.tensor(y, requires_grad=True)

    x_tt = tinytorch.tensor(x, requires_grad=True)
    y_tt = tinytorch.tensor(y, requires_grad=True)

    z_t = x_t @ y_t
    z_t.sum().backward()

    z_tt = x_tt @ y_tt
    z_tt.sum().backward()

    # Assertions
    assert np.allclose(z_t.detach().numpy(), z_tt.data), "Matmul results do not match."
    assert np.allclose(
        x_t.grad.numpy(), x_tt.grad.data
    ), "Gradients for x do not match."
    assert np.allclose(
        y_t.grad.numpy(), y_tt.grad.data
    ), "Gradients for y do not match."


def test_pow():
    x = np.random.rand(3, 3)
    exponent = 2

    x_t = torch.tensor(x, requires_grad=True)
    x_tt = tinytorch.tensor(x, requires_grad=True)

    z_t = x_t ** (exponent)
    z_t.sum().backward()

    z_tt = x_tt ** (exponent)
    z_tt.sum().backward()

    # Assertions
    assert np.allclose(z_t.detach().numpy(), z_tt.data), "Pow results do not match."
    assert np.allclose(x_t.grad.numpy(), x_tt.grad.data), "Gradients do not match."


def test_tanh():
    x = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    x_tt = tinytorch.tensor(x, requires_grad=True)

    z_t = torch.tanh(x_t)
    z_t.sum().backward()

    z_tt = tinytorch.tanh(x_tt)
    z_tt.sum().backward()

    # Assertions
    assert np.allclose(z_t.detach().numpy(), z_tt.data), "Tanh results do not match."
    assert np.allclose(x_t.grad.numpy(), x_tt.grad.data), "Gradients do not match."


def test_relu():
    x = np.random.rand(3, 3)

    x_t = torch.tensor(x, requires_grad=True)
    x_tt = tinytorch.tensor(x, requires_grad=True)

    z_t = torch.relu(x_t)
    z_t.sum().backward()

    z_tt = tinytorch.relu(x_tt)
    z_tt.sum().backward()

    # Assertions
    assert np.allclose(z_t.detach().numpy(), z_tt.data), "ReLU results do not match."
    assert np.allclose(x_t.grad.numpy(), x_tt.grad.data), "Gradients do not match."


def test_mse_loss():
    y_pred = np.random.rand(2, 2)
    y_true = np.random.rand(2, 2)

    y_pred_t = torch.tensor(y_pred, requires_grad=True)
    y_true_t = torch.tensor(y_true, requires_grad=False)

    y_pred_tt = tinytorch.tensor(y_pred, requires_grad=True)
    y_true_tt = tinytorch.tensor(y_true, requires_grad=False)

    loss_t = torch.nn.functional.mse_loss(y_pred_t, y_true_t)
    loss_t.backward()

    loss_tt = tinytorch.mse_loss(y_pred_tt, y_true_tt)
    print(loss_t, loss_tt)
    loss_tt.backward()

    # Assertions
    assert np.allclose(
        loss_t.detach().numpy(), loss_tt.data
    ), "MSE Loss results do not match."
    assert np.allclose(
        y_pred_t.grad.numpy(), y_pred_tt.grad.data
    ), "Gradients do not match."


def test_cross_entropy():
    # Create random logits and labels
    logits = np.random.rand(5, 3)  # 5 samples, 3 classes
    labels = np.random.randint(0, 3, size=(5,))  # 5 samples, labels from 0 to 2

    print(f"{labels.shape=}")

    # Convert to PyTorch tensors
    logits_t = torch.tensor(logits, requires_grad=True)
    labels_t = torch.tensor(labels, requires_grad=False, dtype=torch.long)

    # Convert to tinytorch tensors
    logits_tt = tinytorch.tensor(logits, requires_grad=True)
    labels_tt = tinytorch.tensor(labels, requires_grad=False)

    # Compute loss using PyTorch
    loss_t = torch.nn.functional.cross_entropy(logits_t, labels_t)
    loss_t.backward()

    print(f"{logits.shape=} {labels_t.shape=} {loss_t.shape=}")

    # Compute loss using tinytorch (assuming cross_entropy is implemented)
    loss_tt = tinytorch.cross_entropy(logits_tt, labels_tt)
    loss_tt.sum().backward()

    # print(f"{loss_t=}")
    # print(f"{loss_tt=}")
    # Assertions
    assert np.allclose(
        loss_t.detach().numpy(), loss_tt.detach().numpy(), atol=1e-5
    ), "Cross-entropy Loss results do not match."
    assert np.allclose(
        logits_t.grad.numpy(), logits_tt.grad.data, atol=1e-7
    ), "Gradients do not match."


def test_transpose():
    x = np.random.rand(3, 4)  # Create a 3x4 random matrix

    x_t = torch.tensor(x, requires_grad=True)  # Create a PyTorch tensor
    x_tt = tinytorch.tensor(x, requires_grad=True)  # Create a tinytorch tensor

    # Perform the transpose operation using PyTorch
    z_t = x_t.t()
    z_t.sum().backward()  # Compute the gradients via backpropagation

    # Perform the transpose operation using tinytorch
    z_tt = x_tt.t()
    z_tt.sum().backward()  # Compute the gradients via backpropagation

    # Assert that the transpose operation results are the same for both PyTorch and tinytorch
    assert np.allclose(
        z_t.detach().numpy(), z_tt.data
    ), "Transpose results do not match between PyTorch and tinytorch."

    # Assert that the gradients are the same for both PyTorch and tinytorch
    assert np.allclose(
        x_t.grad.numpy(), x_tt.grad.data
    ), "Gradients do not match between PyTorch and tinytorch."


def test_max():
    x = np.random.rand(2, 3)  # Create a random 2x3 array
    axis = 1  # Axis along which to compute max

    x_t = torch.tensor(x, requires_grad=True)  # Create a PyTorch tensor
    x_tt = tinytorch.tensor(x, requires_grad=True)  # Create a tinytorch tensor

    # Compute max using PyTorch
    y_t, _ = torch.max(x_t, dim=axis)
    loss_t = y_t.sum()
    loss_t.backward()

    # Compute max using tinytorch
    y_tt = x_tt.max(axis=axis)
    # print(f"{x_tt.shape=} {y_tt.shape=}")
    loss_tt = y_tt.sum()
    loss_tt.backward()

    # Assertions
    assert np.allclose(y_t.detach().numpy(), y_tt.data), "Max results do not match."

    assert np.allclose(x_t.grad.numpy(), x_tt.grad.data), "Gradients do not match."


def test_stack_with_square():
    x = np.random.rand(3, 3)
    y = np.random.rand(3, 3)

    # Convert to PyTorch tensors
    x_t = torch.tensor(x, requires_grad=True)
    y_t = torch.tensor(y, requires_grad=True)

    # Perform stack operation in PyTorch
    z_t = torch.stack([x_t, y_t], dim=0)
    z_t = z_t**2  # Square the tensor
    z_t.sum().backward()

    # Assuming you've implemented stack and power operation in tinytorch
    # Convert to tinytorch tensors
    x_tt = tinytorch.tensor(x, requires_grad=True)
    y_tt = tinytorch.tensor(y, requires_grad=True)

    # Perform stack operation in tinytorch
    z_tt = tinytorch.stack([x_tt, y_tt], axis=0)
    z_tt = z_tt**2  # Square the tensor
    z_tt.sum().backward()

    # Verify that the output from PyTorch and tinytorch are the same
    assert np.allclose(
        z_t.detach().numpy(), z_tt.data
    ), "Stack operation results do not match between PyTorch and tinytorch."

    # Verify that the gradients are the same for both PyTorch and tinytorch
    assert np.allclose(
        x_t.grad.numpy(), x_tt.grad.data
    ), "Gradients for x do not match between PyTorch and tinytorch."

    # assert np.allclose(
    #     y_t.grad.numpy(), y_tt.grad.data
    # ), "Gradients for y do not match between PyTorch and tinytorch."


if __name__ == "__main__":
    test_cross_entropy()
