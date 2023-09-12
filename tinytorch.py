from __future__ import annotations
import math
from typing import Any
import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data: np.ndarray = Tensor._data_to_numpy(data)
        self.grad: Tensor = None
        self._ctx: Function = None
        self.requires_grad: bool = requires_grad

    @staticmethod
    def _data_to_numpy(data):
        if isinstance(data, (int, float)):
            return np.array([data])
        if isinstance(data, (list, tuple)):
            return np.array(data)
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, Tensor):
            return data.data.copy()
        raise ValueError("Invalid value passed to tensor")

    def __repr__(self) -> str:
        s = f"{self.data} "
        if self.requires_grad:
            s += f" requires_grad={self.requires_grad}"
        if self._ctx is not None:
            op, fn = self._ctx.op.__name__, self._ctx.op.backward.__name__
            s += f" grad_fn=<{op}.{fn}>"
        return f"Tensor:({s})"

    @staticmethod
    def ensure_tensor(data):
        if isinstance(data, Tensor):
            return data
        return Tensor(data)

    def __add__(self, other) -> Tensor:
        return Add.apply(self, Tensor.ensure_tensor(other))

    def __sub__(self, other) -> Tensor:
        return Sub.apply(self, Tensor.ensure_tensor(other))

    def __mul__(self, other) -> Tensor:
        return Mul.apply(self, Tensor.ensure_tensor(other))

    def __truediv__(self, other) -> Tensor:
        other = Tensor.ensure_tensor(other)
        return self * other.reciprocal()

    def __matmul__(self, other):
        return MatMul.apply(self, Tensor.ensure_tensor(other))

    def __radd__(self, other) -> Tensor:
        return Add.apply(self, Tensor.ensure_tensor(other))

    def __rsub__(self, other) -> Tensor:
        return Sub.apply(Tensor.ensure_tensor(other), self)

    def __rmul__(self, other) -> Tensor:
        return Mul.apply(self, Tensor.ensure_tensor(other))

    def __rtruediv__(self, other) -> Tensor:
        other = Tensor.ensure_tensor(other)
        return other * self.reciprocal()

    def __rmatmul__(self, other) -> Tensor:
        return MatMul.apply(Tensor.ensure_tensor(other), self)

    def __iadd__(self, other) -> Tensor:
        return Add.apply(self, Tensor.ensure_tensor(other))

    def __isub__(self, other) -> Tensor:
        return Sub.apply(self, Tensor.ensure_tensor(other))

    def __imul__(self, other) -> Tensor:
        return Mul.apply(self, Tensor.ensure_tensor(other))

    def __itruediv__(self, other) -> Tensor:
        other = Tensor.ensure_tensor(other)
        return self * other.reciprocal()

    def __neg__(self) -> Tensor:
        return -1 * self

    def __pow__(self, other) -> Tensor:
        return Power.apply(self, Tensor.ensure_tensor(other))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, slice_args):
        return Slice.apply(self, slice_args)

    def reciprocal(self) -> Tensor:
        return Reciprocal.apply(self)

    def exp(self) -> Tensor:
        return Exp.apply(self)

    def reshape(self, *shape) -> Tensor:
        if isinstance(shape[0], tuple):
            shape = shape[0]
        curr = math.prod(self.shape)
        target = math.prod((s for s in shape if s != -1))
        shape = tuple(curr // target if s == -1 else s for s in shape)
        return Reshape.apply(self, shape)

    def transpose(self, dim1=-2, dim2=-1):
        num_axes = len(self.shape)
        dim1, dim2 = (dim1 + num_axes) % num_axes, (dim2 + num_axes) % num_axes
        axes = list(range(num_axes))
        axes[dim1], axes[dim2] = dim2, dim1
        return Transpose.apply(self, axes)

    def t(self):
        return self.transpose()

    def float(self):
        self.data = self.data.astype(np.float32)
        return self

    def to(self, device):
        self.device = device

    def tolist(self):
        return self.data.tolist()

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def size(self) -> tuple:
        return self.data.size

    def _reduce_shape(self, axis=None):
        # Determine which axes to reduce
        if axis is None:
            reduction_axes = list(range(len(self.shape)))
        elif isinstance(axis, int):
            reduction_axes = [axis]
        else:
            reduction_axes = list(axis)

        reduction_axes = [a if a >= 0 else a + len(self.shape) for a in reduction_axes]

        post_reduction_shape = tuple(
            self.shape[i] for i in range(len(self.shape)) if i not in reduction_axes
        )

        shape_with_kept_dims = tuple(
            1 if i in reduction_axes else self.shape[i] for i in range(len(self.shape))
        )

        return post_reduction_shape, shape_with_kept_dims

    def sum(self, axis=None, keepdims=False) -> Tensor:
        shape, new_shape = self._reduce_shape(axis)
        shape = shape if (shape) != () else (1,)
        ret: Tensor = Sum.apply(self, new_shape)
        return ret if keepdims else ret.reshape(shape)

    def max(self, axis=None, keepdims=False) -> Tensor:
        shape, new_shape = self._reduce_shape(axis)
        shape = shape if (shape) != () else (1,)
        ret: Tensor = Max.apply(self, new_shape)
        return ret if keepdims else ret.reshape(shape)

    def mean(x: Tensor, axis=None, keepdim=False):
        out = x.sum(axis=axis, keepdims=keepdim)
        return out * (math.prod(out.shape) / math.prod(x.shape))

    def zero_(self):
        self.data = np.zeros_like(self.data)

    def detach(self) -> Tensor:
        self._ctx = None
        return self

    def clone(self, requires_grad=False) -> Tensor:
        return Tensor(self.data.clone(), requires_grad=requires_grad)

    def numpy(self):
        return self.data.copy()

    def item(self):
        if self.size != 1:
            raise RuntimeError("item can not be called on non 1 tensor")
        return self.data.sum()

    def _undo_broadcast(self, tensor: Tensor, grad: Tensor):
        data = tensor.data
        grad = grad.data
        # print(f"{data.shape= },{grad.shape=}")
        while len(data.shape) != len(grad.shape):
            grad = grad.sum(axis=0, keepdims=(len(grad.shape) == 1))
        return Tensor(grad)

    def backward(self, grad=None):
        if self._ctx is None:
            return
        # print(f"Root: {self.shape= } {self._ctx.op.__name__}")

        if grad is None:
            if self.size != 1:
                raise RuntimeError("Backward can not be called on non zero tensor")
            grad = Tensor([1.0])
            self.grad = grad

        op = self._ctx.op
        child_nodes = self._ctx.args

        grads = op.backward(self._ctx, grad)
        if len(self._ctx.args) == 1:
            grads = [grads]

        for tensor, grad in zip(child_nodes, grads):
            if grad is None:
                continue
            # print(f"{self._ctx.op.__name__} {self.shape=} {tensor.shape=} {grad.shape=}")
            # try:
            #     for t in self._ctx.args:
            #         print(f"{t.shape=}")
            # except:
            #     pass
            grad = self._undo_broadcast(tensor, grad)
            if tensor.grad is None:
                tensor.grad = Tensor(np.zeros_like(tensor.data))
            tensor.grad += grad.detach()
            tensor.backward(grad)


class Function:
    def __init__(self, op, *args):
        self.op: Function = op
        self.args: list[Tensor] = args

    @classmethod
    def apply(cls, *args):
        ctx = Function(cls, *args)
        result = cls.forward(*args)
        if Function._is_part_of_graph(ctx):
            result._ctx = ctx
        return result

    @staticmethod
    def _is_part_of_graph(ctx: Function):
        for node in ctx.args:
            if isinstance(node, Tensor) and (
                node.requires_grad or node._ctx is not None
            ):
                return True
        return False

    @staticmethod
    def forward(self, *args) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(self, *args) -> Tensor:
        raise NotImplementedError


class Add(Function):
    @staticmethod
    def forward(x, y):
        return Tensor(x.data + y.data)

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.args
        return Tensor([1]) * grad, Tensor([1]) * grad


class Sub(Function):
    @staticmethod
    def forward(x, y):
        return Tensor(x.data - y.data)

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.args
        return Tensor([1]) * grad, Tensor([-1]) * grad


class Mul(Function):
    @staticmethod
    def forward(x, y):
        return Tensor(x.data * y.data)  # z = x*y

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.args
        return Tensor(y.data) * grad, Tensor(x.data) * grad  #  dz/dx, dz/dy


class Reciprocal(Function):
    @staticmethod
    def forward(x):
        return Tensor(1.0 / x.data)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.args
        return Tensor(-1.0 / (x.data**2)) * grad


class MatMul(Function):
    @staticmethod
    def forward(x, y):
        return Tensor(x.data @ y.data)

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.args

        def transpose_last_axis(x: np.ndarray):
            dim1, dim2 = -2, -1
            num_axes = len(x.shape)
            dim1, dim2 = (dim1 + num_axes) % num_axes, (dim2 + num_axes) % num_axes
            axes = list(range(num_axes))
            axes[dim1], axes[dim2] = dim2, dim1
            return x.transpose(axes)

        grad_x = np.matmul(grad.data, transpose_last_axis(y.data))
        grad_y = transpose_last_axis(x.data) @ grad.data
        return Tensor(grad_x), Tensor(grad_y)


class Exp(Function):
    @staticmethod
    def forward(x):
        return Tensor(np.exp(x.data))

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.args
        return Tensor(np.exp(x.data)) * grad


class Reshape(Function):
    @staticmethod
    def forward(x, shape) -> Tensor:
        return Tensor(x.data.reshape(tuple(shape)))

    @staticmethod
    def backward(ctx: Function, grad: Tensor) -> Tensor:
        x, _ = ctx.args
        return Tensor(grad.data.reshape(x.shape)), None


class Transpose(Function):
    @staticmethod
    def forward(x: Tensor, axes) -> Tensor:
        return Tensor(x.data.transpose(axes))

    @staticmethod
    def backward(ctx: Function, grad: Tensor) -> Tensor:
        _, axes = ctx.args
        return Tensor(grad.data.transpose(axes)), None


class Slice(Function):
    @staticmethod
    def forward(x, slice_args):
        return Tensor(x.data[slice_args])

    @staticmethod
    def backward(ctx, grad):
        x, slice_args = ctx.args
        grad_x = np.zeros_like(x.data)
        grad_x[slice_args] = grad.data
        return Tensor(grad_x), None


class Sum(Function):
    def forward(x: Tensor, new_shape: tuple) -> Tensor:
        axis = tuple(
            idx for idx, (a, b) in enumerate(zip(x.shape, new_shape)) if a != b
        )
        x = x.data.sum(axis, keepdims=True)
        return Tensor(x)

    def backward(ctx: Function, grad: Tensor) -> tuple[Tensor | None]:
        x, _ = ctx.args
        return Tensor(np.broadcast_to(grad.data, x.shape)), None


class Max(Function):
    @staticmethod
    def forward(x: Tensor, new_shape: tuple) -> Tensor:
        axis = tuple(
            idx for idx, (a, b) in enumerate(zip(x.shape, new_shape)) if a != b
        )
        max_values = np.max(x.data, axis=axis, keepdims=True)
        return Tensor(max_values)

    @staticmethod
    def backward(ctx: Function, grad: Tensor) -> Tensor:
        x, new_shape = ctx.args
        max_values = Max.forward(x, new_shape).data
        axis = tuple(
            idx for idx, (a, b) in enumerate(zip(x.shape, grad.shape)) if a != b
        )

        # Create a mask where the max values are
        max_mask = (x.data == np.broadcast_to(max_values, x.shape)).astype(int)

        # Count the number of max values along the axis
        count_max = np.sum(max_mask, axis=axis, keepdims=True)

        # Normalize the mask by the number of max values along the axis
        normalized_mask = max_mask / (np.broadcast_to(count_max, x.shape) + 1e-20)

        # Broadcast grad to the shape of x
        grad_broadcasted = np.broadcast_to(grad.data, x.shape)

        # Compute the gradient
        grad_x = normalized_mask * grad_broadcasted

        return Tensor(grad_x), None


class Power(Function):
    @staticmethod
    def forward(x, y):
        return Tensor(np.power(x.data, y.data))

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.args
        grad_x = y.data * np.power(x.data, y.data - 1) * grad.data
        return Tensor(grad_x), None


class Tanh(Function):
    @staticmethod
    def forward(x):
        return Tensor(np.tanh(x.data))

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.args
        grad_tanh = 1 - np.tanh(x.data) ** 2
        return Tensor(grad_tanh) * grad


class ReLU(Function):
    @staticmethod
    def forward(x):
        return Tensor(np.maximum(0, x.data))

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.args
        grad_relu = np.where(x.data > 0, 1, 0)
        return Tensor(grad_relu) * grad


class Sin(Function):
    @staticmethod
    def forward(x):
        return Tensor(np.sin(x.data))

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.args
        return Tensor(np.cos(x.data)) * grad


class CrossEntropy(Function):
    @staticmethod
    def forward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        exps = np.exp(y_pred.data - np.max(y_pred.data, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True) + 1e-22
        log_likelihood = -np.log(
            probs[np.arange(len(y_true.data)), y_true.data.astype(int)]
        )
        return Tensor(np.mean(log_likelihood))

    @staticmethod
    def backward(ctx: Function, grad: Tensor) -> list[Tensor]:
        y_pred, y_true = ctx.args
        exps = np.exp(y_pred.data - np.max(y_pred.data, axis=1, keepdims=True)) + 1e-22
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        d_loss = np.zeros_like(probs)
        d_loss[np.arange(len(y_true.data)), y_true.data.astype(int)] -= 1
        d_loss += probs
        d_loss /= len(y_true.data)
        return [Tensor(d_loss) * grad, None]


class Stack(Function):
    @staticmethod
    def forward(*args):
        *tensors, axis = args
        data = [t.data for t in tensors]
        stacked_data = np.stack(data, axis=axis)
        return Tensor(stacked_data)

    @staticmethod
    def backward(ctx: Function, grad: Tensor):
        *tensors, axis = ctx.args
        grad_data = grad.data
        grads = np.split(grad_data, grad_data.shape[axis], axis=axis)
        return tuple([*[Tensor(g) for g in grads], None])


def stack(tensors: list[Tensor], axis: int = 0) -> Tensor:
    return Stack.apply(*tensors, axis)


def dropout(x: Tensor, p: int):
    mask = Tensor(np.random.choice([0, 1], size=x.shape, p=[1 - p, p]))
    return x * mask


def cross_entropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return CrossEntropy.apply(y_pred, y_true)


def exp(x) -> Tensor:
    return Exp.apply(x)


def tanh(x) -> Tensor:
    return Tanh.apply(x)


def relu(x) -> Tensor:
    return ReLU.apply(x)


def sigmoid(x) -> Tensor:
    return 1 / (1 + exp(-1 * x))


def softmax(x: Tensor, dim: int = 0) -> Tensor:
    e_x = exp(x - x.max(axis=dim, keepdims=True))
    return e_x / e_x.sum(axis=dim, keepdims=True)


def sin(x) -> Tensor:
    return Sin.apply(x)


def cos(x) -> Tensor:
    return sin(x + np.pi)


def mean(x: Tensor, axis=None, keepdims=False):
    out = x.sum(axis=axis, keepdims=keepdims)
    return out * (math.prod(out.shape) / math.prod(x.shape))


def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return ((y_pred - y_true) ** 2).sum() / y_pred.size


def ones(shape, requires_grad=False) -> Tensor:
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def zeros(shape, requires_grad=False) -> Tensor:
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def rand(*shape, requires_grad=False) -> Tensor:
    if isinstance(shape[0], tuple):
        shape = shape[0]
    return Tensor(np.random.rand(*shape), requires_grad=requires_grad)


def tensor(*args, **kwargs):
    return Tensor(*args, **kwargs)


def arange(*args, requires_grad=False):
    return Tensor(np.arange(*args), requires_grad=requires_grad)


class Device:
    def __init__(self, name: str = "cpu"):
        self.name = name


class Parameter(Tensor):
    def __init__(self, tensor):
        super().__init__(tensor, requires_grad=True)


class Module:
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | Any:
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[Parameter]:
        params = []
        for key, value in self.__dict__.items():
            if isinstance(value, Parameter):
                params.append(value)
            if isinstance(value, Module):
                params.extend(value.parameters())
            if isinstance(value, ModuleList):
                for module in value:
                    params.extend(module.parameters())
        return params

    def state_dict(self):
        state = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Parameter):
                state[key] = value.data
            if isinstance(value, Module):
                state[key] = value.state_dict()
            if isinstance(value, list):  # Assuming ModuleList is a list for this demo
                state[key] = [module.state_dict() for module in value]
        return state

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            attr = getattr(self, key, None)
            if attr is None:
                raise KeyError(f"Key {key} not found in module's state.")
            if isinstance(attr, Parameter):
                attr.data = value
            elif isinstance(attr, Module):
                attr.load_state_dict(value)
            elif isinstance(attr, list):  # Assuming ModuleList
                for param, state in zip(attr, value):
                    param.load_state_dict(state)

    def forward(self, *args, **kwargs):
        raise NotImplemented


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            rand((out_features, in_features)) / math.sqrt(in_features)
        )
        self.bias = Parameter(zeros(out_features)) if bias else None

    def forward(self, x):
        x = x @ self.weight.t()
        if self.bias:
            x = x + self.bias
        return x


class ModuleList(Module, list):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            self += modules

    def append(self, module: Module):
        super().append(module)

    def __setitem__(self, i: int, module: Module):
        return super().__setitem__(i, module)

    def parameters(self):
        params = []
        for module in self:
            params.extend(module.parameters())
        return params


class Optimizer:
    def __init__(self, params: list[Tensor], lr: int) -> None:
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for params in self.params:
            params.grad = zeros(params.shape)


class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: int) -> None:
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            param.data -= param.grad.data * self.lr


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(param.data) for param in self.params]
        self.v = [np.zeros_like(param.data) for param in self.params]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data
            self.m[i] = self.betas[0] * self.m[i] + (1.0 - self.betas[0]) * grad
            self.v[i] = self.betas[1] * self.v[i] + (1.0 - self.betas[1]) * grad**2
            m_hat = self.m[i] / (1.0 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1.0 - self.betas[1] ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


if __name__ == "__main__":
    x = arange(10)
    z = x[1:4]
    print(z)
