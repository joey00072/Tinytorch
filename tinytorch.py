from __future__ import annotations
import numpy as np


class Tensor:
    def __init__(self, data):
        self.data: np.ndarray = Tensor._data_to_numpy(data)
        self.grad: Tensor = None
        self._ctx: Function = None

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

    @staticmethod
    def ensure_tensor(data):
        if isinstance(data, Tensor):
            return data
        return Tensor(data)

    def __add__(self, other) -> Tensor:
        return Add.apply(self, Tensor.ensure_tensor(other))

    def __mul__(self, other) -> Tensor:
        return Mul.apply(self, Tensor.ensure_tensor(other))

    def __radd__(self, other) -> Tensor:
        return Add.apply(self, Tensor.ensure_tensor(other))

    def __rmul__(self, other) -> Tensor:
        return Mul.apply(self, Tensor.ensure_tensor(other))

    def __iadd__(self, other) -> Tensor:
        return Add.apply(self, Tensor.ensure_tensor(other))

    def __imul__(self, other) -> Tensor:
        return Mul.apply(self, Tensor.ensure_tensor(other))

    def __repr__(self):
        return f"tensor({self.data})"

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def detach(self) -> Tensor:
        self._ctx = None
        return self

    def clone(self) -> Tensor:
        return Tensor(self.data.clone())

    def backward(self, grad=None):
        if self._ctx is None:
            return

        if grad is None:
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
        result._ctx = ctx
        return result

    @staticmethod
    def forward(self, *args):
        raise NotImplementedError

    @staticmethod
    def backward(self, *args):
        raise NotImplementedError


class Add(Function):
    @staticmethod
    def forward(x, y):
        return Tensor(x.data + y.data)

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.args
        return Tensor([1]) * grad, Tensor([1]) * grad


class Mul(Function):
    @staticmethod
    def forward(x, y):
        return Tensor(x.data * y.data)  # z = x*y

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.args
        return Tensor(y.data) * grad, Tensor(x.data) * grad  #  dz/dx, dz/dy


def ones(shape) -> Tensor:
    return Tensor(np.ones(shape))


def zeros(shape) -> Tensor:
    return Tensor(np.zeros(shape))


if __name__ == "__main__":

    def f(x):
        return x * x * x + x

    x = Tensor([1.2])

    z = f(x)
    z.backward()
    print(f"X: {x} grad: {x.grad}")
