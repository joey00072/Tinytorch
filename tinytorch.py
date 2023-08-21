from __future__ import annotations
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

    def clone(self, requires_grad=False) -> Tensor:
        return Tensor(self.data.clone(), requires_grad=requires_grad)

    def _undo_broadcast(self, tensor: Tensor, grad: Tensor):
        data = tensor.data
        grad = grad.data
        while data.shape != grad.shape:
            grad = grad.sum(axis=0, keepdims=(len(grad.shape) == 1))
        return Tensor(grad)

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


class Mul(Function):
    @staticmethod
    def forward(x, y):
        return Tensor(x.data * y.data)  # z = x*y

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.args
        return Tensor(y.data) * grad, Tensor(x.data) * grad  #  dz/dx, dz/dy


def ones(shape, requires_grad=False) -> Tensor:
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def zeros(shape, requires_grad=False) -> Tensor:
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


if __name__ == "__main__":
    x = Tensor([1, 2, 3])
    y = Tensor([4])
    z = x * y
    z.backward(ones(z.shape))  # don't forget this
    print(f"X: {x} grad: {x.grad}")
    print(f"Y: {y} grad: {y.grad}")

    print("="*100)
    x = Tensor([1, 2, 3],requires_grad=True)
    y = Tensor([4],requires_grad=True)
    z = x * y
    z.backward(ones(z.shape))  # don't forget this
    print(f"X: {x} grad: {x.grad}")
    print(f"Y: {y} grad: {y.grad}")
