from __future__ import annotations
import math
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

    def __radd__(self, other) -> Tensor:
        return Add.apply(self, Tensor.ensure_tensor(other))

    def __rsub__(self, other) -> Tensor:
        return Sub.apply(Tensor.ensure_tensor(other), self)

    def __rmul__(self, other) -> Tensor:
        return Mul.apply(self, Tensor.ensure_tensor(other))

    def __rtruediv__(self, other) -> Tensor:
        other = Tensor.ensure_tensor(other)
        return other * self.reciprocal()

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

    def reciprocal(self) -> Tensor:
        return Reciprocal.apply(self)

    def exp(self) -> Tensor:
        return Exp.apply(self)
    
    def reshape(self,shape):
        curr = math.prod(self.shape)
        target = math.prod((s for s in shape if s!=-1))
        shape = tuple(curr//target if s==-1 else s for s in shape)
        return Reshape.apply(self,shape)

    def _reduce_shape(self,axis=None):
        # Determine which axes to reduce
        if axis is None:
            reduction_axes = list(range(len(self.shape)))
        elif isinstance(axis, int):
            reduction_axes = [axis]
        else:
            reduction_axes = list(axis)
        
        reduction_axes = [a if a >= 0 else a + len(self.shape) for a in reduction_axes]
        
        post_reduction_shape = [self.shape[i] for i in range(len(self.shape)) if i not in reduction_axes]
        
        shape_with_kept_dims = tuple(1 if i in reduction_axes else self.shape[i] for i in range(len(self.shape)))
        
        return post_reduction_shape, shape_with_kept_dims

    def sum(self, axis=None, keepdims=False) -> Tensor:
        shape , new_shape =  self._reduce_shape(axis)
        ret:Tensor = Sum.apply(self,new_shape)
        return ret if keepdims else ret.reshape(shape=shape)
  

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def detach(self) -> Tensor:
        self._ctx = None
        return self

    def clone(self, requires_grad=False) -> Tensor:
        return Tensor(self.data.clone(), requires_grad=requires_grad)

    def numpy(self):
        return self.data.copy()

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
    def forward(x,shape) -> Tensor:
        return Tensor(x.data.reshape(tuple(shape)))
    
    @staticmethod
    def backward(ctx:Function,grad:Tensor) -> Tensor:
        print(f"{ctx.args=}")
        x,_ = ctx.args
        return Tensor(grad.data.reshape(x.shape)),None


class Sum(Function):
    def forward(x:Tensor, new_shape:tuple) -> Tensor:
      axis = tuple(idx for idx,(a,b) in enumerate(zip(x.shape, new_shape)) if a != b)
      x = x.data.sum(axis, keepdims=True)
      return Tensor(x)
  
    def backward(ctx, grad:Tensor) -> Tensor:
      x,_ = ctx.args
      return Tensor(np.broadcast_to(grad.data, x.shape)) , None

def ones(shape, requires_grad=False) -> Tensor:
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def zeros(shape, requires_grad=False) -> Tensor:
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def rand(shape, requires_grad=False) -> Tensor:
    return Tensor(np.random.rand(shape), requires_grad=requires_grad)


def tensor(*args, **kwargs):
    return Tensor(*args, **kwargs)


if __name__ == "__main__":
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([4], requires_grad=True)
    z = x * y
    z.backward(ones(z.shape))  # don't forget this
    print(f"X: {x} grad: {x.grad}")
    print(f"Y: {y} grad: {y.grad}")
