import numpy as np


class Tensor:
    def __init__(self, data):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.grad = None
        self._ctx = None
        

    def __add__(self, other):
        fn = Function(Add, self, other)
        result = Add.forward(self, other)
        result._ctx = fn
        return result

    def __mul__(self, other):
        fn = Function(Mul, self, other)
        result = Mul.forward(self, other)
        result._ctx = fn
        return result

    def __repr__(self):
        return f"tensor({self.data})"

    def backward(self,grad=None):
        if self._ctx is None:
            return 
        
        if grad is None:
            grad = Tensor([1.])
            self.grad = grad
        
        op = self._ctx.op
        child_nodes = self._ctx.args
        
        grads = op.backward(self._ctx,grad)
        
        for tensor,grad in zip(child_nodes,grads):
            if tensor.grad is None:
                tensor.grad = Tensor(np.zeros_like(self.data))
            tensor.grad += grad
            tensor.backward(grad)
    

class Function:
    def __init__(self, op, *args):
        self.op = op
        self.args = args


class Add:
    @staticmethod
    def forward(x, y):
        return Tensor(x.data + y.data)

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.args
        return Tensor([1])*grad, Tensor([1]) *grad


class Mul:
    @staticmethod
    def forward(x, y):
        return Tensor(x.data * y.data)  # z = x*y

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.args
        return Tensor(y.data)*grad, Tensor(x.data)*grad #  dz/dx, dz/dy




if __name__ == "__main__":
    
    def f(x):
        return x*x*x + x
    
    x = Tensor([1.2])
    
    z = f(x)
    z.backward()
    print(f"X: {x} grad: {x.grad}")

    
    
