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
        if grad is None:
            grad = Tensor([1.])
            self.grad = grad
        
        op = self._ctx.op
        child_nodes = self._ctx.args
        
        grads = op.backward(self._ctx,grad)
        
        for tensor,grad in zip(child_nodes,grads):
            tensor.grad = grad
            

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
        return Tensor([1]), Tensor([1])


class Mul:
    @staticmethod
    def forward(x, y):
        return Tensor(x.data * y.data)  # z = x*y

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.args
        return Tensor(y.data), Tensor(x.data)  #  dz/dx, dz/dy


if __name__ == "__main__":
    x = Tensor([8])
    y = Tensor([5])
    
    print("Add")
    z = x + y
    print(z)
    z.backward()
    print(f"x: {x} , grad {x.grad}")
    print(f"y: {y} , grad {y.grad}")
    print("="*100)
    
    print("Mul")
    z = x * y
    print(z)
    z.backward()
    print(f"x: {x} , grad {x.grad}")
    print(f"y: {y} , grad {y.grad}")
    print("="*100)
    
    
    
