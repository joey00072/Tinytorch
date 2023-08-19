import numpy as np

class Tensor:
    def __init__(self,data):
        self.data = data if isinstance(data,np.ndarray) else np.array(data)
        self._ctx = None
        
    def __add__(self,other):
        fn = Function(Add,self,other)
        result = Add.forward(self,other)
        result._ctx = fn
        return result

    def __mul__(self,other):
        fn = Function(Mul,self,other)
        result = Mul.forward(self,other)
        result._ctx = fn
        return result
        
    
    def __repr__(self):
        return f"tensor({self.data})"
    
class Function:
    def __init__(self,op,*args):
        self.op = op
        self.args = args        

class Add:
    @staticmethod
    def forward(x,y):
        return Tensor(x.data + y.data)
    
    @staticmethod
    def backward(ctx,grad):
        x,y = ctx.args
        return Tensor([1]),Tensor([1])

class Mul:
    @staticmethod
    def forward(x,y):
        return Tensor(x.data * y.data) # z = x*y
    
    @staticmethod
    def backward(ctx,grad):
        x,y = ctx.args
        return  Tensor(y.data), Tensor(x.data) #  dz/dx, dz/dy
    
if __name__ == "__main__":
    x = Tensor([8])
    y = Tensor([5])    
    z = x*y
    print(z)
    