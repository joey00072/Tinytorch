import numpy as np


class Tensor:
    def __init__(self,data):
        self.data = data if isinstance(data,np.ndarray) else np.array(data)
        
    def __add__(self,other):
        return Tensor(self.data + other.data)

    def __mul__(self,other):
        return Tensor(self.data + other.data)
    
    def __repr__(self):
        return f"tensor({self.data})"
    
    
if __name__ == "__main__":
    x = Tensor([8])
    y = Tensor([5])    
    z = x+y

    print(z)
    