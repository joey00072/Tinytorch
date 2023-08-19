import numpy as np


class Tensor:
    def __init__(self,data):
        self.data = data if isinstance(data,np.ndarray) else np.array(data)
        
    def __add__(self,other):
        return Tensor(self.data + other.data)

    def __mul__(self,other):
        return Tensor(self.data + other.data)