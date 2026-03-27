# 神经网络的基本骨架 nn.Module
import torch
from torch import nn

class pp(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

pp = pp()
x = torch.randn(1, 3, 32, 32)
output = pp(x)
print(output.shape)
