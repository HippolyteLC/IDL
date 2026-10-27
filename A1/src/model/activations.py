import torch.nn as nn
import torch

class SmoothedTanh(nn.Module):
    def __init__(self, k=1):
        super().__init__()
        self.k = k
    def forward(self, x):
        return torch.tanh(self.k*x)
    def __str__(self):
        return "SmoothedTanh"

class SmoothedSigmoid(nn.Module):
    def __init__(self, k=0.3):
        super().__init__()
        self.k = k
    def forward(self, x):
        return torch.sigmoid(self.k*x)
    def __str__(self):
        return "SmoothedSigmoid"

class L2NormActivation(nn.Module):
    def forward(self,x):
        return nn.functional.normalize(x, dim=-1)