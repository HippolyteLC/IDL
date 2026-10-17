import torch.nn as nn
import torch.nn.functional as F


# K = 3
# D = 1 << 9


class DynamicMLP(nn.Module):
    """
    Module with fixed width at each hidden layers.
    With K=0, it is a simple Perceptron,
    with K>=1 it is an MLP.
    """
    def __init__(self, in_feat, out_feat, K, D):
        if K <= 0:
            self.layers = nn.ModuleList([nn.Linear(in_feat, out_feat)])
        else:
            self.layers = nn.ModuleList([nn.Linear(in_feat, D)])
            for _ in range(K-1):
                self.layers.append(nn.Linear(D, D))
            self.layers.append(nn.Linear(D, out_feat))
        self.activation = nn.ReLU()

    def forward(self, x):
        return [self.activation(lay(x)) for lay in self.layers][-1]


class LinearReLU(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.ReLU(),
        )
    def forward(self, x):
        return
class MLP(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.ReLU(),
            nn.Linear(512, out_feat)
        )



model = nn.Sequent
