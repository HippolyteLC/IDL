import torch.nn as nn
import torch.nn.functional as F
import numpy as np

D_max = 1024

PARAMETER_BUDGET = 1.5e9 # max number of parameters for an RTX 4060 GPU 

def init_weights(m, init_method, generator, verbose=False):
    if verbose:
        print(m)
    if type(m) is nn.Module:
        init_method(m, generator=generator)
        if verbose:
            print(m.weight) 

def compute_parameters(m):
    if isinstance(m, Conv2DActivationPool):
        return m.in_channels * m.out_channels * m.kernel_size + m.out_channels

class LinearActivation(nn.Module):
    def __init__(self, in_feat, out_feat, activation, dropout=0):
        super().__init__()
        self.a = activation
        self.flatten = nn.Flatten()
        layers = [
            nn.Linear(in_feat, out_feat),
            self.a
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

class Conv2DActivationPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, dropout=0):
        super().__init()
        self.a = activation
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            self.a,
            nn.MaxPool2d(2,2, ceil_mode=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.conv2d_relu_pool_stack = nn.Sequential(*layers)
    def forward(self, x):
        return self.conv2d_relu_pool_stack(x)
    
class DynamicMLP(nn.Module):
    """ 
    Still requires the flatten input size to be input into the network
    
    Produces K hidden layers of fixed D width. Outputs raw logits. 
    """
    def __init__(self, H, W, out_feat, K, D=None):
        super().__init__()
        in_feat = H*W
        if D is None:
            D = min(D_max, np.floor(np.sqrt(PARAMETER_BUDGET/K)))
        layers = [LinearActivation(in_feat, D)]
        for _ in range(K-1):
            layers.append(LinearActivation(D, D))
        layers.append(nn.Linear(D, out_feat))
        
        self.mlp = nn.Sequential(*layers)

        self.width = D

    def forward(self, x):
        return self.mlp(x)
    
class DynamicCNN(nn.Module):
    """ 
    Grow the number of channels as a factor of the number of possible kernels. 
    First consider the input size, compute number of possible kernels, and then 

    Uses Conv2D, Relu, MaxPool sequences to halve the input size repeatedly. Ceil_mode
    is True so that max pool does not throw away modulo leftover rows/cols. 
    Does this dim halving up to a output size of 2x2. The n_channels is first set to the input width,
    it is then doubled each time as the input H, W are halved.
    
    At the end K hidden FC layers are appended with fixed width D. 
    OUtputs raw logits. 
    """
    def __init__(self, H, W, C_i, out_feat, kernel_size, K, D):
        super().__init__()
        # have some base channel output, then
        n_pools = max(0, np.ceil(np.log2(H/2)))

        layers = [Conv2DActivationPool(C_i, H, kernel_size)]
        C_o = H*1
        for _ in range(n_pools-1):
            layers.append(Conv2DActivationPool(C_o, C_o*2, kernel_size))
            C_o *= 2

        used_parameter_budget = sum([m.apply(compute_parameters) for m in layers])
        left_over_params = PARAMETER_BUDGET - used_parameter_budget

        if D is None:
            D = min(D_max, np.floor(np.sqrt(PARAMETER_BUDGET/K)))
        
        in_feat = 2*2*C_o
        layers.append(LinearActivation(in_feat, D))
        for _ in range(K-1):
            layers.append(LinearActivation(D, D))
        layers.append(nn.Linear(D, out_feat))

        self.cnn = nn.Sequential(*layers)

        self.width = D

    def forward(self, x):
        return self.cnn(x)
        










