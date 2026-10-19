import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearReLU(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

class Conv2DPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init()
        self.conv2d_relu_pool_stack = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2,2, ceil_mode=True),
        )
    def forward(self, x):
        return self.conv2d_relu_pool_stack(x)
    
class DynamicMLP(nn.Module):
    """ 
    Still requires the flatten input size to be input into the network
    
    Produces K hidden layers of fixed D width. Outputs raw logits. 
    """
    def __init__(self, H, W, out_feat, K, D):
        super().__init__()
        in_feat = H*W
        layers = [LinearReLU(in_feat, D)]
        for _ in range(K-1):
            layers.append(LinearReLU(D, D))
        layers.append(nn.Linear(D, out_feat))

        self.mlp = nn.Sequential(*layers)

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

        layers = [Conv2DPool(C_i, H, kernel_size)]
        C_o = H*1
        for _ in range(n_pools-1):
            layers.append(Conv2DPool(C_o, C_o*2, kernel_size))
            C_o *= 2

        in_feat = 2*2*C_o
        layers.append(LinearReLU(in_feat, D))
        for _ in range(K-1):
            layers.append(LinearReLU(D, D))
        layers.append(nn.Linear(D, out_feat))

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)
        










