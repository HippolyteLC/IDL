import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch

def init_weights(m, init_method, verbose=False):
    if verbose:
        print(m)
    if type(m) == nn.Linear:
        init_method(m.weight)
        if verbose:
            print(m.weight) 
    elif type(m) == nn.Conv2d:
        init_method(m.weight)
        if verbose:
            print(m.weight) 
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
class LinearActivation(nn.Module):
    def __init__(self, in_feat, out_feat, activation, dropout, bias, batch_norm):
        super().__init__()
        self.a = activation
        self.bias = bias
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_feat, out_feat, bias=bias)
        layers = [
            nn.Linear(in_feat, out_feat, bias=bias),
            self.a()
        ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm1d(out_feat))
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

class Conv2DActivationPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, dropout, bias, batch_norm):
        super().__init__()
        self.a = activation
        self.bias = bias
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', bias=bias),
            self.a(),
            nn.MaxPool2d(2,2, ceil_mode=True),
        ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        self.conv2d_relu_pool_stack = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv2d_relu_pool_stack(x)
    
class DynamicMLP(nn.Module):
    """ 
    Still requires the flatten input size to be input into the network
    
    Produces K hidden layers of fixed D width. Outputs raw logits. 
    """
    def __init__(self, H, W, C_i, out_feat, K, D=None, dropout=0, activation=None, bias=False, batch_norm=False):
        super().__init__()
        in_feat = H*W*C_i
        layers = [LinearActivation(in_feat, D, activation, dropout=dropout, bias=bias, batch_norm=batch_norm)]
        for _ in range(K-1):
            layers.append(LinearActivation(D, D, activation, dropout=dropout, bias=bias, batch_norm=batch_norm))
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
    Does this dim halving up to a output size of 4x4. The n_channels is first set to the input width,
    it is then doubled each time as the input H, W are halved.
    
    At the end K hidden FC layers are appended with fixed width D. 
    OUtputs raw logits. 
    """
    def __init__(self, H, W, C_i, out_feat, kernel_size, K, D, dropout=0, activation=None, bias=False, batch_norm=False):
        super().__init__()
        n_pools = int(max(0, np.ceil(np.log2(H/(1<<2)))))

        layers = [Conv2DActivationPool(C_i, H, kernel_size, activation, dropout=dropout, bias=bias, batch_norm=batch_norm)]
        C_o = H*1
        for _ in range(n_pools-1):
            layers.append(Conv2DActivationPool(C_o, C_o*2, kernel_size, activation, dropout=dropout, bias=bias, batch_norm=batch_norm))
            C_o *= 2

        conv = nn.Sequential(*layers)
        with torch.no_grad():
            in_feat = conv(torch.zeros(1, C_i, H, W)).numel()
        layers.append(LinearActivation(in_feat, D, activation, dropout=dropout, bias=bias, batch_norm=batch_norm))
        for _ in range(K-1):
            layers.append(LinearActivation(D, D, activation, dropout=dropout, bias=bias, batch_norm=batch_norm))
        layers.append(nn.Linear(D, out_feat))

        self.cnn = nn.Sequential(*layers)

        self.width = D

    def forward(self, x):
        return self.cnn(x)

class ClockCNN(nn.Module):
    def __init__(self, H, W, C_i, out_feat, kernel_size, K, D, dropout=0, activation=None, bias=False, batch_norm=False):
        super().__init__(H, W, C_i, out_feat, kernel_size, K, D, dropout=0, activation=None, bias=False, batch_norm=False)
        n_pools = int(max(0, np.ceil(np.log2(H/(1<<2)))))

        layers = [Conv2DActivationPool(C_i, H, kernel_size, activation, dropout=dropout, bias=bias, batch_norm=batch_norm)]
        C_o = H*1
        for _ in range(n_pools-1):
            layers.append(Conv2DActivationPool(C_o, C_o*2, kernel_size, activation, dropout=dropout, bias=bias, batch_norm=batch_norm))
            C_o *= 2

        conv = nn.Sequential(*layers)
        with torch.no_grad():
            in_feat = conv(torch.zeros(1, C_i, H, W)).numel()
        self.shared_backbone = nn.Sequential(*layers)
        self.in_feat = in_feat
        self.out_feat = out_feat

        
class ClockCNNMultihead(ClockCNN):
    def __init__(self, H, W, C_i, out_feat, kernel_size, K, D, dropout=0, activation=None, bias=False, batch_norm=False, num_heads=2):
        super().__init__(H, W, C_i, out_feat, kernel_size, K, D, dropout=0, activation=None, bias=False, batch_norm=False)

        output_heads = []
        if isinstance(out_feat, int):
            out_feats = [self.out_feat for _ in range(num_heads)]

        for i in range(num_heads):
            output_heads.append(LinearActivation(self.in_feat, D, activation, dropout=dropout, bias=bias, batch_norm=batch_norm))
            for _ in range(K-1):
                output_heads.append(LinearActivation(D, D, activation, dropout=dropout, bias=bias, batch_norm=batch_norm))
            output_heads.append(nn.Linear(D, out_feats[i]))

        self.output_heads = nn.ModuleList([nn.Sequential(*oh) for oh in output_heads])

    def forward(self, x):
        return tuple(f(self.shared_backbone(x)) for f in self.output_heads)

class CyclicRegressionLoss(nn.Module):
    def forward(self, pred, labels):
        """ 
        Assumes scaled output between 0 and 1.
        """
        return 1 - torch.cos((pred-labels)*2*torch.pi)


class SmoothedTanh(nn.Module):
    def __init__(self, k=1):
        self.k = k
    def forward(self, x):
        return torch.tanh(self.k*x)
    def __str__(self):
        return "SmoothedTanh"

class SmoothedSigmoid(nn.Module):
    def __init__(self, k=0.3):
        self.k = k
    def forward(self, x):
        return torch.sigmoid(self.k*x)
    def __str__(self):
        return "SmoothedSigmoid"

class L2NormActivation(nn.Module):
    def forward(self,x):
        return nn.functional.normalize(x, dim=-1)





