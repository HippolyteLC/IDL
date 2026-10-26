import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch

from utils import get_activation_method, load_init_method

from activations import SmoothedTanh, SmoothedSigmoid, L2NormActivation

from functools import partial



class ModelLoader:
    @staticmethod
    def get_net(model_config):
        model = model_config.model

        if model == "mlp":
            net = DynamicMLP(model_config)
        elif model == "cnn":
            net = DynamicCNN(model_config)
        elif model == "clock_cnn":
            net = ClockCNNMultihead(model_config)
            cm = model_config.clock_method

            for oh in net.output_heads:
                if cm == 2:
                    oh.append(SmoothedSigmoid())
                elif cm in (4, 5, 6) and getattr(oh[-1], "out_features", None) == 1:
                    oh.append(SmoothedSigmoid())
                elif cm == 7:
                    oh.append(L2NormActivation())
        
        net.apply(partial(init_weights,
                init_method=load_init_method(model_config.init_method),
                ))

        return net

        # raise ValueError(f"Unknown model: {model}")

def init_weights(m, init_method, verbose=False):
    if verbose:
        print(m)
    if type(m) == nn.Linear:
        init_method(m.weight)
        if verbose:
            print(m.weight) 
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) == nn.Conv2d:
        init_method(m.weight)
        if verbose:
            print(m.weight) 
        if m.bias is not None:
            nn.init.zeros_(m.bias)

           
class LinearActivation(nn.Module):
    def __init__(self, in_feat, out_feat, activation, dropout, bias, batch_norm):
        super().__init__()

        layers = [nn.Linear(in_feat, out_feat, bias=bias)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_feat))
        layers.append(activation())
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_relu_stack(nn.Flatten()(x))


class Conv2DActivationPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, dropout, bias, batch_norm):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding="same", bias=bias)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend([activation(), nn.MaxPool2d(2, 2, ceil_mode=True)])
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
    def __init__(self, model_config):
        super().__init__()
        K = model_config.n_hidden_fc_layers
        D = model_config.fc_width
        activation = get_activation_method(model_config.activation_method)
        batch_norm = model_config.batch_norm
        bias = model_config.bias
        dropout = model_config.dropout

        in_feat = model_config.H*model_config.W*model_config.C_i
        layers = [LinearActivation(in_feat, D, activation, dropout=dropout, bias=bias, batch_norm=batch_norm)]
        for _ in range(K-1):
            layers.append(LinearActivation(D, D, activation, dropout=dropout, bias=bias, batch_norm=batch_norm))
        layers.append(nn.Linear(D, model_config.out_feat))
        
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
    def __init__(self, model_config):
        super().__init__()
        K = model_config.n_hidden_fc_layers
        D = model_config.fc_width
        kernel_size = model_config.kernel_size
        activation = get_activation_method(model_config.activation_method)
        batch_norm = model_config.batch_norm
        bias = model_config.bias
        dropout = model_config.dropout
        H, W, C_i = model_config.H, model_config.W, model_config.C_i
        out_feat = model_config.out_feat
    
        layers = self.build_backbone(C_i, H, kernel_size, activation, dropout, bias, batch_norm)
        conv = nn.Sequential(*layers)
        with torch.no_grad():
            in_feat = conv(torch.zeros(1, C_i, H, W)).numel()
        layers.append(LinearActivation(in_feat, D, activation, dropout=dropout, bias=bias, batch_norm=batch_norm))
        for _ in range(K-1):
            layers.append(LinearActivation(D, D, activation, dropout=dropout, bias=bias, batch_norm=batch_norm))
        layers.append(nn.Linear(D, out_feat))

        self.cnn = nn.Sequential(*layers)

        self.width = D

    @staticmethod
    def build_backbone(C_i, H, kernel_size, activation, dropout, bias, batch_norm):
        n_pools = int(max(0, np.ceil(np.log2(H/(1<<2)))))
        layers = [Conv2DActivationPool(C_i, H, kernel_size, activation, dropout=dropout, bias=bias, batch_norm=batch_norm)]
        C_o = H*1
        for _ in range(n_pools-1):
            layers.append(Conv2DActivationPool(C_o, C_o*2, kernel_size, activation, dropout=dropout, bias=bias, batch_norm=batch_norm))
            C_o *= 2
        return layers

    def forward(self, x):
        return self.cnn(x)


class ClockCNN(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        kernel_size = model_config.kernel_size
        activation = get_activation_method(model_config.activation_method)
        batch_norm = model_config.batch_norm
        bias = model_config.bias
        dropout = model_config.dropout
        H, W, C_i = model_config.H, model_config.W, model_config.C_i

        layers = DynamicCNN.build_backbone(C_i, H, kernel_size, activation, dropout, bias, batch_norm)

        conv = nn.Sequential(*layers)
        with torch.no_grad():
            in_feat = conv(torch.zeros(1, C_i, H, W)).numel()
        self.in_feat = in_feat

        self.shared_backbone = nn.Sequential(*layers)

        
class ClockCNNMultihead(ClockCNN):
    def __init__(self, model_config):
        super().__init__(model_config)

        K = model_config.n_hidden_fc_layers
        D = model_config.fc_width
        activation = get_activation_method(model_config.activation_method)
        batch_norm = model_config.batch_norm
        bias = model_config.bias
        dropout = model_config.dropout
        out_feat = model_config.out_feat
        num_heads = model_config.num_output_heads

        output_heads = []
        if isinstance(out_feat, int):
            out_feats = [out_feat for _ in range(num_heads)]

        for i in range(num_heads):
            output_head = []
            output_head.append(LinearActivation(self.in_feat, D, activation, dropout=dropout, bias=bias, batch_norm=batch_norm))
            for _ in range(K-1):
                output_head.append(LinearActivation(D, D, activation, dropout=dropout, bias=bias, batch_norm=batch_norm))
            output_head.append(nn.Linear(D, out_feats[i]))
            output_heads.append(nn.Sequential(*output_head))

        self.output_heads = nn.ModuleList(output_heads)

    def forward(self, x):
        r = tuple(f(self.shared_backbone(x)) for f in self.output_heads)
        out = r[0] if len(r) == 1 else r
        return out

    