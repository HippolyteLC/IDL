from torch.nn.init import xavier_uniform_, xavier_normal_, kaiming_uniform_, normal_
from torch.optim import SGD, Adam, AdamW
from torch.nn import ReLU, GELU, LeakyReLU, Module

def load_init_method(method):
    method = method.lower()
    if method == "xavier_normal":
        return xavier_normal_
    elif method == "xavier_uniform":
        return xavier_uniform_
    elif method == "kaiming_uniform":
        return kaiming_uniform_
    elif method == "normal":
        return normal_
    return None

def load_optimizer_method(method):
    method = method.lower()
    if method == "sgd":
        return SGD
    elif method == "adam":
        return Adam
    elif method == "adamw":
        return AdamW
    return None

def get_activation_method(method):
    method = method.lower()
    if method == "relu":
        return ReLU
    elif method == "gelu":
        return GELU
    elif method == "leakyrelu":
        return LeakyReLU
    return None

def get_regularization_term(method, m):
    method = method.lower()
    if method == "l2":
        if isinstance(m, Module):
            return sum(p.pow(2) for p in m.parameters())
    return 0
    