from torch.nn.init import xavier_uniform_, xavier_normal_, kaiming_uniform_, normal_
from torch.optim import SGD, Adam, AdamW
from torch.nn import ReLU, GELU, LeakyReLU
import torch
import torch.nn as nn

from modules import CyclicRegressionLoss

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
        return sum(p.pow(2).sum() for p in m.parameters())
    return 0

def get_net_dims(method, dataset_name, num_clock_classes):
    datasets = {
        "clock_75": (75, 75, 1),
        "clock_150": (150, 150, 1),
        "cifar10": (32, 32, 3),
        "fashionmnist": (28, 28, 1),
    }
    if method not in range(1, 8):
        raise ValueError("method must be an integer in [1, 7]")
    if dataset_name not in datasets:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    H, W, C_i = datasets[dataset_name]

    if dataset_name.startswith("clock"):
        out_feat, num_output_heads = {
            1: (num_clock_classes, 1),
            2: (1, 1),
            3: ([12, 60], 2),
            4: ([12, 1], 2),
            5: ([1, 60], 2),
            6: (1, 2),
            7: (1, 2),
        }[method]
    else:
        out_feat, num_output_heads = 10, 1

    return H, W, out_feat, C_i, num_output_heads

def get_net_dims(method, dataset_name, num_clock_classes):
    if not method:
        raise ValueError("Provide a valid method integer: [1,2,...,7]")
    
    if dataset_name in ("clock_75", "clock_150"):
        if method == 1:
            out_feat = num_clock_classes
            num_output_heads = 1
        elif method == 2:
            out_feat = 1
            num_output_heads = 1
        else:
            num_output_heads = 2 
        if method == 3:
            out_feat = [12, 60]
        elif method == 4:
            out_feat = [12, 1]
        elif method == 5:
            out_feat = [1, 60]
        elif method in (6, 7):
            out_feat = 1
        if dataset_name == "clock_75":
            H, W = 75, 75
        elif dataset_name == "clock_150":
            H, W = 150, 150
        C_i = 1
    elif dataset_name == "cifar10":
        H, W = 32, 32
        out_feat = 10
        C_i = 3
        num_output_heads = 1
    elif dataset_name == "fashionmnist":
        H, W = 28, 28 
        out_feat = 10
        C_i = 1
        num_output_heads = 1

    return H, W, out_feat, C_i, num_output_heads

def clock_accuracy(pred, label):
    """
    Common sense accuracy.
    input: pred, label. outputs the shortest distance from pred to label considering
    the full cycle of 720 minutes. 
    I.e. error is in minutes that the prediction is off by. 
    """
    pred_in_mins = pred[:,0]*60 + pred[:,1]
    label_in_mins = label[:,0]*60 + label[:,1]
    diff = torch.abs(pred_in_mins - label_in_mins)
    return torch.minimum(diff, 720 - diff) 

