from torch.nn.init import xavier_uniform_, xavier_normal_, kaiming_uniform_, normal_
from torch.optim import SGD, Adam, AdamW
from torch.nn import ReLU, GELU, LeakyReLU
import torch

from modules import DynamicCNN, DynamicMLP, ClockCNNMultihead

from activations import SmoothedTanh, SmoothedSigmoid, L2NormActivation

def load_init_method(init_method):
    method = init_method.lower()
    if method == "xavier_normal":
        return xavier_normal_
    elif method == "xavier_uniform":
        return xavier_uniform_
    elif method == "kaiming_uniform":
        return kaiming_uniform_
    elif method == "normal":
        return normal_
    return None

def load_optimizer_method(optim_method):
    method = optim_method.lower()
    if method == "sgd":
        return SGD
    elif method == "adam":
        return Adam
    elif method == "adamw":
        return AdamW
    return None

def get_activation_method(act_method):
    method = act_method.lower()
    if method == "relu":
        return ReLU
    elif method == "gelu":
        return GELU
    elif method == "leakyrelu":
        return LeakyReLU
    return None

def get_regularization_term(reg_method, m):
    method = reg_method.lower()
    if method == "l2":
        return sum(p.pow(2).sum() for p in m.parameters())
    return 0

def get_net_dims(clock_method, dataset_name, num_clock_classes):
    datasets = {
        "clock_75": (75, 75, 1),
        "clock_150": (150, 150, 1),
        "cifar10": (32, 32, 3),
        "fashionmnist": (28, 28, 1),
    }
    if clock_method not in range(1, 8):
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
        }[clock_method]
    else:
        out_feat, num_output_heads = 10, 1

    return H, W, out_feat, C_i, num_output_heads


def get_net(model_config):
    model = model_config.model
    if model == "mlp":
        net = DynamicMLP(model_config)
    elif model == "cnn":
        net = DynamicCNN(model_config)
    elif model == "clock_cnn":
        net = ClockCNNMultihead(model_config)
        cm = model_config.clock_method
        if cm == 2:
            for oh in net.output_heads:
                oh.append(SmoothedSigmoid())
        elif cm in (4, 5, 6): 
            for oh in net.output_heads:
                if hasattr(oh[-1], 'out_features') and oh[-1].out_features == 1:
                    oh.append(SmoothedSigmoid())
        elif cm == 7:
            for oh in net.output_heads:
                oh.append(L2NormActivation())
    return net