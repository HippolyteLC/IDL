from torch.nn.init import xavier_uniform_, xavier_normal_, kaiming_uniform_, normal_
from torch.optim import SGD, Adam, AdamW
from torch.nn import ReLU, GELU, LeakyReLU, Module
import torch
import torch.nn as nn

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


def get_label_converter_fn(method: int, out_feat):
    """
    method=0: classification,
        convert labels to distinct bins based on num of output_feat 
    method=1: regression,
        convert labels to scalars. Given that we have 60 mins max in the labels,
        we simply need 10^n values, given by n decimal places. in this case n=2. 
    """
    def convert_single_to_scalar(label):
        return torch.where(label > 11, torch.round(label / 60.0 * 100) / 100, torch.round(label / 12.0 * 100) / 100)
    if method == 0:
        def convert_to_bins(label):
            total_mins = 12*60
            assert out_feat <= total_mins, "maximum 720 minutes possible"
            mins_per_bin = total_mins/out_feat
            assert mins_per_bin%1 == 0, "provide valid number of output features"
            h, m = label[:,0], label[:,1] 
            label_in_mins = h*60 + m 
            return torch.round(label_in_mins/mins_per_bin) * mins_per_bin 
        return convert_to_bins
    elif method == 1: 
        def convert_to_two_decimals(label):
            h, m = label[:,0], label[:,1] 
            label_in_mins = h*60 + m 
            return torch.round((label_in_mins/720)*100) / 100
        return convert_to_two_decimals
    elif method == 2:
        def split_label(label):
            return label[:,0], label[:,1]
        return split_label
    elif method == 3: 
        def cls_reg(label):
            h, m = label[:,0], label[:,1] 
            return h, convert_single_to_scalar(m)
        return cls_reg
    elif method == 4: 
        def reg_cls(label):
            h, m = label[:,0], label[:,1] 
            return convert_single_to_scalar(h), m
        return reg_cls
    elif method == 5:
        def reg_reg(label):
            h, m = label[:,0], label[:,1] 
            return convert_single_to_scalar(h), convert_single_to_scalar(m)
        return reg_reg
    elif method == 6:
        def trig(label):
            hours = label[:,0]
            mins = label[:,1]
            angle_hour = 2 * torch.pi * (hours+mins/60)/12
            angle_min = 2 * torch.pi * mins/60
            return torch.stack([
                torch.sin(angle_hour), 
                torch.cos(angle_hour),
                ], dim=1), \
                torch.stack([ 
                torch.sin(angle_min),
                torch.cos(angle_min)
                ], dim=1)
        return trig
    else:
        def same_label(label):
            return label
        return same_label

def get_loss_fc(method: int, ):
    if method == 0:
        return nn.CrossEntropyLoss
    elif method == 1: 

def clock_accuracy(pred, label):
    """
    Common sense accuracy.
    input: pred, label. outputs the shortest distance from pred to label considering
    the full cycle of 720 minutes. 
    """
    pred_in_mins = pred[:,0]*60 + pred[:,1]
    label_in_mins = label[:,0]*60 + label[:,1]
    diff = torch.abs(pred_in_mins - label_in_mins)
    return torch.minimum(diff, 720 - diff) 
