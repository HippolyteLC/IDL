from torch.nn.init import xavier_uniform_, xavier_normal_, kaiming_uniform_, normal_
from torch.optim import SGD, Adam, AdamW
from torch.nn import ReLU, GELU, LeakyReLU, Module
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

        return H, W, out_feat, C_i, num_output_heads
    return None

def get_label_converter_fn(method: int, out_feat):
    """
    method=0: classification,
        convert labels to distinct bins based on num of output_feat 
    method=1: regression,
        convert labels to scalars. Given that we have 60 mins max in the labels,
        we simply need 10^n values, given by n decimal places. in this case n=2. 
    """
    def same_label(label):
        return label
    def convert_single_to_scalar(label):
        return torch.where(label > 11, torch.round(label / 60.0 * 100) / 100, torch.round(label / 12.0 * 100) / 100)
    if method == 1:
        def convert_to_bins(label):
            total_mins = 12*60
            assert out_feat <= total_mins, "maximum 720 minutes possible"
            mins_per_bin = total_mins/out_feat
            assert mins_per_bin%1 == 0, "provide valid number of output features"
            h, m = label[:,0], label[:,1] 
            label_in_mins = h*60 + m 
            return (torch.round(label_in_mins/mins_per_bin) * mins_per_bin).to(torch.int64) 
        return convert_to_bins
    elif method == 2: 
        def convert_to_two_decimals(label):
            h, m = label[:,0], label[:,1] 
            label_in_mins = h*60 + m 
            return torch.tensor(torch.round((label_in_mins/720)*100) / 100)
        return convert_to_two_decimals
    elif method == 3:
        return same_label
    elif method == 4: 
        def cls_reg(label):
            h, m = label[:,0], label[:,1] 
            return torch.tensor(h).to(torch.int64) , torch.tensor(convert_single_to_scalar(m))
        return cls_reg
    elif method == 5: 
        def reg_cls(label):
            h, m = label[:,0], label[:,1] 
            return torch.tensor(convert_single_to_scalar(h)).to(torch.int64), torch.tensor(m).to(torch.int64) 
        return reg_cls
    elif method == 6:
        def reg_reg(label):
            h, m = label[:,0], label[:,1] 
            return torch.tensor(convert_single_to_scalar(h)), torch.tensor(convert_single_to_scalar(m))
        return reg_reg
    elif method == 7:
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
    return same_label

def get_loss_fc(method: int):
    if method == 1:
        return nn.CrossEntropyLoss()
    elif method == 2: 
        return CyclicRegressionLoss()
    elif method == 3: 
        return nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
    elif method == 4:
        return nn.CrossEntropyLoss(), CyclicRegressionLoss()
    elif method == 5:
        return CyclicRegressionLoss(), nn.CrossEntropyLoss
    elif method == 6:
        return CyclicRegressionLoss(), CyclicRegressionLoss()
    elif method == 7:
        return nn.CosineSimilarity(), nn.CosineSimilarity()
    return nn.CrossEntropyLoss()

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

def output_to_label(method):
    def scalar_to_lab(output):
        r = output * 720
        h = torch.floor(r)
        m = ((r - h) * 60).to(torch.int32)
        return torch.stack([h, m])

    def scal_min_to_int(output):
        m = (output * 60).to(torch.int32)
        return m
    def scal_h_to_int(output):
        h = (output * 12).to(torch.int32)
        return h
        
    if method == 2:
        return scalar_to_lab
    elif method == 3:
        def split_to_whole(output):
            out_1, out_2 = output[:,0], output[:,1]
            return torch.stack([out_1, out_2])
        return split_to_whole
    elif method == 4:
        def cls_reg(output):
            out_1, out_2 = output[:,0], output[:,1]
            out = torch.stack([out_1, scal_min_to_int(out_2)])
            return out
        return cls_reg
    elif method == 5:
        def reg_cls(output):
            out_1, out_2 = output[:,0], output[:,1]
            out = torch.stack([scal_h_to_int(out_1), out_2])
            return out
        return reg_cls
    elif method == 6:
        def reg_reg(output):
            out_1, out_2 = output[:,0], output[:,1]
            out = torch.stack([scal_h_to_int(out_1), scal_min_to_int(out_2)])
            return out
        return reg_reg
    elif method == 7:
        def trig_to_angle(output):
            y, x = output[:,0], output[:,1]
            return torch.atan2(y,x)
        def trig_to_lab(output):
            out_1, out_2 = output[:,0], output[:,1]
            angle_h, angle_m = trig_to_angle(out_1), trig_to_angle(out_2)
            m = 60 * angle_m / (2 * torch.pi)
            h = 12 * angle_h / (2 * torch.pi) - (m/60)
            return torch.stack([h, m])
        return trig_to_lab
    def return_same(output):
        return output
    return return_same
        