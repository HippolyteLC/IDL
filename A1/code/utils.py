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
        return sum(p.pow(2).sum() for p in m.parameters())
    return 0


def convert_label(method: int, out_feat):
    """
    method=0: classification,
        convert labels to distinct bins based on num of output_feat 
    method=1: regression,
        convert labels to scalars. Given that we have 60 mins max in the labels,
        we simply need 10^n values, given by n decimal places. in this case n=2. 
    """
    if method == 0:
        def convert_to_bins(label):
            total_mins = 12*60
            assert out_feat <= total_mins, "maximum 720 minutes possible"
            mins_per_bin = total_mins/out_feat
            assert mins_per_bin%1 == 0, "provide valid number of output features"
            h, m = label 
            label_in_mins = h*60 + m 
            return round(label_in_mins/mins_per_bin) * mins_per_bin 
        return convert_to_bins
    elif method == 1: 
        def convert_to_two_decimals(label):
            h, m = label 
            label_in_mins = h*60 + m 
            
            return
        return convert_to_two_decimals
