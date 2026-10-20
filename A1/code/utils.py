from torch.nn.init import xavier_uniform_, xavier_normal_, kaiming_uniform_, normal_


def load_init_method(method):
    if method == "xavier_normal":
        return xavier_normal_
    elif method == "xavier_uniform":
        return xavier_uniform_
    elif method == "kaiming_uniform":
        return kaiming_uniform_
    elif method == "normal":
        return normal_

def load_optimizer_method(method):
    pass