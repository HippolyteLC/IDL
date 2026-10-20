# write a training module

# should include optimizer choice, epochs, initialization, model selection, regularization
import torch as nn
import torch.optim as optim
from modules import init_weights

def train(
        net, 
        init_method, 
        gain,
        loss_function=None, 
        optimizer=None, 
        lr=1e-3, 
        momentum=0.9, 
        n_epochs=1,
    ):

    net.apply(init_weights(init_method=init_method, gain=gain))
    
    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Ada


    criterion = loss_function
    optimizer = optimizer(net.parameters(), lr=lr, momentum=momentum)