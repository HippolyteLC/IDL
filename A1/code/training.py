# write a training module

# should include optimizer choice, epochs, initialization, model selection, regularization
import argparse

import torch as nn
import torch.optim as optim
from modules import init_weights, DynamicMLP, DynamicCNN
from data_loader import DataHandler
from torch.nn.init import xavier_uniform_, xavier_normal_, kaiming_uniform_, normal_

DATA_DIR = "../data"
DATASET_NAME = "fashionmnist"
BATCH_SIZE = 64 

NET_TYPE = "mlp"

K = 3
D = 512
kernel_size = 3


datahandler = DataHandler(data_dir=DATA_DIR, dataset_name=DATASET_NAME, batch_size=BATCH_SIZE)
train_loader, val_loader, test_loader = datahandler.data_loader()

init_method = xavier_uniform_

if DATASET_NAME == "fashionmnist":
    H, W = 28, 28 
    out_feat = 10
    C_i = 1
elif DATASET_NAME == "cifar10":
    H, W = 32, 32
    out_feat = 10
    C_i = 3

if NET_TYPE.lower() == "mlp":
    net = DynamicMLP(H=H, W=W, out_feat=out_feat, K=K, D=D)
elif NET_TYPE.lower() == "cnn":
    net = DynamicCNN(H=H, W=W, C_i=C_i, out_feat=out_feat, kernel_size=kernel_size, K=K, D=D)

def train(
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

    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        print('Finished Training')
        