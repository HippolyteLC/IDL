# write a training module

# should include optimizer choice, epochs, initialization, model selection, regularization
import argparse

from Pathlib import Path

import torch as nn
import torch.optim as optim
from torch.optim import SGD, Adam, AdamW
from modules import init_weights, DynamicMLP, DynamicCNN
from data_loader import DataHandler

from utils import load_init_method

import numpy as np

DATA_DIR = "../data"
DATASET_NAME = "fashionmnist"
BATCH_SIZE = 64 

NET_TYPE = "mlp"

def parse_args() -> argparse.Namespace:
    """Parse the reproducible experiment command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="fashionmnist")
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--init-method", type=str, default="kaiming")
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--regularization", type=str, default="dropout")
    parser.add_argument("--n-hidden-fc-layers", type=int, default=5)
    parser.add_argument("--fc-width", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=3)
    return parser.parse_args()

args = parse_args()

if args.dataset.lower() == "fashionmnist":
    H, W = 28, 28 
    out_feat = 10
    C_i = 1
elif args.dataset.lower() == "cifar10":
    H, W = 32, 32
    out_feat = 10
    C_i = 3

K = args.n_hidden_fc_layers
D = args.fc_width
kernel_size = args.kernel_size

if args.model.lower() == "mlp":
    net = DynamicMLP(H=H, W=W, out_feat=out_feat, K=K, D=D)
elif args.model.lower() == "cnn":
    net = DynamicCNN(H=H, W=W, C_i=C_i, out_feat=out_feat, kernel_size=kernel_size, K=K, D=D)

initialization_method = load_init_method(args.init_method)

i
optimizer_method = 

datahandler = DataHandler(data_dir=DATA_DIR, dataset_name=DATASET_NAME, batch_size=BATCH_SIZE)
train_loader, val_loader, test_loader = datahandler.data_loader()


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

loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD()

def train_one_epoch(
    gain,
    lr=1e-3, 
    momentum=0.9, 
    n_epochs=1,
    ):
    running_loss = 0.
    last_loss = 0.

    net.apply(init_weights(init_method=initialization_method, gain=gain))
    



    criterion = loss_function
    optimizer = optimizer(net.parameters(), lr=lr, momentum=momentum)

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

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0.

        # print('Finished Training')
        
def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
