# write a training module

# should include optimizer choice, epochs, initialization, model selection, regularization
import argparse

from Pathlib import Path

import torch as nn
import torch.optim as optim
from torch.optim import SGD, Adam, AdamW
from modules import init_weights, DynamicMLP, DynamicCNN
from data_loader import DataHandler

from utils import load_init_method, load_optimizer_method, get_activation_method, get_regularization_term

import numpy as np

from datetime import datetime 

DATA_DIR = "../data"
DATASET_NAME = "fashionmnist"
BATCH_SIZE = 64 

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

lambda_reg=0.01 # for loss regularization terms

def parse_args() -> argparse.Namespace:
    """Parse the reproducible experiment command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--init-method", type=str, default="kaiming")
    parser.add_argument("--gain", type=float, default=)
    parser.add_argument("--activation-method", type=str, default="relu")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--regularization-term", type=str, default=None, help="Options include 'l2', 'l1', etc.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--n-hidden-fc-layers", type=int, default=5)
    parser.add_argument("--fc-width", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--num-epochs", type=int, default=20)
    return parser.parse_args()

args = parse_args()

generator = nn.Generator()
generator.manual_seed(args.seed)

datahandler = DataHandler(data_dir=args.data_dir, dataset_name=args.dataset, batch_size=args.batch_size, seed=args.split_seed)
train_loader, val_loader, test_loader = datahandler.data_loader()

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

activation = get_activation_method(args.activation_method)
dropout = args.dropout

if args.model.lower() == "mlp":
    net = DynamicMLP(H=H, W=W, out_feat=out_feat, K=K, D=D, activation=activation, dropout=dropout)
elif args.model.lower() == "cnn":
    net = DynamicCNN(H=H, W=W, C_i=C_i, out_feat=out_feat, kernel_size=kernel_size, K=K, D=D, activation=activation,dropout=dropout)



initialization_method = load_init_method(args.init_method)
optimizer_method = load_optimizer_method(args.optimizer)
loss_function = nn.CrossEntropyLoss()


def train_one_epoch(
    ):
    net.apply(init_weights(init_method=initialization_method, generator=generator))

    criterion = loss_function
    optimizer = optimizer_method(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    total_loss = 0.
    last_loss = 0.
    running_loss = 0.0
    total_samples = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        reg_term = get_regularization_term(args.regularization_method, net) # compute l2 (or other) regularization terms
        loss+= lambda_reg*reg_term

        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item() * inputs.size(0)
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        if i % 1000 == 999:
            last_loss = (running_loss) / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0. 

        # print('Finished Training')
    epoch_train_loss = total_loss / total_samples
    return epoch_train_loss

for epoch in range(args.num_epochs):
    print(f'EPOCH {epoch_number + 1}:')

    # Make sure gradient tracking is on, and do a pass over the data
    net.train(True)
    avg_loss = train_one_epoch(epoch_number)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    net.eval()

    # Disable gradient computation and reduce memory consumption.
    with nn.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            voutputs = net(vinputs)
            vloss = loss_function(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print(f'Loss: train {avg_loss} | valid {avg_vloss}')

    # Log the running loss averaged per batch
    # for both training and validation

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f'model_{timestamp}_{epoch_number}'
        nn.save(net.state_dict(), model_path)

    epoch_number += 1
