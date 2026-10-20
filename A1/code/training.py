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

import json

DATA_DIR = "../data"
DATASET_NAME = "fashionmnist"
BATCH_SIZE = 64 

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

LOGS_DIR = Path("../logs")

lambda_reg=0.01 # for loss regularization terms

device = nn.accelerator.current_accelerator().type if nn.accelerator.is_available() else "cpu"
print(f"Using {device} device")

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
optimizer = optimizer_method(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

loss_function = nn.CrossEntropyLoss()

net.apply(init_weights(init_method=initialization_method, generator=generator))
    
def train():
    size = len(train_loader.dataset)
    n_batches = len(val_loader)
    train_loss, correct = 0., 0
    for batch, (X, y) in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        X, y = X.to(device), y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(X)
        loss = loss_function(outputs, y)
        reg_term = get_regularization_term(args.regularization_method, net) # compute l2 (or other) regularization terms
        loss += lambda_reg*reg_term

        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item() * X.size(0)
        train_loss += loss.item() * X.size(0)
        correct += (outputs.argmax(1) == y).type(nn.float).sum().item()
        if batch % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss/=n_batches
    correct/=size

    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")

    return train_loss, correct

def val():
    size = len(val_loader.dataset)
    n_batches = len(val_loader)
    vloss, correct = 0., 0
    with nn.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            vloss += loss_function(outputs, y).item()
            correct += (outputs.argmax(1) == y).type(nn.float).sum().item()
    vloss/=n_batches
    correct/=size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {vloss:>8f} \n")
    return vloss, correct

train_logs_path = LOGS_DIR + f"{timestamp}_training_logs.jsonl"

for epoch in range(args.num_epochs):
    print(f'EPOCH {epoch + 1}:')

    # Make sure gradient tracking is on, and do a pass over the data
    net.train(True)
    train_loss, train_correct = train()

    net.eval()
    vloss, correct = val()
    record = {}
    record["train_loss"] = train_loss
    record["train_accuracy"] = train_correct
    record["validation_loss"] = vloss
    record["validation_accuracy"] = correct
    with open(train_logs_path, 'w') as f:
        f.write(json.dumps(record) + "\n")

hp_logs = vars(args)
del hp_logs["data_dir"]
hp_logs["fc_width"] = net.width

hp_logs_path = LOGS_DIR + f"{timestamp}_training_hyperparameters.json"

with open(hp_logs_path, 'w') as f:
    json.dump(hp_logs_path, indent=4)

model_path = Path(f"../checkpoints/model_{args.model}_{args.dataset}_{timestamp}.pth")
nn.save(net.state_dict(), model_path)
print(f"Saved PyTorch Model State to {model_path}")
