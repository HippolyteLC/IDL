import argparse

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD, Adam, AdamW
from modules import init_weights, DynamicMLP, DynamicCNN
from data_loader import DataHandler

from utils import load_init_method, load_optimizer_method, get_activation_method, get_regularization_term

import numpy as np

from datetime import datetime 

from functools import partial

import json

import os

DATA_DIR = "../data"
DATASET_NAME = "fashionmnist"
BATCH_SIZE = 1024 

LOGS_DIR = Path(__file__).parent.parent / "logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

CHECKPOINTS_DIR = Path(__file__).parent.parent / "checkpoints"
if not os.path.exists(CHECKPOINTS_DIR):
    os.makedirs(CHECKPOINTS_DIR)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
train_logs_path = LOGS_DIR / f"{timestamp}_training_logs.jsonl"
hp_logs_path = LOGS_DIR / f"{timestamp}_training_hyperparameters.json"

lambda_reg=0.01 # for loss regularization terms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def parse_args() -> argparse.Namespace:
    """Parse the reproducible experiment command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    # general reproducibility / 
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--num-epochs", type=int, default=100)

    # optimizer + lr 
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.3e-2)

    # weights init, activation function, and regularizer methods
    parser.add_argument("--init-method", type=str, default="kaiming_uniform")
    parser.add_argument("--activation-method", type=str, default="leakyrelu")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--regularization-term", type=str, default="", help="Options include 'l2', 'l1', etc.")
    parser.add_argument("--weight-decay", type=float, default=0)

    # architecture 
    parser.add_argument("--n-hidden-fc-layers", type=int, default=8)
    parser.add_argument("--fc-width", type=int, default=128)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--kernel-size", type=int, default=3)
    
    return parser.parse_args()

   
def train(data_loader, optimizer, net, loss_fn, regularization_term):
    size = len(data_loader.dataset)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    for batch, (X, y) in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        X, y = X.to(device), y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(X)
        loss = loss_fn(outputs, y)
        reg_term = get_regularization_term(regularization_term, net) # compute l2 (or other) regularization terms
        loss_reg = loss + lambda_reg*reg_term

        loss_reg.backward()
        optimizer.step()

        # Gather data and report
        total_loss += loss.detach() * X.size(0)
        total_correct += (outputs.argmax(1) == y).sum()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    total_loss = total_loss.item()/size
    total_correct = total_correct.item()/size

    print(f"Train Error: \n Accuracy: {(100*total_correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")

    return total_loss, total_correct

def val(data_loader, net, loss_fn):
    size = len(data_loader.dataset)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            total_loss += loss_fn(outputs, y) * X.size(0)
            total_correct += (outputs.argmax(1) == y).sum() 
    total_loss = total_loss / size
    total_correct = total_correct / size
    print(f"Validation Error: \n Accuracy: {(100*total_correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")
    return total_loss, total_correct

def test(data_loader, net, loss_fn):
    size = len(data_loader.dataset)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            total_loss += loss_fn(outputs, y) * X.size(0)
            total_correct += (outputs.argmax(1) == y).sum() 
    total_loss/=size
    total_correct/=size
    print(f"Test Error: \n Accuracy: {(100*total_correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")
    return total_loss, total_correct

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    datahandler = DataHandler(data_dir=args.data_dir, dataset_name=args.dataset, device=device, batch_size=args.batch_size, seed=args.split_seed)
    train_loader, val_loader, test_loader = datahandler.data_loader()

    if args.dataset.lower() == "fashionmnist":
        H, W = 28, 28 
        out_feat = 10
        C_i = 1
    elif args.dataset.lower() == "cifar10":
        H, W = 32, 32
        out_feat = 10
        C_i = 3
    else:
        raise NotImplementedError("Define a valid dataset!")

    K = args.n_hidden_fc_layers
    D = args.fc_width
    kernel_size = args.kernel_size

    activation = get_activation_method(args.activation_method)

    if args.model.lower() == "mlp":
        net = DynamicMLP(H=H, W=W, C_i=C_i, out_feat=out_feat, K=K, D=D, activation=activation, dropout=args.dropout, bias=args.bias)
    elif args.model.lower() == "cnn":
        net = DynamicCNN(H=H, W=W, C_i=C_i, out_feat=out_feat, kernel_size=kernel_size, K=K, D=D, activation=activation, dropout=args.dropout, bias=args.bias)

    net.apply(partial(init_weights,
                    init_method=load_init_method(args.init_method),
                    ))
    net.to(device)

    optimizer_method = load_optimizer_method(args.optimizer)
    optimizer = optimizer_method(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    loss_function = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        print(f'EPOCH {epoch + 1}:')

        # Make sure gradient tracking is on, and do a pass over the data
        net.train(True)
        train_loss, train_correct = train(train_loader, optimizer, net, loss_function, args.regularization_term)

        net.eval()
        vloss, vcorrect = val(val_loader, net, loss_function)
        record = {}
        record["train_loss"] = train_loss
        record["train_accuracy"] = train_correct
        record["validation_loss"] = vloss
        record["validation_accuracy"] = vcorrect
        record = {k: float(v) for k, v in record.items()}
        with open(train_logs_path, 'a') as f:
            f.write(json.dumps(record) + "\n")

    hp_logs = vars(args).copy()
    del hp_logs["data_dir"]
    hp_logs["last_train_accuracy"] = record["train_accuracy"]
    hp_logs["last_val_accuracy"] = record["validation_accuracy"]

    with open(hp_logs_path, 'w') as f:
        json.dump(hp_logs, f, indent=4)
    
    model_path = CHECKPOINTS_DIR / f"model_{args.model}_{args.dataset}_{timestamp}.pth"

    torch.save(net.state_dict(), model_path)
    print(f"Saved PyTorch Model State to {model_path}")
    
if __name__=="__main__":
    main()