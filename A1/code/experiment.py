import argparse

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD, Adam, AdamW
from modules import init_weights, DynamicMLP, DynamicCNN, ClockCNNMultihead, SmoothedSigmoid, SmoothedTanh
from modules import L2NormActivation
from data_loader import DataHandler

from utils import load_init_method, load_optimizer_method, get_activation_method, get_regularization_term
from utils import get_net_dims, get_loss_fc, clock_accuracy, get_label_converter_fn, output_to_label
import numpy as np

from datetime import datetime 

from functools import partial

import json

import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def parse_args() -> argparse.Namespace:
    """Parse the reproducible experiment command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    # general reproducibility / 
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--dataset", default="fashionmnist")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--model", default="cnn")
    parser.add_argument("--num-epochs", type=int, default=50)

    # clock time predictor params
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--clock-method", type=int, default=1)
    parser.add_argument("--num-clock-classes", type=int, default=720)

    # optimizer + lr 
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    # weights init, activation function, and regularizer methods
    parser.add_argument("--init-method", type=str, default="kaiming_uniform")
    parser.add_argument("--activation-method", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--regularization-term", type=str, default="", help="Options include 'l2', 'l1', etc.")
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--batch-norm", action="store_true")

    # architecture 
    parser.add_argument("--n-hidden-fc-layers", type=int, default=2)
    parser.add_argument("--fc-width", type=int, default=256)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--kernel-size", type=int, default=7)
    
    return parser.parse_args()

   
def main_train(data_loader, optimizer, net, loss_fn, regularization_term):
    size = len(data_loader.dataset)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    for _, (X, y) in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        X, y = X.to(device), y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(X)
        loss = loss_fn(outputs, y)
        reg_term = get_regularization_term(regularization_term, net) # compute l2 (or other) regularization terms
        lambda_reg=0.01 # for loss regularization terms
        loss_reg = loss + lambda_reg*reg_term

        loss_reg.backward()
        optimizer.step()

        total_loss += loss.detach() * X.size(0)
        total_correct += (outputs.argmax(1) == y).sum()
    
    total_loss = total_loss.item()/size
    total_correct = total_correct.item()/size

    print(f"Train Error: \n Accuracy: {(100*total_correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")

    return total_loss, total_correct

def clock_train(data_loader, optimizer, net, loss_fn, cm, num_output_heads, output_converter, label_converter):
    """
    cm: is the clock method 
    """
    if isinstance(loss_fn, tuple) and num_output_heads == 2:
        loss_fn_1, loss_fn_2 = loss_fn
    size = len(data_loader.dataset)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    for _, (X, y) in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        X, y = X.to(device), y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        if num_output_heads == 1:
            outputs = net(X)
            labels = label_converter(y)
            loss = loss_fn(outputs, labels)
            loss.backward()
        elif num_output_heads == 2:
            outputs = net(X)
            output_1, output_2 = outputs
            lab_1, lab_2 = label_converter(y)       
            loss_1, loss_2 = loss_fn_1(output_1, lab_1), loss_fn_2(output_2, lab_2)
            loss = loss_1 + loss_2

        loss.backward()
        optimizer.step()

        total_loss += loss.detach() * X.size(0)
        if cm is None or cm == 1:
            total_correct += (outputs.argmax(1) == y).sum()
        elif cm in (2,3,4,5,6):
            outputs_correct_format = output_converter(outputs)
            error = clock_accuracy(outputs_correct_format, y)
            total_correct += error.sum()
        
    total_loss = total_loss.item()/size
    total_correct = total_correct.item()/size

    print(f"Train Error: \n Avg mins off: {(total_correct):>0.1f}, Avg loss: {total_loss:>8f} \n")

    return total_loss, total_correct    

def clock_val(data_loader, net, loss_fn, cm, num_output_heads, output_converter, label_converter):
    """
    cm: is the clock method 
    """
    if isinstance(loss_fn, tuple) and num_output_heads == 2:
        loss_fn_1, loss_fn_2 = loss_fn
    size = len(data_loader.dataset)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    with torch.no_grad():
        for _, (X, y) in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            X, y = X.to(device), y.to(device)

            # forward + backward + optimize
            if num_output_heads == 1:
                outputs = net(X)
                labels = label_converter(y)
                loss = loss_fn(outputs, labels)
                loss.backward()
            elif num_output_heads == 2:
                outputs = net(X)
                output_1, output_2 = outputs
                lab_1, lab_2 = label_converter(y)       
                loss_1, loss_2 = loss_fn_1(output_1, lab_1), loss_fn_2(output_2, lab_2)
                loss = loss_1 + loss_2

            total_loss += loss.detach() * X.size(0)
            if cm is None or cm == 1:
                total_correct += (outputs.argmax(1) == y).sum()
            elif cm in (2,3,4,5,6):
                outputs_correct_format = output_converter(outputs)
                error = clock_accuracy(outputs_correct_format, y)
                total_correct += error.sum()
        
    total_loss = total_loss.item()/size
    total_correct = total_correct.item()/size

    print(f"Val Error: \n Avg mins off: {(total_correct):>0.1f}, Avg loss: {total_loss:>8f} \n")

    return total_loss, total_correct    

def clock_test(data_loader, net, loss_fn, cm, num_output_heads, output_converter, label_converter):
    """
    cm: is the clock method 
    """
    if isinstance(loss_fn, tuple) and num_output_heads == 2:
        loss_fn_1, loss_fn_2 = loss_fn
    size = len(data_loader.dataset)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    with torch.no_grad():
        for _, (X, y) in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            X, y = X.to(device), y.to(device)

            # forward + backward + optimize
            if num_output_heads == 1:
                outputs = net(X)
                labels = label_converter(y)
                loss = loss_fn(outputs, labels)
                loss.backward()
            elif num_output_heads == 2:
                outputs = net(X)
                output_1, output_2 = outputs
                lab_1, lab_2 = label_converter(y)       
                loss_1, loss_2 = loss_fn_1(output_1, lab_1), loss_fn_2(output_2, lab_2)
                loss = loss_1 + loss_2

            total_loss += loss.detach() * X.size(0)
            if cm is None or cm == 1:
                total_correct += (outputs.argmax(1) == y).sum()
            elif cm in (2,3,4,5,6):
                outputs_correct_format = output_converter(outputs)
                error = clock_accuracy(outputs_correct_format, y)
                total_correct += error.sum()
        
    total_loss = total_loss.item()/size
    total_correct = total_correct.item()/size

    print(f"Test Error: \n Avg mins off: {(total_correct):>0.1f}, Avg loss: {total_loss:>8f} \n")

    return total_loss, total_correct    

def main_val(data_loader, net, loss_fn):
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

def main_test(data_loader, net, loss_fn):
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

    cm = args.clock_method
    task = 1  
    if cm:
        task = 2

    LOGS_DIR = Path(__file__).parent.parent / f"logs/task_{task}/{args.model}"
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    CHECKPOINTS_DIR = Path(__file__).parent.parent / f"checkpoints/task_{task}/{args.model}"
    if not os.path.exists(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_logs_path = LOGS_DIR / f"{timestamp}_training_logs.jsonl"
    hp_logs_path = LOGS_DIR / f"{timestamp}_training_hyperparameters.json"

    torch.manual_seed(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    dataset_name = args.dataset.lower()
    datahandler = DataHandler(dataset_name=dataset_name, device=device, batch_size=args.batch_size, seed=args.split_seed)
    train_loader, val_loader, test_loader = datahandler.data_loader()
    
    if dataset_name == "fashionmnist":
        H, W = 28, 28 
        out_feat = 10
        C_i = 1
    elif dataset_name == "cifar10":
        H, W = 32, 32
        out_feat = 10
        C_i = 3
    elif dataset_name in ("clock_75", "clock_150"):
        H, W, out_feat, C_i, num_output_heads = get_net_dims(
            cm, dataset_name, args.num_clock_classes
        )

    K = args.n_hidden_fc_layers
    D = args.fc_width
    kernel_size = args.kernel_size

    output_converter = output_to_label(method=cm)

    activation = get_activation_method(args.activation_method)

    if args.model.lower() == "mlp":
        net = DynamicMLP(H=H, W=W, C_i=C_i, out_feat=out_feat, K=K, D=D, activation=activation, dropout=args.dropout, bias=args.bias, batch_norm=args.batch_norm)
    elif args.model.lower() == "cnn":
        net = DynamicCNN(H=H, W=W, C_i=C_i, out_feat=out_feat, kernel_size=kernel_size, K=K, D=D, activation=activation, dropout=args.dropout, bias=args.bias, batch_norm=args.batch_norm)
    elif args.model.lower() == "clock_cnn":
        net = ClockCNNMultihead(H=H, W=W, C_i=C_i, out_feat=out_feat, kernel_size=kernel_size, K=K, D=D, activation=activation, dropout=args.dropout, bias=args.bias, batch_norm=args.batch_norm, num_heads=num_output_heads)
        if cm == 2:
            for oh in net.output_heads:
                oh.append(SmoothedSigmoid())
        elif cm in (4, 5, 6): 
            for oh in net.output_heads:
                if hasattr(oh[-1], 'out_features') and oh[-1].out_features == 1:
                    oh.append(SmoothedSigmoid())
        elif cm == 7:
            for oh in net.output_heads:
                oh.append(L2NormActivation())

    net.apply(partial(init_weights,
                    init_method=load_init_method(args.init_method),
                    ))
    net.to(device)

    optimizer_method = load_optimizer_method(args.optimizer)
    optimizer = optimizer_method(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if not cm:
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = get_loss_fc(cm)
        
    label_converter = get_label_converter_fn(cm)
        
    for epoch in range(args.num_epochs):
        print(f'EPOCH {epoch + 1}:')

        if cm:
            # Make sure gradient tracking is on, and do a pass over the data
            net.train(True)
            train_loss, train_correct = clock_train(train_loader, optimizer, net, loss_function, \
                                                    args.regularization_term, cm, num_output_heads,\
                                                    output_converter, label_converter)
            vloss, vcorrect = clock_val(val_loader, net, loss_function, cm, num_output_heads,\
                                                    output_converter, label_converter)
        else:
            net.eval()
            train_loss, train_correct = main_train(train_loader, optimizer, net, loss_function, args.regularization_term)
            vloss, vcorrect = main_val(val_loader, net, loss_function)
        
        record = {}
        record["train_loss"] = train_loss
        record["train_accuracy"] = train_correct
        record["validation_loss"] = vloss
        record["validation_accuracy"] = vcorrect
        record = {k: float(v) for k, v in record.items()}
        with open(train_logs_path, 'a') as f:
            f.write(json.dumps(record) + "\n")

    hp_logs = vars(args).copy()
    hp_logs["last_train_accuracy"] = record["train_accuracy"]
    hp_logs["last_val_accuracy"] = record["validation_accuracy"]

    with open(hp_logs_path, 'w') as f:
        json.dump(hp_logs, f, indent=4)

    if args.save:
        model_path = CHECKPOINTS_DIR / f"model_{args.model}_{args.dataset}_{timestamp}.pth"
        
        torch.save(net.state_dict(), model_path)
        print(f"Saved PyTorch Model State to {model_path}")
    
if __name__=="__main__":
    main()