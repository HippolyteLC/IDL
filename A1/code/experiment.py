import argparse

from pathlib import Path

import torch

import torch.nn as nn

from modules import ModelLoader

from loss import Criterion

from train import clock_train, clock_val, clock_test
from train import main_train, main_val, main_test

from data_loader import DataHandler

from utils import load_optim

from config import ExperimentConfig

from datetime import datetime 

import json

import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def parse_args() -> argparse.Namespace:
    """Parse the reproducible experiment command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    # general reproducibility / 
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--dataset-name", default="fashionmnist")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--model", default="cnn")
    parser.add_argument("--num-epochs", type=int, default=50)

    # clock time predictor params
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--clock-method", type=int, default=None)
    parser.add_argument("--num-clock-classes", type=int, default=720)

    # optimizer + lr 
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    # weights init, activation function, and regularizer methods
    parser.add_argument("--init-method", type=str, default="kaiming_uniform")
    parser.add_argument("--activation-method", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--batch-norm", action="store_true")

    # architecture 
    parser.add_argument("--n-hidden-fc-layers", type=int, default=2)
    parser.add_argument("--fc-width", type=int, default=256)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--kernel-size", type=int, default=7)
    
    return parser.parse_args()

def main():
    args = parse_args()
    config = ExperimentConfig.from_argparse(args)

    task = 1 if not config.model.clock_method else 2

    LOGS_DIR = Path(__file__).parent.parent / f"logs/task_{task}/{args.model}"
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    CHECKPOINTS_DIR = Path(__file__).parent.parent / f"checkpoints/task_{task}/{args.model}"
    if not os.path.exists(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_logs_path = LOGS_DIR / f"{timestamp}_training_logs.jsonl"
    hp_logs_path = LOGS_DIR / f"{timestamp}_training_hyperparameters.json"

    datahandler = DataHandler(config.data_config, device)
    train_loader, val_loader, test_loader = datahandler.data_loader()

    torch.manual_seed(config.training.seed)

    net = ModelLoader().get_net(config.model)

    criterion = Criterion(net, config.model)

    net.to(device)

    optimizer = load_optim(net, config.training) 

    for epoch in range(args.num_epochs):
        print(f'EPOCH {epoch + 1}:')

        if config.model.clock_method:
            # Make sure gradient tracking is on, and do a pass over the data
            net.train(True)
            train_loss, train_correct = clock_train(train_loader, optimizer, criterion, device=device)
            net.eval()
            vloss, vcorrect = clock_val(val_loader, criterion, device=device)
        else:
            net.train(True)
            train_loss, train_correct = main_train(train_loader, optimizer, criterion, device=device)
            net.eval()
            vloss, vcorrect = main_val(val_loader, criterion, device=device)
        
        record = {}
        record["train_loss"] = train_loss
        record["train_accuracy"] = train_correct
        record["validation_loss"] = vloss
        record["validation_accuracy"] = vcorrect
        record = {k: float(v) for k, v in record.items()}
        with open(train_logs_path, 'a') as f:
            f.write(json.dumps(record) + "\n")

    hp_logs = config.to_dict()
    hp_logs["last_train_accuracy"] = record["train_accuracy"]
    hp_logs["last_val_accuracy"] = record["validation_accuracy"]

    with open(hp_logs_path, 'w') as f:
        json.dump(hp_logs, f, indent=4)

    if config.training.save:
        model_path = CHECKPOINTS_DIR / f"model_{config.model.model}_{config.data.dataset_name}_{timestamp}.pth"
        
        torch.save(net.state_dict(), model_path)
        print(f"Saved PyTorch Model State to {model_path}")
    
if __name__=="__main__":
    main()