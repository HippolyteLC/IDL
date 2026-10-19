import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2

from sklearn.model_selection import train_test_split



class DataHandler:
    def __init__(self, data_dir="../data", dataset_name="fashionmnist", batch_size=64, val_split=0.1):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.val_split = val_split
        self.data_class = datasets.FashionMNIST if self.dataset_name == "fashionmnist" else datasets.CIFAR10

    def load_dataset(self, name="fashionmnist"):
        if name == "fasionmnist":
            train_val = datasets.FashionMNIST(
            root="data",
            train=True,
            download=False,
            transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
            )
            train, val = train_test_split()

            test = datasets.FashionMNIST(
            root="data",
            train=False,
            download=False,
            transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
            )
            # optionally add standardization to 0 mean and 1 std. 

def main():
    """ 
    Download required datasets
    """
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )

    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )

if __name__=="main":
    main()