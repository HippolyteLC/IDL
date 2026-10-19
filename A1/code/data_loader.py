import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data.dataset import random_split

from sklearn.model_selection import train_test_split



class DataHandler:
    def __init__(self, data_dir="../data", dataset_name="fashionmnist", batch_size=64, val_split=0.1):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.val_split = val_split
        self.data_class = datasets.FashionMNIST if self.dataset_name == "fashionmnist" else datasets.CIFAR10

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def initialize_dataset(self, seed=111):
        """
        Loads dataset (either dl or from data dir)
        Then wraps them in dataloaders. 
        """
        train_val = self.data_class(
            root=self.data_dir,
            train=True,
            transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        )
        
        generator = torch.Generator().manual_seed(seed)
        train_length = int(round(len(train_val)*(1-self.val_split)))
        val_length = len(train_val) - train_length
        train, val = random_split(train_val, [train_length, val_length], generator)

        test = self.data_class(
                root=self.data_dir,
                train=False,
                transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        )

        self.train_loader = DataLoader(train, self.batch_size)
        self.val_loader = DataLoader(val, self.batch_size)
        self.test_loader = DataLoader(test, self.batch_size)
        # optionally add standardization to 0 mean and 1 std. 
    
    def data_loader(self):
        if self.train_loader is None:
            self.initialize_dataset()
        return self.train_loader, self.val_loader, self.test_loader

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