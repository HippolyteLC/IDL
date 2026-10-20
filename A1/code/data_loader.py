import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data.dataset import random_split

from pathlib import Path


class DataHandler:
    def __init__(self, data_dir="../data", dataset_name="fashionmnist", batch_size=64, val_split=0.1, seed=111):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        if self.dataset_name.lower() == "fashionmnist": 
            self.data_class = datasets.FashionMNIST 
        elif self.dataset_name.lower() == "cifar10":
            self.data_class = datasets.CIFAR10

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def initialize_dataset(self):
        """
        Loads dataset (either dl or from data dir)
        Then wraps them in dataloaders. 
        """
        train_val = self.data_class(
            root=str(self.data_dir),
            train=True,
            download=True,
            transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        )
        
        generator = torch.Generator().manual_seed(self.seed)
        train_length = int(round(len(train_val)*(1-self.val_split)))
        val_length = len(train_val) - train_length
        train, val = random_split(train_val, [train_length, val_length], generator)

        test = self.data_class(
                root=str(self.data_dir),
                train=False,
                download=True,
                transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        )

        self.train_loader = DataLoader(train, self.batch_size, )
        self.val_loader = DataLoader(val, self.batch_size)
        self.test_loader = DataLoader(test, self.batch_size)
        # optionally add standardization to 0 mean and 1 std. 
    
    def data_loader(self):
        if self.train_loader is None:
            self.initialize_dataset()
        return self.train_loader, self.val_loader, self.test_loader

def main():
    train_loader, val_loader, test_loader = DataHandler().data_loader()
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}") # Should be 64, 1 (channel), 28,28
    print(f"Batch label shape: {labels.shape}") # Should be 64, (labels)
    print(f"Pixel value range: min={images.min():.2f}, max={images.max():.2f}") # the scaling from v2Compose in [0,1]

if __name__=="__main__":
    main()