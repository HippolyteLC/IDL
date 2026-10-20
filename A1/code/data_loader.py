import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, BatchSampler


from pathlib import Path


class DataHandler:
    def __init__(self, data_dir="../data", dataset_name="fashionmnist", device="cpu", batch_size=64, val_split=0.1, seed=111):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        if self.dataset_name.lower() == "fashionmnist": 
            self.data_class = datasets.FashionMNIST 
        elif self.dataset_name.lower() == "cifar10":
            self.data_class = datasets.CIFAR10
        self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    @staticmethod
    def to_tensors(dataset, batch_size):
        """One pass through a torchvision dataset -> (X, y) tensors."""
        xs, ys = [], []
        for x, y in DataLoader(dataset, batch_size=batch_size):
            xs.append(x); ys.append(y)
        return torch.cat(xs), torch.cat(ys)

    def make_loader(self, X, y, shuffle=True):
        ds = TensorDataset(X.to(self.device), y.to(self.device))
        if shuffle:
            generator = torch.Generator().manual_seed(self.seed)
            base = RandomSampler(ds, generator=generator)
        else:
            base = SequentialSampler(ds)
        return DataLoader(ds, batch_size=None, sampler=BatchSampler(base, self.batch_size, drop_last=False))

    def initialize_dataset(self):
        """
        Loads dataset (either dl or from data dir)
        Then wraps them in dataloaders. 
        """
        train_val = self.data_class(root=str(self.data_dir), train=True, download=True, transform=self.transform)

        generator = torch.Generator().manual_seed(self.seed)
        train_length = int(round(len(train_val)*(1-self.val_split)))
        val_length = len(train_val) - train_length
        train, val = random_split(train_val, [train_length, val_length], generator)

        X, y = self.to_tensors(train_val, self.batch_size)
        X_train, y_train = X[train.indices], y[train.indices] 
        X_val, y_val = X[val.indices], y[val.indices]

        test = self.data_class(root=str(self.data_dir), train=False, download=True, transform=self.transform)
        X_test, y_test = self.to_tensors(test, self.batch_size)

        self.train_loader = self.make_loader(X_train, y_train, shuffle=True)
        self.val_loader = self.make_loader(X_val, y_val, shuffle=False)
        self.test_loader = self.make_loader(X_test, y_test, shuffle=False)
    
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