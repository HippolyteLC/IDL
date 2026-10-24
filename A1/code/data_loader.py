import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.functional import normalize
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, BatchSampler

import os
import pandas as pd
import numpy as np
from torchvision.io import decode_image

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

class ClockDataset(Dataset):
    def __init__(self, data_dir, dataset_name="clock_75", target_transform=None, seed=111):
        if dataset_name == "clock_75":
            sub_dir = "clock_data_75" 
        elif dataset_name == "150":
            sub_dir = "clock_data_150" 
        else:
            raise NameError("choose a valid dataset name")
        ds_dir = data_dir / sub_dir
        annotations_file = ds_dir / "labels.npy"
        img_file = ds_dir / "images.npy"
        self.img_labels = torch.from_numpy(np.load(annotations_file))
        raw_imgs = torch.from_numpy(np.load(img_file))
        imgs = torch.unsqueeze(raw_imgs, 1)#.to(torch.float32)
        generator = torch.Generator().manual_seed(seed)

        dataset_length = len(self.img_labels)
        train_length = int(dataset_length * 0.8)
        val_length = int(dataset_length * 0.1) 
        test_length = dataset_length - train_length - val_length
        train, val, test = random_split(self.img_labels, [train_length, val_length, test_length], generator=generator)
        self.splits = train, val, test
        imgs = v2.ToDtype(torch.float32, scale=True)(imgs)
        std, mean = torch.std_mean(imgs[train.indices], dim=(0,2,3))
        self.imgs = v2.Normalize(mean, std)(imgs)

        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.img_labels[idx]

class DataHandler:
    def __init__(self, data_dir=DATA_DIR, dataset_name="fashionmnist", device="cpu", batch_size=1024, val_split=0.1, seed=111, target_transform=None):
        self.data_dir = data_dir
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
        elif self.dataset_name.lower() in ("clock_75", "clock_150"):
            self.data_class = ClockDataset(data_dir=data_dir, dataset_name=dataset_name, target_transform=target_transform, seed=seed)

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
        generator = torch.Generator().manual_seed(self.seed)

        if self.dataset_name in ("clock_75", "clock_150"):
            train, val, test = self.data_class.splits
            imgs, labels = self.data_class.imgs, self.data_class.img_labels
            X_train, y_train = imgs[train.indices], labels[train.indices] 
            X_val, y_val = imgs[val.indices], labels[val.indices] 
            X_test, y_test = imgs[test.indices], labels[test.indices] 

            if self.data_class.target_transform is not None:
                y_train = self.data_class.target_transform(y_train) 
                y_val = self.data_class.target_transform(y_val)
                y_test = self.data_class.target_transform(y_test)
        else:
            train_val = self.data_class(root=str(self.data_dir), train=True, download=True, transform=self.transform)

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
    train_loader, val_loader, test_loader = DataHandler(dataset_name="clock_75", target_transform=None).data_loader()
    images, labels = next(iter(train_loader))
    has_hour_12 = any([(labels[:, 0] == 12).any() for _, labels in test_loader]) # check if hour 12 is a value, it isnt
    print(f"Batch image shape: {images.shape}") # Should be 64, 1 (channel), 28,28
    print(f"Batch label shape: {labels.shape}") # Should be 64, (labels)
    print(f"Label looks like:{labels[0]} andhas type {type(labels[0])}")
    print(f"Pixel value range: min={images.min():.2f}, max={images.max():.2f}") # the scaling from v2Compose in [0,1]
    print(f"Has hour == 12: {has_hour_12}")
if __name__=="__main__":
    main()