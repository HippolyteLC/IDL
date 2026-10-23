import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, BatchSampler

import os
import pandas as pd
import numpy as np
from torchvision.io import decode_image
from torchvision.transforms import Normalize

from pathlib import Path


def transform(ds):
    std, mean = torch.std_mean(ds)
    return Normalize(mean, std)

def target_transform(ds):
    hours_in_mins = ds[:,0:] * 60 
    mins = ds[:,1:]
    angle_hour = 2 * np.pi * (hours_in_mins+mins)/60/12
    angle_min = 2 * np.pi * mins/60
    return torch.stack([np.sin(angle_hour), np.cos(angle_min), np.sin(angle_hour), np.cos(angle_min)])

class ClockDataset(Dataset):
    def __init__(self, dataset_name="clock_75", transform=None, target_transform=None):
        if dataset_name == "clock_75":
            sub_dir = "clock_data_75" 
        elif dataset_name == "150":
            sub_dir = "clock_data_150" 
        else:
            raise NameError("choose a valid dataset name")
        ds_dir = Path(__file__).parent.parent / f"data/{sub_dir}"
        annotations_file = ds_dir / "labels.npy"
        img_file = ds_dir / "images.npy"
        self.img_labels = torch.from_numpy(np.load(annotations_file))
        raw_imgs = torch.from_numpy(np.load(img_file))
        if len(raw_imgs.shape) == 4:
            self.imgs = raw_imgs.permute(0, 3, 1, 2)
        else:
            self.imgs = raw_imgs 
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # image = decode_image(img_path)
        image = self.imgs[idx]
        label = self.img_labels[idx]
        return image, label

class DataHandler:
    def __init__(self, data_dir, dataset_name="fashionmnist", device="cpu", batch_size=1024, val_split=0.1, seed=111, transform=None, target_transform=None):
        
        self.data_dir = data_dir if data_dir else Path(data_dir).expanduser().resolve()
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
            self.data_class = ClockDataset(dataset_name=dataset_name, transform=transform, target_transform=target_transform)

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
            dataset_length = len(self.data_class)
            train_length = int(dataset_length * 0.8)
            val_length = int(dataset_length * 0.1) 
            test_length = dataset_length - train_length - val_length
            train, val, test = random_split(self.data_class, [train_length, val_length, test_length], generator=generator)

            imgs, labels = self.data_class.imgs, self.data_class.img_labels
            X_train, y_train = imgs[train.indices], labels[train.indices] 
            X_val, y_val = imgs[val.indices], labels[val.indices] 
            X_test, y_test = imgs[test.indices], labels[test.indices] 

            if self.data_class.transform is not None:
                X_train = self.data_class.transform(X_train)
                X_val = self.data_class.transform(X_val)
                X_test = self.data_class.transform(X_test)
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
    train_loader, val_loader, test_loader = DataHandler().data_loader()
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}") # Should be 64, 1 (channel), 28,28
    print(f"Batch label shape: {labels.shape}") # Should be 64, (labels)
    print(f"Pixel value range: min={images.min():.2f}, max={images.max():.2f}") # the scaling from v2Compose in [0,1]

if __name__=="__main__":
    main()