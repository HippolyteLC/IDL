import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, BatchSampler

import os
import pandas as pd
import numpy as np

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"

class ClockDataset(Dataset):
    def __init__(self, data_dir, dataset_name="clock_75", seed=111, test_true=False):
        if dataset_name == "clock_75":
            sub_dir = "clock_data_75" 
        elif dataset_name == "clock_150":
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
        train_val_length = int(dataset_length * 0.9)
        test_length = dataset_length - train_val_length
        train_val, test = random_split(self.img_labels, [train_val_length, test_length], generator=generator)
        if not test_true:
            train_length = int(dataset_length * 0.8)
            val_length = train_val_length - train_length
            train, val = random_split(train_val, [train_length, val_length], generator=generator)
        else:
            train = train_val
            val = Subset(self.img_labels, [])

        self.splits = train, val, test
        imgs = v2.ToDtype(torch.float32, scale=True)(imgs)
        std, mean = torch.std_mean(imgs[train.indices], dim=(0,2,3))
        self.imgs = v2.Normalize(mean, std)(imgs)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.img_labels[idx]

class DataHandler:
    def __init__(self, experiment_config, device):
        self.data_dir = DATA_DIR
        self.device = device

        self.test_true = experiment_config.training.test

        data_config = experiment_config.data
        self.dataset_name = data_config.dataset_name
        self.batch_size = data_config.batch_size
        self.val_split = data_config.val_split
        self.seed = data_config.split_seed
        if self.dataset_name.lower() == "fashionmnist": 
            self.data_class = datasets.FashionMNIST 
        elif self.dataset_name.lower() == "cifar10":
            self.data_class = datasets.CIFAR10
        elif self.dataset_name.lower() in ("clock_75", "clock_150"):
            self.clock_ds = ClockDataset(data_dir=self.data_dir, dataset_name=self.dataset_name, \
                                         seed=self.seed, test_true=self.test_true)
        self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def helper(self, subset):
        ds = self.clock_ds
        imgs, labels = ds.imgs, ds.img_labels
        return imgs[subset.indices], labels[subset.indices] 

    def initialize_dataset(self):
        """
        Loads dataset (either dl or from data dir)
        Then wraps them in dataloaders. 
        """
        generator = torch.Generator().manual_seed(self.seed)

        loader_kwargs = {
        "batch_size": self.batch_size,
        "num_workers": 4,          
        "pin_memory": True,        
        "persistent_workers": True          
        }

        if self.dataset_name in ("clock_75", "clock_150"):
            train, val, test = self.clock_ds.splits

            train_ds = TensorDataset(self.helper(train)[0], self.helper(train)[1]) 
            val_ds =  TensorDataset(self.helper(val)[0], self.helper(val)[1])
            test_ds = TensorDataset(self.helper(test)[0], self.helper(test)[1])

        else:
            train_val = self.data_class(root=str(self.data_dir), train=True, download=True, transform=self.transform)
            if not self.test_true:
                train_length = int(round(len(train_val)*(1-self.val_split)))
                val_length = len(train_val) - train_length
                train_ds, val_ds = random_split(train_val, [train_length, val_length], generator)
            else:
                train_ds = train_val
                val_ds = Subset(train_val, [])
            test_ds = self.data_class(root=str(self.data_dir), train=False, download=True, transform=self.transform)     

        self.train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
        self.test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    
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