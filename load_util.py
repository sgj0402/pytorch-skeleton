# This file is depedent on the experiment
# Modify this file to load the model from the model directory

from model.model import myCNN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ignite.metrics import Accuracy, Loss

from omegaconf import DictConfig


def load_device(preferred_device=None):
    
    if preferred_device is not None:
        print(f'{preferred_device} is available, Using {preferred_device}')
        return torch.device(preferred_device)
    
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        print('mps is available, Using mps')
        return torch.device('mps')
    
    elif torch.cuda.is_available():
        print('cuda is available, Using cuda')
        return torch.device('cuda')
    
    else:
        print('No gpu is available, Using cpu')
        return torch.device('cpu')


# Do Everyting you need with this function!
def load_model(device):
    model = myCNN()
    model.to(device)
    return model


def load_train_dataloader() -> DataLoader:
    """
    return train dataloader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train = True, download = True, transform = transform)

    train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True)

    return train_dataloader


def load_test_dataloader() -> DataLoader:
    """
    return test dataloader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('./data', train = False, transform = transform)

    test_dataloader = DataLoader(test_dataset, batch_size = 1000, shuffle = False)

    return test_dataloader


def load_dataloaders() -> tuple[DataLoader, DataLoader]:

    train_dataloader = load_train_dataloader()
    test_dataloader = load_test_dataloader()

    return train_dataloader, test_dataloader


def load_test_metrics() -> dict:
    test_metric = {
                "accuracy": Accuracy(),
                "loss": Loss(load_loss_fn())
            }
    
    return test_metric


def load_loss_fn():
    return nn.CrossEntropyLoss()