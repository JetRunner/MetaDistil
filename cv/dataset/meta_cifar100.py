from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from .cifar_with_held import CIFAR100WithHeld
from PIL import Image

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


def get_cifar100_dataloaders(batch_size=128, num_workers=8, held_size=0, num_held_samples=0):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = CIFAR100WithHeld(root=data_folder,
                                 download=True,
                                 train=True,
                                 held=False,
                                 held_samples=held_size,
                                 transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    held_set = CIFAR100WithHeld(root=data_folder,
                                download=True,
                                train=True,
                                held=True,
                                held_samples=held_size,
                                transform=train_transform)

    if num_held_samples == 0:
        held_loader = DataLoader(held_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    else:
        held_sampler = RandomSampler(held_set,
                                     replacement=True,
                                     num_samples=num_held_samples)
        held_loader = DataLoader(held_set,
                                 sampler=held_sampler,
                                 batch_size=batch_size)

    test_set = CIFAR100WithHeld(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, held_loader, test_loader
