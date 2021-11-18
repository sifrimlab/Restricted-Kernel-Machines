import math
import os
import shutil
import time
import urllib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyper-parameters =================
capacity = 32
x_fdim = 128
y_fdim = 20

class create_dirs:
    """ Creates directories for Checkpoints and saving trained models """

    def __init__(self, ct):
        self.ct = ct
        self.dircp = 'checkpoint.pth_{}.tar'.format(self.ct)
        self.dirout = 'Mul_trained_RKM_{}.tar'.format(self.ct)

    def create(self):
        if not os.path.exists('cp/'):
            os.makedirs('cp/')

        if not os.path.exists('out/'):
            os.makedirs('out/')

    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state, 'cp/{}'.format(self.dircp))


""" Note: Do not change the architecture, since it is used to initialize the pre-trained models while generation.
    However, while training from scratch, one can ofcourse define new architectures """


# Feature-maps - network architecture
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=1)
        self.fc1 = torch.nn.Linear(c * 2 * 7 * 7, x_fdim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = torch.nn.Linear(10, 15)
        self.fc2 = torch.nn.Linear(15, y_fdim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.fc2(x)
        return x


# Pre-image maps - network architecture
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        c = capacity
        self.fc1 = nn.Linear(in_features=x_fdim, out_features=c * 2 * 7 * 7)
        self.conv2 = nn.ConvTranspose2d(in_channels=c * 2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        if x.dim() == 1:
            x = x.view(1, capacity * 2, 7, 7)
        else:
            x = x.view(x.size(0), capacity * 2, 7, 7)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = torch.sigmoid(self.conv1(x))
        return x


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.fc1 = torch.nn.Linear(y_fdim, 15)
        self.fc2 = torch.nn.Linear(15, 10)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = torch.sigmoid(self.fc2(x))
        return x


# Download training data if already doesn't exist ===================

class FastMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).double().div(255)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data, torch.nn.functional.one_hot(self.targets).double()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target


def get_mnist_dataloader(args, path_to_data='mnist'):
    """MNIST dataloader with (28, 28) images."""

    all_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = FastMNIST(path_to_data, train=True, download=True, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle, pin_memory=False,
                              num_workers=0)
    _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader, c * x * y, c


def final_compute(args, net1, net2, kPCA, device=torch.device('cuda')):
    """ Function to compute embeddings of full dataset. """
    args.shuffle = False
    xt, _, _ = get_mnist_dataloader(args=args)  # loading data without shuffle
    xtr = net1(xt.dataset.train_data[:args.N, :, :, :].to(args.device))
    ytr = net2(xt.dataset.targets[:args.N, :].to(args.device))

    h, s = kPCA(xtr, ytr)
    return torch.mm(torch.t(xtr), h), torch.mm(torch.t(ytr), h), h, s
