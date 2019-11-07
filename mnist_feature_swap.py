from svcca import cca_core
import fastai.vision as fv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import pickle

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn.functional as F


batch_size = 32

train_loader_mnist = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
test_loader_mnist = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

train_loader_fashion = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
test_loader_fashion = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
device = "cuda"

model_mnist = Net().to(device)
model_fashion = Net().to(device)
model_mnist.train()
model_fashion.train()

optimizer_mnist = torch.optim.Adam(model_mnist.parameters())
optimizer_fashion = torch.optim.Adam(model_fashion.parameters())

print("initialization of last layers")
print(cca_core.get_cca_similarity(model_mnist.fc2.weight.cpu().detach().numpy(),model_fashion.fc2.weight.cpu().detach().numpy())['mean'])


def update_model(model, optimizer, data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def validate_model(model, data, target):
    with torch.no_grad():
        output = model(data)
    return (np.argmax(output, axis=1) == target.cpu()).sum().item()/len(target)
    


for epoch in range(1):
    # Train mnist model
    for batch_idx, (data, target) in enumerate(train_loader_mnist):
        data, target = data.float().to(device), target.to(device)
        loss_val_mnist = update_model(model_mnist, optimizer_mnist, data, target)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_mnist: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader_mnist.dataset),
                100. * batch_idx / len(train_loader_mnist), loss_val_mnist))

    acc_val_mnist = []
    for batch_idx, (data, target) in enumerate(test_loader_mnist):
        data, target = data.float().to(device), target.to(device)
        acc_val_mnist.append(validate_model(model_mnist, data, target))
    print('Test Epoch: {} [{}/{} ({:.0f}%)]\tacc_mnist: {:.4f}'.format(
        epoch, batch_idx * len(data), len(test_loader_mnist.dataset),
        100. * batch_idx / len(train_loader_mnist),np.mean(acc_val_mnist)))

    # Train fashion model
    for batch_idx, (data, target) in enumerate(train_loader_fashion):
        data, target = data.float().to(device), target.to(device)
        loss_val_fashion = update_model(model_fashion, optimizer_fashion, data, target)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_fashion: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader_fashion.dataset),
                100. * batch_idx / len(train_loader_fashion), loss_val_fashion))

    acc_val_fashion = []
    for batch_idx, (data, target) in enumerate(test_loader_fashion):
        data, target = data.float().to(device), target.to(device)
        acc_val_fashion.append(validate_model(model_fashion, data, target))
    print('Test Epoch: {} [{}/{} ({:.0f}%)]\tacc_fashion: {:.4f}'.format(
        epoch, batch_idx * len(data), len(test_loader_fashion.dataset),
        100. * batch_idx / len(train_loader_fashion),np.mean(acc_val_fashion)))


#now we have the trained models, we need to swap around the heads and train them again.
model_mnist.conv1.requires_grad=False
model_mnist.conv2.requires_grad=False
model_mnist.fc1.requires_grad=False

prev_head_mnist = model_mnist.fc2

model_fashion.conv1.requires_grad=False
model_fashion.conv2.requires_grad=False
model_fashion.fc1.requires_grad=False

prev_head_fashion = model_fashion.fc2

cca_after_first = cca_core.get_cca_similarity(prev_head_mnist.weight.cpu().detach().numpy(), prev_head_fashion.weight.cpu().detach().numpy())['mean']

# we need to first reset the heads of each of the networks
model_mnist.fc2 = nn.Linear(500,10).cuda()
model_fashion.fc2 = nn.Linear(500,10).cuda()

for epoch in range(1):
    # Train mnist->fashion model
    for batch_idx, (data, target) in enumerate(train_loader_fashion):
        data, target = data.float().to(device), target.to(device)
        loss_val_mnist = update_model(model_mnist, optimizer_mnist, data, target)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_mnist->fashion: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader_fashion.dataset),
                100. * batch_idx / len(train_loader_fashion), loss_val_mnist))

    acc_val_mnist = []
    for batch_idx, (data, target) in enumerate(test_loader_fashion):
        data, target = data.float().to(device), target.to(device)
        acc_val_mnist.append(validate_model(model_mnist, data, target))
    print('Test Epoch: {} [{}/{} ({:.0f}%)]\tacc_mnist->fashion: {:.4f}'.format(
        epoch, batch_idx * len(data), len(test_loader_fashion.dataset),
        100. * batch_idx / len(train_loader_fashion),np.mean(acc_val_mnist)))

    # Train fashion->mnist model
    for batch_idx, (data, target) in enumerate(train_loader_mnist):
        data, target = data.float().to(device), target.to(device)
        loss_val_fashion = update_model(model_fashion, optimizer_fashion, data, target)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_fashion->mnist: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader_mnist.dataset),
                100. * batch_idx / len(train_loader_mnist), loss_val_fashion))

    acc_val_fashion = []
    for batch_idx, (data, target) in enumerate(test_loader_mnist):
        data, target = data.float().to(device), target.to(device)
        acc_val_fashion.append(validate_model(model_fashion, data, target))
    print('Test Epoch: {} [{}/{} ({:.0f}%)]\tacc_fashion: {:.4f}'.format(
        epoch, batch_idx * len(data), len(test_loader_mnist.dataset),
        100. * batch_idx / len(train_loader_mnist),np.mean(acc_val_fashion)))


cca_after_second = cca_core.get_cca_similarity(model_mnist.fc2.weight.cpu().detach().numpy(), model_fashion.fc2.weight.cpu().detach().numpy())['mean']
inter_training_cca_mnist = cca_core.get_cca_similarity(model_mnist.fc2.weight.cpu().detach().numpy(), prev_head_fashion.weight.cpu().detach().numpy())['mean']
inter_training_cca_fashion = cca_core.get_cca_similarity(model_fashion.fc2.weight.cpu().detach().numpy(), prev_head_mnist.weight.cpu().detach().numpy())['mean']
final_mnist = cca_core.get_cca_similarity(prev_head_mnist.weight.cpu().detach().numpy(), model_mnist.fc2.weight.cpu().detach().numpy())['mean']
final_fashion = cca_core.get_cca_similarity(prev_head_fashion.weight.cpu().detach().numpy(), model_fashion.fc2.weight.cpu().detach().numpy())['mean']

print("mnist vs fashion")
print(cca_after_first)
print("mnist->fashion vs fashion->mnist")
print(cca_after_second)
print("mnist->fashion vs fashion")
print(inter_training_cca_mnist)
print("fashion->mnist vs mnist")
print(inter_training_cca_fashion)
print("mnist vs mnist->fashion")
print(final_mnist)
print("fashion vs fashion->mnist")
print(final_fashion)