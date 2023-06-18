"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self, inputSize, outputSize):
        super(NetLin, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(inputSize, outputSize)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, val):
        val = self.flatten(val)
        val = self.linear(val)
        val = self.logsoftmax(val)
        return val


class NetFull(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(NetFull, self).__init__()
        self.tanh = torch.nn.Tanh()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(inputSize, 100)
        self.output = nn.Linear(100, outputSize)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, val):
        val = self.flatten(val)
        val = self.linear(val)
        val = self.tanh(val)
        val = self.output(val)
        val = self.logsoftmax(val)
        return val


class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.fc = nn.Linear(64 * 22 * 22, 128)
        self.flatten = nn.Flatten()
        self.output = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.logsoftmax(x)
        return x
