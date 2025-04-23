import pytest
import torch
from torch import nn


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    def forward(self, x):
        return self.net(x)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3)  # Output: (batch, 4, 6, 6)
        self.fc = nn.Linear(4 * 6 * 6, 10)  # Correct size: 144 → 10 classes

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 100)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def tiny_mlp():
    return TinyMLP()


@pytest.fixture
def tiny_cnn():
    return TinyCNN()


@pytest.fixture
def simple_cnn():
    return SimpleCNN()
