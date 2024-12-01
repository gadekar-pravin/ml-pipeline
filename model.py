import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        # First conv layer with stride
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=2, padding=2)  # 28->14
        self.bn1 = nn.BatchNorm2d(10)

        # Second conv layer with stride
        self.conv2 = nn.Conv2d(10, 12, kernel_size=3, stride=2, padding=1)  # 14->7
        self.bn2 = nn.BatchNorm2d(12)

        # Small fully connected layers
        self.fc1 = nn.Linear(12 * 7 * 7, 24)
        self.fc2 = nn.Linear(24, 10)

    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))

        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))

        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)