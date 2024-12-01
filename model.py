import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        # First convolutional layer (1 -> 8)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        # Second convolutional layer (8 -> 16)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # Third maxpool to reduce FC layer size
        # After three max pools: 28->14->7->3
        self.fc1 = nn.Linear(16 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, 10)

        # Minimal dropout
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.max_pool2d(x, 2)  # Additional pooling

        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)