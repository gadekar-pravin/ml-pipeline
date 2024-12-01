import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        # Layer 1: Convolutional layer
        # Input: 1x28x28, Output: 8x14x14 (after pooling)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(10)

        # Layer 2: Convolutional layer
        # Input: 8x14x14, Output: 16x7x7 (after pooling)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # Layer 3: Fully connected layer
        # Input: 16*7*7 = 784, Output: 10
        self.fc = nn.Linear(16 * 7 * 7, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # First conv layer + batch norm + relu + pooling
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)

        # Second conv layer + batch norm + relu + pooling
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)

        # Flatten
        x = x.view(-1, 16 * 7 * 7)

        # Dropout before final layer
        x = self.dropout(x)

        # Fully connected layer
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)