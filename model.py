
import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        # First convolutional layer: input=1, output=4, kernel=3x3
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        # Second convolutional layer: input=4, output=8, kernel=3x3
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        # Third convolutional layer: input=8, output=16, kernel=3x3
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, 10)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)

        # Dropout
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        # Flatten and fully connected layers
        x = x.view(-1, 16 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Print model summary
    model = OptimizedCNN()
    total_params = count_parameters(model)
    print(f"\nModel Parameter Analysis:")
    print(f"Total trainable parameters: {total_params}")

    # Print layer-wise parameters
    print("\nLayer-wise parameter count:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")