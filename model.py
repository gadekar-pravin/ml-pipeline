import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # Fully connected layers
        self.fc1 = nn.Linear(784, 28)  # Reduced from 30 to 28 nodes
        self.fc2 = nn.Linear(30, 10)

        # Dropout layers
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

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