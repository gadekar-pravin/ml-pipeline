
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Reduced number of filters and feature maps
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # Changed from 16 to 8 channels
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # Changed from 32 to 16 channels
        self.fc1 = nn.Linear(16 * 7 * 7, 64)  # Reduced from 128 to 64 neurons
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Print model summary
    model = SimpleCNN()
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")

    # Print layer-wise parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")