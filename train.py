
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
from model import SimpleCNN
import ssl
import os
import urllib.request
import gzip
import numpy as np


def download_mnist():
    """Download MNIST dataset from alternative source"""
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    filenames = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }

    os.makedirs('data/MNIST/raw', exist_ok=True)

    for name, filename in filenames.items():
        filepath = os.path.join('data/MNIST/raw', filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            url = base_url + filename
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                return False
    return True


def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Download dataset
    print("Downloading and loading MNIST dataset...")
    if not download_mnist():
        print("Failed to download dataset")
        return

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        train_dataset = datasets.MNIST('data', train=True, download=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    # Initialize model
    print("Initializing model...")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    print("Starting training...")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Training Batch: {batch_idx}/{len(train_loader)} Loss: {loss.item():.6f}')

    # Save model with timestamp
    print("Saving model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'model_{timestamp}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")
    return model_path


if __name__ == "__main__":
    train_model()