# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
from model import SimpleCNN
import os
import urllib.request
import gzip
import numpy as np
import idx2numpy


def download_and_extract_mnist():
    """Download MNIST dataset and extract files"""
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }

    data_dir = 'data/MNIST/raw'
    os.makedirs(data_dir, exist_ok=True)

    # Download and extract files
    for name, filename in files.items():
        gz_path = os.path.join(data_dir, filename)
        out_path = os.path.join(data_dir, filename[:-3])  # Remove .gz

        # Download if not exists
        if not os.path.exists(gz_path):
            print(f"Downloading {filename}...")
            url = base_url + filename
            urllib.request.urlretrieve(url, gz_path)
            print(f"Successfully downloaded {filename}")

        # Extract if not exists
        if not os.path.exists(out_path):
            print(f"Extracting {filename}...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(out_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            print(f"Successfully extracted {filename}")


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = idx2numpy.convert_from_file(images_path)
        self.labels = idx2numpy.convert_from_file(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])

        # Add channel dimension and convert to float32
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        if self.transform:
            image = self.transform(torch.from_numpy(image))

        return image, label


def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download and extract MNIST dataset
    print("Downloading and extracting MNIST dataset...")
    download_and_extract_mnist()

    # Data loading
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load dataset
    try:
        train_dataset = MNISTDataset(
            'data/MNIST/raw/train-images-idx3-ubyte',
            'data/MNIST/raw/train-labels-idx1-ubyte',
            transform=transform
        )
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