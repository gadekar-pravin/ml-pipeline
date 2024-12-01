
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
from model import OptimizedCNN
import os
from tqdm import tqdm


def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading with augmentation
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomRotation(5)
    ])

    # Load dataset
    try:
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    # Initialize model
    print("Initializing model...")
    model = OptimizedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=1,
                                              steps_per_epoch=len(train_loader))

    # Training loop
    print("Starting training...")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Calculate accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Update running loss
        running_loss += loss.item()

        # Update progress bar
        if batch_idx % 100 == 0:
            acc = 100. * correct / total
            pbar.set_description(f'Loss: {running_loss / (batch_idx + 1):.3f} | Acc: {acc:.2f}%')

    # Print final accuracy
    final_acc = 100. * correct / total
    print(f'\nFinal Training Accuracy: {final_acc:.2f}%')

    # Save model with timestamp and accuracy
    print("Saving model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'model_{timestamp}_acc{final_acc:.1f}.pth'
    torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=True)
    print(f"Model saved as {model_path}")
    return model_path


if __name__ == "__main__":
    train_model()