import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
from model import OptimizedCNN
import os
from tqdm import tqdm
import json
from pathlib import Path
import random
import numpy as np
import torchvision.transforms.functional as TF

class MNISTAugmentation:
    """Simplified augmentation class"""
    def __init__(self, p=0.2):  # Reduced probability
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            angle = random.uniform(-10, 10)  # Reduced rotation range
            img = TF.rotate(img, angle)
        return img

def get_transform(train=True):
    """Minimal transforms for better training fit"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def train_model(save_dir='models'):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Create save directory
    Path(save_dir).mkdir(exist_ok=True)
    Path('metrics').mkdir(exist_ok=True)

    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load dataset with optimized settings for CPU
    try:
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=get_transform(train=True))
        val_dataset = datasets.MNIST('data', train=False, download=True, transform=get_transform(train=False))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,  # Keep small batch size
            shuffle=True,
            num_workers=2,
            pin_memory=False
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1000,
            shuffle=False,
            num_workers=2,
            pin_memory=False
        )
        print("Datasets loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    # Initialize model and verify parameters
    print("Initializing model...")
    model = OptimizedCNN().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} parameters")

    if num_params >= 25000:
        raise ValueError(f"Model has {num_params} parameters, exceeding limit of 25,000")

    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay=0)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.025,  # Even higher max LR
        epochs=1,
        steps_per_epoch=len(train_loader),
        pct_start=0.5,  # 50% warmup
        div_factor=3,  # Higher initial LR
        final_div_factor=10
    )

    # Training loop
    print("Starting training...")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_accuracies = []

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Calculate accuracy
        with torch.no_grad():
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            batch_acc = 100. * correct / total
            batch_accuracies.append(batch_acc)

        # Update running loss and progress bar
        running_loss += loss.item()
        if batch_idx % 10 == 0:
            pbar.set_description(
                f'Loss: {running_loss / (batch_idx + 1):.3f} | '
                f'Acc: {batch_acc:.2f}% | '
                f'LR: {scheduler.get_last_lr()[0]:.6f}'
            )

    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    print("\nRunning validation...")
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            val_total += target.size(0)
            val_correct += predicted.eq(target).sum().item()

    # Calculate final metrics
    final_training_acc = batch_accuracies[-1]
    final_val_acc = 100. * val_correct / val_total
    final_train_loss = running_loss / len(train_loader)
    final_val_loss = val_loss / len(val_loader)

    print(f'\nFinal Training Accuracy: {final_training_acc:.2f}%')
    print(f'Final Validation Accuracy: {final_val_acc:.2f}%')
    print(f'Final Training Loss: {final_train_loss:.4f}')
    print(f'Final Validation Loss: {final_val_loss:.4f}')

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'epochs': 1,
        'training_accuracy': final_training_acc,
        'validation_accuracy': final_val_acc,
        'training_loss': final_train_loss,
        'validation_loss': final_val_loss,
        'parameters': num_params,
        'timestamp': timestamp,
        'device': str(device),
        'batch_size': train_loader.batch_size,
        'optimizer': optimizer.__class__.__name__,
        'learning_rate': {
            'initial': scheduler.get_last_lr()[0],
            'max': 0.005,
            'final': scheduler.get_last_lr()[0]
        }
    }

    # Save results to JSON
    results_path = 'metrics/training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Training results saved to {results_path}")

    # Save model
    model_path = os.path.join(save_dir, f'model_{timestamp}_acc{final_training_acc:.1f}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': 1,
        'training_accuracy': final_training_acc,
        'validation_accuracy': final_val_acc,
        'training_loss': final_train_loss,
        'validation_loss': final_val_loss,
        'parameters': num_params
    }, model_path)
    print(f"Model saved as {model_path}")

    if final_training_acc < 95:
        print(f"Warning: Training accuracy {final_training_acc:.2f}% is below target of 95%")

    return model_path, results

if __name__ == "__main__":
    train_model()