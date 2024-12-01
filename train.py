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
from torch.cuda.amp import GradScaler, autocast


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(save_dir='models'):
    # Create save directory
    Path(save_dir).mkdir(exist_ok=True)
    Path('metrics').mkdir(exist_ok=True)

    # Set device and enable CUDA optimizations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    # Enhanced data augmentation for better single-epoch performance
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomErasing(p=0.5)
    ])

    # Load dataset with optimized settings
    try:
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=512,  # Larger batch size for faster training
            shuffle=True,
            num_workers=4,
            pin_memory=True  # Faster data transfer to GPU
        )
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    # Initialize model and verify parameters
    print("Initializing model...")
    model = OptimizedCNN().to(device)
    num_params = count_parameters(model)
    print(f"Model has {num_params} parameters")

    if num_params >= 25000:
        raise ValueError(f"Model has {num_params} parameters, exceeding limit of 25,000")

    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.05,  # L2 regularization
        amsgrad=True  # Better optimization for Adam
    )

    # Use OneCycleLR with optimized parameters
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=1,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,  # Warm-up for 20% of the training
        div_factor=25,  # Initial lr = max_lr/25
        final_div_factor=1000,  # Final lr = max_lr/1000
    )

    # Initialize mixed precision training
    scaler = GradScaler()

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

        # Mixed precision training
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # Optimize
        optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
        if batch_idx % 20 == 0:
            pbar.set_description(
                f'Loss: {running_loss / (batch_idx + 1):.3f} | '
                f'Acc: {batch_acc:.2f}% | '
                f'LR: {scheduler.get_last_lr()[0]:.6f}'
            )

    # Calculate final metrics
    final_training_acc = batch_accuracies[-1]
    final_loss = running_loss / len(train_loader)
    print(f'\nFinal Training Accuracy: {final_training_acc:.2f}%')
    print(f'Final Loss: {final_loss:.4f}')

    # Save training results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'epochs': 1,
        'training_accuracy': final_training_acc,
        'parameters': num_params,
        'final_loss': final_loss,
        'timestamp': timestamp,
        'device': str(device),
        'batch_size': train_loader.batch_size,
        'optimizer': optimizer.__class__.__name__,
        'learning_rate': {
            'initial': scheduler.get_last_lr()[0],
            'max': 0.01,
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
        'accuracy': final_training_acc,
        'loss': final_loss,
        'parameters': num_params
    }, model_path)
    print(f"Model saved as {model_path}")

    if final_training_acc < 95:
        print(f"Warning: Training accuracy {final_training_acc:.2f}% is below target of 95%")

    return model_path, results


if __name__ == "__main__":
    train_model()