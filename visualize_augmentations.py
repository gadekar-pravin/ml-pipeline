import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np


def visualize_augmentations(num_samples=5):
    """
    Visualize original and augmented MNIST images side by side
    with more pronounced augmentations
    """
    # Original transform (just convert to tensor, no normalization for visualization)
    orig_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Augmented transform with more pronounced effects
    aug_transform = transforms.Compose([
        transforms.RandomRotation(30),  # Increased rotation
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Add translation
            scale=(0.8, 1.2),  # Add scaling
            shear=20  # Add shearing
        ),
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = datasets.MNIST('data', train=True, download=True)

    # Set up the plot
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2.5 * num_samples))
    plt.suptitle('Original vs Augmented MNIST Images', size=16)

    # Get random indices
    indices = np.random.randint(0, len(dataset), num_samples)

    for i, idx in enumerate(indices):
        # Get original image
        img, label = dataset[idx]

        # Apply transforms
        orig_img = orig_transform(img).squeeze()
        aug_img = aug_transform(img).squeeze()

        # Plot original
        axes[i, 0].imshow(orig_img, cmap='gray')
        axes[i, 0].set_title(f'Original (Label: {label})')
        axes[i, 0].axis('off')

        # Plot augmented
        axes[i, 1].imshow(aug_img, cmap='gray')
        axes[i, 1].set_title('Augmented')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


# Show multiple augmentations of the same image
def visualize_multiple_augmentations(num_augmentations=5):
    """
    Visualize one original image with multiple different augmentations
    """
    # Define transforms
    orig_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    aug_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=20
        ),
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = datasets.MNIST('data', train=True, download=True)

    # Set up the plot
    fig, axes = plt.subplots(1, num_augmentations + 1, figsize=(2.5 * (num_augmentations + 1), 3))
    plt.suptitle('One Image with Multiple Augmentations', size=16)

    # Get a random image
    idx = np.random.randint(0, len(dataset))
    img, label = dataset[idx]

    # Plot original
    orig_img = orig_transform(img).squeeze()
    axes[0].imshow(orig_img, cmap='gray')
    axes[0].set_title(f'Original\n(Label: {label})')
    axes[0].axis('off')

    # Plot multiple augmentations
    for i in range(num_augmentations):
        aug_img = aug_transform(img).squeeze()
        axes[i + 1].imshow(aug_img, cmap='gray')
        axes[i + 1].set_title(f'Aug {i + 1}')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Showing different samples with augmentation:")
    visualize_augmentations()

    print("\nShowing multiple augmentations of the same image:")
    visualize_multiple_augmentations()