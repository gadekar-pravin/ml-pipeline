import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from train import MNISTAugmentation, get_transform
import random


def show_images(images, title):
    """Display a batch of images"""
    plt.figure(figsize=(15, 8))
    plt.title(title)
    plt.imshow(images.permute(1, 2, 0))
    plt.axis('off')


def visualize_augmentations(num_examples=8):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Load MNIST dataset
    dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())

    # Get a batch of images
    images = []
    augmented_images = []
    transform = get_transform(train=True)

    # Select some examples
    indices = random.sample(range(len(dataset)), num_examples)

    # Original images
    for idx in indices:
        img, _ = dataset[idx]
        images.append(img)

    # Create grid of original images
    orig_grid = make_grid(images, nrow=num_examples, normalize=True, pad_value=1)

    # Apply augmentations
    for idx in indices:
        img, _ = dataset[idx]
        aug_img = transform(img)
        augmented_images.append(aug_img)

    # Create grid of augmented images
    aug_grid = make_grid(augmented_images, nrow=num_examples, normalize=True, pad_value=1)

    # Display results
    plt.figure(figsize=(20, 8))

    plt.subplot(2, 1, 1)
    plt.title('Original Images')
    plt.imshow(orig_grid.permute(1, 2, 0), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.title('Augmented Images')
    plt.imshow(aug_grid.permute(1, 2, 0), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_examples.png')
    plt.close()

    # Demonstrate individual augmentations
    augmentations = {
        'Original': transforms.ToTensor(),
        'Rotation': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(15)
        ]),
        'Perspective': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomPerspective(distortion_scale=0.2, p=1.0)
        ]),
        'Elastic': transforms.Compose([
            transforms.ToTensor(),
            lambda x: transforms.functional.elastic_transform(x, alpha=50.0, sigma=4.0)
        ]),
        'Affine': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5)
        ]),
        'Erasing': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(p=1.0)
        ]),
        'Blur': transforms.Compose([
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=3, sigma=0.5)
        ])
    }

    # Create visualization of individual augmentations
    fig, axes = plt.subplots(len(augmentations), 3, figsize=(12, 3 * len(augmentations)))
    fig.suptitle('Individual Augmentation Examples', fontsize=16, y=0.95)

    sample_idx = random.randint(0, len(dataset) - 1)
    orig_img, _ = dataset[sample_idx]

    for i, (aug_name, aug_transform) in enumerate(augmentations.items()):
        for j in range(3):
            if j == 0 and aug_name == 'Original':
                img = orig_img
            else:
                img = aug_transform(orig_img)

            axes[i, j].imshow(img.squeeze(), cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(f'{aug_name}')

    plt.tight_layout()
    plt.savefig('individual_augmentations.png')
    plt.close()


if __name__ == "__main__":
    # Create output directory if needed
    import os

    os.makedirs('visualizations', exist_ok=True)

    print("Generating augmentation visualizations...")
    visualize_augmentations()
    print("Visualizations saved as 'augmentation_examples.png' and 'individual_augmentations.png'")