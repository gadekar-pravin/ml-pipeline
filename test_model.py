# test_model.py
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import OptimizedCNN
import pytest
import glob
import os
import json
from pathlib import Path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_architecture():
    """Test model parameters are within limits"""
    model = OptimizedCNN()
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params}")
    assert num_params < 25000, f"Model has {num_params} parameters, should be less than 25,000"


def test_model_shape():
    """Test model handles correct input/output shapes"""
    model = OptimizedCNN()
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"
    except Exception as e:
        pytest.fail(f"Model failed to process 28x28 input: {str(e)}")


def test_training_requirements():
    """Test training met all requirements (1 epoch, >95% accuracy)"""
    results_path = 'metrics/training_results.json'

    # Check if results file exists
    if not os.path.exists(results_path):
        pytest.skip("No training results found. Run training first.")

    with open(results_path, 'r') as f:
        results = json.load(f)

    # Test requirements
    assert results['epochs'] == 1, "Model must be trained for exactly 1 epoch"
    assert results['training_accuracy'] > 95, \
        f"Training accuracy {results['training_accuracy']}% is below required 95%"
    assert results['parameters'] < 25000, \
        f"Model has {results['parameters']} parameters, exceeding limit of 25,000"


def test_model_inference():
    """Test model performance on test set"""
    # Skip if no model file exists
    model_files = glob.glob('model_*.pth')
    if not model_files:
        pytest.skip("No trained model found. Run training first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptimizedCNN().to(device)

    # Load the latest model
    latest_model = max(model_files)
    model.load_state_dict(torch.load(latest_model, weights_only=True))

    # Prepare test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

    # Evaluate
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test accuracy: {test_accuracy}%")


if __name__ == "__main__":
    pytest.main([__file__])