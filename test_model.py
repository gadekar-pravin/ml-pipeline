
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import SimpleCNN
import pytest
import glob
import os


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_architecture():
    model = SimpleCNN()

    # Test 1: Check number of parameters
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params}")  # Added for debugging
    assert num_params < 100000, f"Model has {num_params} parameters, should be less than 100000"

    # Test 2: Check input shape handling
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"
    except Exception as e:
        pytest.fail(f"Model failed to process 28x28 input: {str(e)}")


def test_model_accuracy():
    # Skip accuracy test if no model file exists
    model_files = glob.glob('model_*.pth')
    if not model_files:
        pytest.skip("No trained model found. Run training first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    # Load the latest model using weights_only=True for security
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

    accuracy = 100 * correct / total
    print(f"Model accuracy: {accuracy}%")  # Added for debugging
    assert accuracy > 80, f"Model accuracy is {accuracy}%, should be > 80%"


if __name__ == "__main__":
    pytest.main([__file__])