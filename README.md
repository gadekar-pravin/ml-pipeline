# ML Pipeline with CI/CD

This project demonstrates a complete CI/CD pipeline for a machine learning project using PyTorch and GitHub Actions. It includes a simple CNN model trained on the MNIST dataset, with automated testing and deployment processes.

## Project Structure

```
ml-pipeline/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
├── data/               # MNIST dataset storage
├── model.py           # CNN architecture
├── train.py           # Training script
├── test_model.py      # Testing and validation
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Model Architecture

The project uses a simple CNN with:
- 2 convolutional layers
- 2 fully connected layers
- ReLU activation and max pooling
- Output size: 10 classes (digits 0-9)

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pytest
- numpy

## Local Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd ml-pipeline
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

Run the training script:
```bash
python train.py
```

The model will be saved with a timestamp: `model_YYYYMMDD_HHMMSS.pth`

## Testing

Run the tests:
```bash
pytest test_model.py -v
```

The tests verify:
1. Model has less than 100,000 parameters
2. Model correctly handles 28x28 input
3. Model outputs 10 classes
4. Model achieves >80% accuracy on test set

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Runs all tests
5. Saves the trained model as an artifact

The pipeline is triggered on every push to the repository.

## Model Artifacts

Trained models are saved with timestamps and uploaded as artifacts in GitHub Actions. You can download them from the Actions tab in your GitHub repository.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Notes

- The training is limited to 1 epoch for demonstration purposes
- The model architecture is intentionally kept simple
- MNIST dataset will be automatically downloaded when running the training script