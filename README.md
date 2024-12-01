
[![CI/CD](https://github.com/gadekar-pravin/ml-pipeline/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/gadekar-pravin/ml-pipeline/actions/workflows/ml-pipeline.yml)

# ML Pipeline with CI/CD - Optimized MNIST Classification

This project implements a highly optimized Convolutional Neural Network (CNN) for MNIST digit classification, achieving >95% accuracy in a single epoch while maintaining a parameter count below 25,000. The project includes a complete CI/CD pipeline using GitHub Actions.

## Project Structure

```
ml-pipeline/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
├── data/               # MNIST dataset storage
├── model.py           # Optimized CNN architecture
├── train.py           # Training script with OneCycleLR policy
├── test_model.py      # Comprehensive testing suite
├── visualize_augmentations.py  # Data augmentation visualization
├── requirements.txt   # Dependencies
└── README.md          # Project documentation
```

## Model Architecture

The project uses an optimized CNN architecture (`OptimizedCNN`) designed for efficient learning:

- **Input Layer**: 1x28x28 (MNIST image size)
- **First Convolutional Block**:
  - Conv2D: 20 filters, 3x3 kernel, stride 1, padding 1
  - Batch Normalization
  - ReLU Activation
  - Max Pooling (2x2)
- **Second Convolutional Block**:
  - Conv2D: 16 filters, 3x3 kernel, stride 1, padding 1
  - Batch Normalization
  - ReLU Activation
  - Max Pooling (2x2)
- **Output Layer**:
  - Flatten
  - Dropout (0.2)
  - Fully Connected (784 → 10)
  - Log Softmax

Total Parameters: ~24,000 (well under the 25,000 limit)

## Training Optimizations

The training process incorporates several optimizations to achieve high accuracy in a single epoch:

1. **Learning Rate Schedule**:
   - OneCycleLR policy
   - max_lr: 0.025
   - 50% warmup
   - Div factor: 3
   - Final div factor: 10

2. **Optimizer Configuration**:
   - AdamW optimizer
   - Learning rate: 0.001
   - Betas: (0.9, 0.99)
   - Weight decay: 0

3. **Data Loading**:
   - Batch size: 32 (optimized for CPU training)
   - 2 worker processes
   - Pin memory disabled for CPU optimization

4. **Data Augmentation**:
   - Rotation (-10° to 10°)
   - Normalization (mean=0.1307, std=0.3081)

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pytest
- numpy
- tqdm
- matplotlib
- idx2numpy

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

The script will:
1. Train for exactly 1 epoch
2. Save model checkpoints with timestamp and accuracy
3. Generate training metrics in JSON format
4. Validate against the test set

Training artifacts are saved in:
- Models: `models/model_YYYYMMDD_HHMMSS_accXX.X.pth`
- Metrics: `metrics/training_results.json`

## Visualization

Run the augmentation visualization script:
```bash
python visualize_augmentations.py
```

This will display:
- Original vs augmented image comparisons
- Multiple augmentations of the same image

## Testing

Run the comprehensive test suite:
```bash
pytest test_model.py -v
```

Tests verify:
1. Model parameter count (< 25,000)
2. Input/output shapes (28x28 → 10 classes)
3. Training requirements (1 epoch, >95% accuracy)
4. Model inference on test set

## CI/CD Pipeline

The GitHub Actions workflow (`ml-pipeline.yml`) automates:
1. Environment setup
2. Dependency installation
3. Model training
4. Test execution
5. Artifact storage

Artifacts (trained models and metrics) are automatically uploaded to GitHub Actions.

## Performance Metrics

Typical performance metrics:
- Training accuracy: >95% (single epoch)
- Parameters: ~24,000
- Training time: ~5-10 minutes (CPU)
- Memory usage: <2GB

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
