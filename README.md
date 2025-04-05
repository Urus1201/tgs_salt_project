# TGS Salt Identification Challenge

This repository contains a deep learning solution for the TGS Salt Identification Challenge, which aims to segment salt deposits from seismic images.

## Project Structure
```
├── data/                    # Data directory (not tracked by git)
│   ├── train/              # Training dataset
│   ├── test/               # Test dataset
│   └── valid/              # Validation dataset
├── models/                  # Saved model checkpoints
├── src/                    # Source code
│   ├── dataset.py          # Dataset and data loading utilities
│   ├── inference.py        # Inference pipeline
│   ├── losses.py          # Custom loss functions
│   ├── mean_teacher.py     # Mean Teacher semi-supervised learning implementation
│   ├── model.py           # Model architecture
│   ├── prepare_data.py     # Data preparation utilities
│   ├── train.py           # Training script
│   └── transforms.py       # Data augmentation and transforms
└── logs/                   # Training logs and TensorBoard files
```

## Requirements
- Python 3.8+
- PyTorch 2.0+
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/tgs_salt_project.git
cd tgs_salt_project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preparation:
```bash
python src/prepare_data.py
```

2. Training:
```bash
python src/train.py
```

3. Inference:
```bash
python src/inference.py
```

## Model Architecture

This project implements a semi-supervised learning approach using the Mean Teacher model for salt deposit segmentation. The architecture includes:
- A U-Net based segmentation network
- Mean Teacher semi-supervised learning framework
- Custom loss functions for improved segmentation

## License

[Choose an appropriate license]

## Acknowledgments

- TGS Salt Identification Challenge on Kaggle
- [Add any other acknowledgments]