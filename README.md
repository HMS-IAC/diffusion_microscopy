# Microscopy Image Generation with Diffusion Models

This project implements a deep learning pipeline for generating microscopy images using diffusion models.

## Project Structure

```
.
├── data/
│   ├── raw/           # Examples of original microscopy images
│   └── processed/     # Examples of preprocessed and augmented data
├── src/
│   ├── data/         # Data loading and preprocessing modules
│   ├── models/       # Model architecture definitions
│   ├── training/     # Training scripts and utilities
│   ├── utils/        # Helper functions and utilities
│   └── config/       # Configuration files
├── tests/            # Unit tests
├── notebooks/        # Jupyter notebooks for exploration
└── docs/            # Documentation
```

## Setup

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
uv pip install -r requirements.txt
```

## Usage

1. Data Preparation:
   - Fetch data using the IDR API 
   - Run preprocessing scripts from `src/data/`

2. Training:
   - Configure training parameters in `src/config/`
   - Run training scripts from `src/training/`

3. Evaluation:
   - Use notebooks in `notebooks/` for visualization and analysis
   - Run evaluation scripts from `src/training/`

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Other dependencies listed in `requirements.txt`

## License

[Your chosen license]

