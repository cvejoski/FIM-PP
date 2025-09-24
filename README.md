# In-Context Learning of Temporal Point Processes with Foundation Inference Models

[![CI](https://github.com/anonymous/FIM/actions/workflows/ci.yml/badge.svg)](https://github.com/anonymous/FIM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/anonymous/FIM/blob/main/LICENSE.txt)

This repository contains the implementation of Foundation Inference Models for Temoral Point Processes (FIM-PP), a methodology for zero-shot inference of temporal point processes from noisy and sparse observations.

## Quick Start

The easiest way to get started is to explore the example notebook:

📓 **[Example Notebook](notebooks/example.ipynb)** - Complete walkthrough of loading data, training models, and performing inference

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd FIM-PP
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Usage

### Training

1. Create a configuration file (see `configs/train/example.yaml` for reference)

2. Run training in single-node mode:
   ```bash
   python scripts/train_model.py --config configs/train/example.yaml
   ```

3. For distributed training:
   ```bash
   torchrun --nproc_per_node=<number_of_gpus> scripts/train_model.py --config configs/train/example.yaml
   ```

### Inference

Use the inference script for model evaluation:
```bash
python scripts/inference.py --config configs/inference/example.yaml
```

## Key Features

- **Zero-shot inference** of Markov jump processes from sparse observations
- **Neural network models** trained on synthetic MJP datasets
- **Support for various dynamical systems** including:
  - Discrete flashing ratchet systems
  - Molecular simulation dynamics
  - Ion channel data
  - Protein folding models
- **Distributed training** support with PyTorch
- **Comprehensive evaluation** tools and metrics

## Configuration

The system uses YAML configuration files for all parameters. Key sections include:

- **Experiment**: Name, seed, device settings
- **Model**: Architecture and hyperparameters  
- **Data**: Dataset paths and preprocessing
- **Training**: Optimizers, schedulers, and training parameters

See `configs/` directory for example configurations.

## Project Structure

```
FIM-PP/
├── notebooks/           # Jupyter notebooks and examples
├── scripts/            # Training and inference scripts
├── configs/            # Configuration files
├── src/fim/           # Main package code
│   ├── models/        # Model implementations
│   ├── data/          # Data handling and datasets
│   ├── trainers/      # Training utilities
│   └── utils/         # Helper functions
└── tests/             # Test suite
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
ruff check src/
ruff format src/
```

## License

This project is licensed under the [MIT License](LICENSE.txt).

