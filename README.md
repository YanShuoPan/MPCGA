# MPCGA: Multi-Path Cut Generation Algorithm

A high-dimensional variable selection algorithm using multi-path exploration and automatic cut point generation.

## Overview

MPCGA (Multi-Path Cut Generation Algorithm) is a variable selection method designed for high-dimensional data. It extends traditional CGA (Cut Generation Algorithm) by:

- **Multi-path exploration**: Explores multiple candidate paths simultaneously instead of greedy single-path selection
- **Automatic cut point generation**: Discovers nonlinear effects through binary indicator variables
- **Multiple penalty criteria**: Supports HDAIC, HDBIC for model selection
- **Model trimming (MTrim)**: Removes redundant paths while preserving prediction accuracy

## Features

- Binary classification (DGP1-3)
- Multinomial classification (DGP4-5)
- Handles linear and nonlinear effects
- Efficient implementation with precomputed sorting
- Parallel processing support

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- joblib >= 1.0.0

Optional (for baseline comparisons):
- xgboost >= 1.5.0

## Quick Start

### Basic Usage

```python
from mpcga_algorithm.mpcga_while import fit_model_while
import numpy as np

# Generate data
X_train = np.random.randn(300, 1000)
y_train = np.random.randint(0, 2, 300)
X_test = np.random.randn(100, 1000)

# Fit MPCGA model
K = 25  # number of steps
models = fit_model_while(
    X_train, y_train,
    K=K,
    max_set=5,           # max candidates per step
    import_threshold=0.7,
    max_split=3,
    c3=1.0,              # HDBIC penalty coefficient
    penalty_type='HDBIC',
    use_mtrim=False
)

# Get predictions
from mpcga_algorithm.model import get_result
predictions = get_result(X_train, np.zeros(len(X_test)), X_test, models)
```

### Using MTrim

```python
from mpcga_algorithm.mpcga import Model_Trim

# Apply MTrim to MPCGA paths
hdbic_paths = models['path']
trimmed_paths = Model_Trim(X_train, y_train, hdbic_paths, c2=3.0)
```

## Project Structure

```
MPCGA_clean/
├── mpcga_algorithm/         # Core algorithm
│   ├── mpcga_while.py       # ⭐ Main MPCGA algorithm (RECOMMENDED)
│   ├── mpcga.py             # HDIC_Trim, Model_Trim functions
│   ├── cga.py               # Traditional CGA baseline
│   ├── model.py             # Prediction functions
│   ├── cut_generation_optimized.py  # Cut point generation
│   ├── utils.py             # Utility functions
│   ├── distributions.py     # [OPTIONAL] Distribution classes for extensibility
│   └── mpcga_dist.py        # [OPTIONAL] OOP wrapper (calls mpcga_while internally)
│
├── simulations/             # Simulation studies
│   ├── sim_dgp123_mpcga.py      # DGP1-3 MPCGA methods
│   ├── sim_dgp123_baseline.py   # DGP1-3 baseline methods
│   ├── sim_dgp45_mpcga.py       # DGP4-5 MPCGA methods (multinomial)
│   └── sim_dgp45_baseline.py    # DGP4-5 baseline methods
│
├── data_generation.py       # Data generating processes
├── evaluation_metrics.py    # Evaluation metrics
├── README.md                # This file
└── requirements.txt         # Package dependencies
```

### Notes on Optional Modules

**distributions.py** and **mpcga_dist.py** are optional modules:
- They provide an object-oriented interface for custom distributions
- Internally, they call `mpcga_while.py` with identical results
- **Not used** by any simulation scripts (all use `mpcga_while.py` directly)
- Kept for potential future extensibility (e.g., custom distribution families)
- See module docstrings for detailed usage examples

## Methods

### MPCGA Methods

1. **CGA+HDBIC**: Traditional single-path CGA
2. **MPCGA+HDBIC**: Multi-path with HDBIC penalty
3. **MPCGA+HDAIC(OP)**: One-pass greedy variant
4. **MPCGA+HDBIC+MTrim**: MPCGA with model trimming
5. **MPCGA+RF/XGB**: Ensemble methods on MPCGA-selected variables

### Baseline Methods

1. **Lasso**: L1-penalized logistic regression
2. **Adaptive Lasso**: Weighted L1 penalty
3. **Random Forest (RF)**: With and without Boruta selection
4. **XGBoost (XGB)**: With and without Boruta selection

## Running Simulations

### DGP1-3 (Binary Classification)

```bash
# MPCGA methods
python simulations/sim_dgp123_mpcga.py

# Baseline methods
python simulations/sim_dgp123_baseline.py
```

### DGP4-5 (Multinomial Classification)

```bash
# MPCGA methods
python simulations/sim_dgp45_mpcga.py

# Baseline methods
python simulations/sim_dgp45_baseline.py
```

## Data Generating Processes

### DGP1 (Binary, Linear)
- 5 true variables with linear effects
- p = 600 or 1000 total variables
- n = 300 or 600 samples

### DGP2 (Binary, Cut + Linear)
- 4 true variables: 2 with cut effects, 2 linear
- Indicators: I{|x1| > 0.5}, I{|x2| > 0.5}

### DGP3 (Binary, Quadratic + Cut)
- 4 true variables: 2 quadratic, 2 cut
- x1^2, x2^2, I{x3 > 0.5}, I{x4 > 0.5}

### DGP4-5 (Multinomial)
- 3-class classification
- Similar structure to DGP2-3

## Performance Tuning

### Key Parameters

- **K**: Path length (number of variables to select)
  - Recommended: `K = int(3 * sqrt(n / log(p)))`

- **c3**: HDBIC penalty coefficient
  - Larger c3 → stronger penalty → fewer variables
  - Recommended: 0.8 - 1.0

- **c2**: MTrim tuning parameter
  - Controls trade-off between model size and loss
  - Recommended: 1.0 - 3.0

- **max_set**: Max candidates per step
  - More candidates → better exploration but slower
  - Recommended: 3 - 5

- **n_jobs**: Parallel processing
  - Number of CPU cores to use
  - Recommended: 8 - 16

## Citation

If you use this code in your research, please cite:

```
[Your paper citation here]
```

## License

[Add your license here - e.g., MIT, Apache 2.0]

## Contact

[Your contact information]
