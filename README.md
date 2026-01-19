# MPCGA: Multipath Chebyshev Greedy Algorithm

A tree-based extension of the Chebyshev Greedy Algorithm for high-dimensional prediction and feature selection.

## Overview

**MPCGA (Multipath Chebyshev Greedy Algorithm)** is a feature selection method designed for high-dimensional classification problems. It extends the traditional Chebyshev Greedy Algorithm (CGA) by exploring multiple greedy paths in parallel, integrating a high-dimensional information criterion (HDIC), and applying model trimming to balance predictive accuracy and interpretability.

### Key Features

- **Multi-path exploration**: Explores multiple candidate paths simultaneously instead of greedy single-path selection
- **Automatic cut point generation**: Discovers nonlinear effects through binary indicator variables
- **Model trimming (MTrim)**: Removes redundant paths while preserving prediction accuracy
- **Multiple penalty criteria**: Supports HDAIC, HDBIC for model selection
- **Fast split-finding algorithm**: Efficiently identifies optimal cut points without exhaustive enumeration
- **Handles binary and multinomial outcomes**: Works for both binary and multi-class classification

---

## Methodology

### Background: CGA+HDIC

Given a dataset with $n$ observations and $p$ features, CGA sequentially selects features by maximizing the absolute gradient of the empirical loss function:

$$\ell(\beta) := \frac{1}{n}\sum_{t=1}^n \gamma(\beta, y_t, x_t)$$

where $\gamma(\cdot)$ is a differentiable convex loss function (e.g., logistic loss for binary outcomes).

**CGA Algorithm:**
1. Start with empty feature set $J_0 = \varnothing$
2. At each step $m$, select feature $j_m = \arg\max_{1\leq j\leq p}|\nabla_j\ell(\hat{\beta}_{J_{m-1}})|$
3. Update $J_m = J_{m-1} \cup \{j_m\}$ and fit $\hat{\beta}_{J_m}$
4. Choose optimal model size via HDIC:
   $$\text{HDIC}(J) = \ell(\hat{\beta}_J) + |J|c_1\omega_n\frac{\log p}{n}$$
   where $\omega_n = \log n$ (HDBIC) or $\omega_n = 2$ (HDAIC).

### MPCGA Extensions

MPCGA improves upon CGA through three main steps:

#### **Step 1: Recursive Branching**

Instead of selecting only the single largest gradient, MPCGA considers **all features whose gradients exceed a ratio threshold $r$**:

- At each node, compute gradient scores $s_j = |\nabla_j \ell(\hat{\beta}_J)|$
- Select candidates: $\Omega = \{j : s_j \geq r \cdot \max_j s_j\}$
- Each selected feature spawns a new branch
- Continue until path length reaches $K$

Setting $r = 1$ recovers the original CGA. Lower values of $r$ create more diverse paths.

**Duplicate path pruning**: Since the order of feature inclusion doesn't affect subsequent gradients, paths with identical feature sets are eliminated early.

#### **Step 2: Model Refinement via HDIC**

Each path $l$ generates a sequence of nested feature sets:
$$J_1^l \subset J_2^l \subset \cdots \subset J_K^l$$

For each path, select the optimal model size:
$$\hat{k}_l = \arg\min_{1\leq k\leq K}\text{HDIC}(J_k^l)$$

This produces a refined collection of candidate models.

#### **Step 3: Model Trimming (MTrim)**

To reduce redundancy, remove paths that are clearly inferior in terms of loss and model size:

1. Find the best model: $J^* = \arg\min_J \ell(\hat{\beta}_J)$ with $\ell_{\min} = \ell(\hat{\beta}_{J^*})$
2. Retain candidate $m$ only if:
   $$\ell(\hat{\beta}_{J_m}) - \ell_{\min} \leq c_2 \cdot \max(1, |J^*| - |J_m|)$$

where $c_2$ is a tuning parameter.

**Intuition**: If a model uses more features than $J^*$, it must achieve sufficiently lower loss. If it uses fewer features, a larger tolerance is allowed.

---

## Split-Finding Algorithm for Discrete Outcomes

### Binary Outcomes

For binary outcomes $y_t \in \{0,1\}$, MPCGA expands each continuous feature into indicator features to capture nonlinear effects.

For feature $j$, define indicator features at each cut position $i$:
$$I_j^{(i)} = \mathbb{I}\{X_j \leq x_{(i),j}\}$$

where $x_{(i),j}$ is the $i$-th order statistic.

**Key insight**: Instead of materializing all $(n-1)p$ indicators, MPCGA uses a **fast split-finding routine** that:
1. Sorts each feature once: $O(n\log n)$
2. Scans cumulative gradients to find optimal cut: $O(n)$
3. Selects at most one best indicator per feature per iteration

### Multinomial Outcomes

For $K+1$ class outcomes, the multinomial logistic objective is:
$$\ell_K(\beta_1,\ldots,\beta_K) = \frac{1}{n}\sum_{t=1}^n \left[\sum_{k=1}^K \mathbb{I}(y_t=k)\beta_k^\top x_t - \log\left(1+\sum_{k=1}^K e^{\beta_k^\top x_t}\right)\right]$$

The split-finding algorithm applies class-wise, yielding at most one best indicator per feature per class.

---

## Algorithm Implementation Details

### Computational Control Mechanisms

A practical challenge of MPCGA is the rapid growth of candidate paths. Two complementary mechanisms control complexity:

| Mechanism | Parameter | Description | Impact |
|-----------|-----------|-------------|--------|
| **Budgeted Expansion** | `k_max` | Maximum expansion depth (iterations) | Bounds total paths: $M_{\max} = L_c^{k_{\max}}$ |
| | `L_c` (`max_set`) | Cap on new branches per iteration | Limits branching factor |
| **Duplicate Pruning** | `path_exist` | Global record of visited feature sets | Eliminates redundant computations |

**Key insight**: Since feature inclusion order doesn't affect gradients, different sequences yielding identical feature sets are pruned immediately.

### Complete MPCGA Workflow

The full procedure integrates all components into a unified framework:

```
┌─────────────────────────────────────────────────────────────┐
│  MPCGA Complete Procedure                                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  Step 1: Multipath Search + SF  │
        │  • Initialize: J₀ = {intercept} │
        │  • While paths exist:           │
        │    - Fit model on current path  │
        │    - Apply SF for indicators    │
        │    - Compute gradients          │
        │    - Branch on top features     │
        │    - Prune duplicates           │
        │  • Output: Path collection 𝒫    │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  Step 2: HDIC Refinement        │
        │  For each path l in 𝒫:          │
        │    k̂ₗ = argmin HDIC(Jₖˡ)        │
        │  • Output: Refined models ℋ     │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  Step 3: Model Trimming (MTrim) │
        │  • Find best: J* = argmin ℓ(β̂)  │
        │  • Retain only if:              │
        │    ℓ(β̂ⱼ) - ℓ_min ≤ c₂·Δ|J|      │
        │  • Output: Final models 𝓜       │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  Prediction                     │
        │  • Voting: mode{V̂ᵢ}            │
        │  • Or probability averaging     │
        └─────────────────────────────────┘
```

### Prediction Strategies

After obtaining final models 𝓜 = {M₁, ..., Mₕ}, MPCGA offers two prediction strategies:

<table>
<tr>
<th>Strategy</th>
<th>Formula</th>
<th>Use Case</th>
</tr>
<tr>
<td><b>Unweighted Voting</b></td>
<td>

```
V̂ₜᵢ = argmax p̂ₜᵢₖ  (per model)
      k
V̂ₜ = mode{V̂ₜ₁, ..., V̂ₜₕ}
```

</td>
<td>Default; robust to outlier models</td>
</tr>
<tr>
<td><b>Probability Averaging</b></td>
<td>

```
p̄ₜₖ = (1/h)Σᵢ p̂ₜᵢₖ
V̂ₜ = argmax p̄ₜₖ
      k
```

</td>
<td>Smoother predictions; better calibration</td>
</tr>
</table>

### Complete Algorithm Pseudocode

```python
"""
MPCGA: Complete Procedure
Inputs:
  X         : Feature matrix (n × p)
  y         : Response vector
  K         : Path length
  r         : Ratio threshold
  k_max     : Max expansion depth
  L_c       : Max branches per iteration
  c1, c2    : Penalty coefficients

Outputs:
  M         : Final model collection
"""

# Step 1: Multipath Search with Split-Finding
path_exist = {}  # Global duplicate tracker
S = [(J0={intercept}, X)]  # Initialize queue
P = []  # Path collection

while S is not empty and depth < k_max:
    (J, X_all) = S.pop()

    # Fit current model
    β̂_J = fit_logistic(X_all, y, active_set=J)

    # Generate indicators via Split-Finding
    for j in 1..p:
        I_j = SplitFinding(X_j, β̂_J, X_all)
        X_all = X_all ∪ {I_j}

    # Compute gradients for all features (original + indicators)
    s_c = |∇_c ℓ(β̂_J | X_all)| for all c in X_all
    M = max(s_c)
    Ω = {c : s_c ≥ r·M}

    # Branch on top-L_c candidates
    Ω_top = top_L_c(Ω)

    for c in Ω_top:
        J_new = J ∪ {c}

        if J_new in path_exist:
            continue  # Skip duplicate
        else:
            path_exist[J_new] = True

        if |J_new| == K:
            P.append(J_new)  # Complete path
        else:
            S.append((J_new, X_all))  # Continue expansion

# Step 2: HDIC Refinement
H = []
for path_l in P:
    # Each path has nested models: J₁ˡ ⊂ J₂ˡ ⊂ ... ⊂ Jₖˡ
    k̂_l = argmin_{1≤k≤K} HDIC(J_k^l)
    H.append(J_{k̂_l}^l)

# Step 3: Model Trimming (MTrim)
J_star = argmin_{J∈H} ℓ(β̂_J)
ℓ_min = ℓ(β̂_{J_star})
L_min = |J_star|

M = []
for J in H:
    Δloss = ℓ(β̂_J) - ℓ_min
    Δsize = max(1, L_min - |J|)

    if Δloss ≤ c2 · Δsize:
        M.append(J)

return M
```

### MPCGA + Machine Learning Integration

MPCGA can serve as a feature preselector for ML methods (Random Forest, XGBoost):

```
MPCGA (original + indicator features)
         │
         ▼
   [Feature Consolidation]
   • If indicator I{Xⱼ ≤ c} selected → retain original Xⱼ
   • Union all selected originals across paths
         │
         ▼
   Consolidated Feature Set
         │
         ▼
   ML Model (RF/XGB)
   • ML automatically generates splits
   • MPCGA provides relevant feature subset
         │
         ▼
   Enhanced Predictions
```

**Rationale**: ML methods can automatically discover cut points, so we only pass original features identified as important (via their indicators).

---

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

---

## 🆕 Latest Updates (2026-01-19)

### Version 2.0.0 - Major Refactoring

**Key Improvements**:
- ✅ **Global Parameter Configuration**: All simulation parameters centralized for easy modification
- ✅ **Two-Stage Optimization** (DGP4-5): ~50% faster when testing multiple c3 values
- ✅ **Expanded Methods** (DGP4-5): 8 → 15 methods (now tests both c3=0.8 and c3=1.0)
- ✅ **Enhanced Ensemble Methods**: Grid search + cross-validation for better hyperparameter tuning
- ✅ **Complete Code Alignment**: DGP1-3 and DGP4-5 now use identical design patterns

**See [CHANGELOG.md](CHANGELOG.md) for details.**

---

## Quick Start

### Basic Usage

```python
from mpcga_algorithm.mpcga_while import fit_model_while
import numpy as np

# Generate sample data
X_train = np.random.randn(300, 1000)
y_train = np.random.randint(0, 2, 300)
X_test = np.random.randn(100, 1000)

# Fit MPCGA model
K = 25  # number of steps
models = fit_model_while(
    X_train, y_train,
    K=K,
    max_set=5,           # max candidates per step
    import_threshold=0.7, # ratio threshold r
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

---

## Key Parameters

### Algorithm Parameters

- **K**: Path length (number of iterations)
  - Recommended: $K = 3\sqrt{n/\log p}$

- **r** (`import_threshold`): Gradient ratio threshold
  - Controls branching diversity
  - Recommended: 0.7 - 0.8

- **c3** (c1): HDBIC penalty coefficient
  - Default: 0.8 - 1.0
  - Higher values → more penalized → fewer variables

- **c2**: MTrim tuning parameter
  - Controls trade-off between model size and loss
  - Recommended: 3.0 - 10.0

- **max_set**: Max candidates per step
  - Limits computational complexity
  - Recommended: 3 - 5

### Global Parameter Configuration (NEW in v2.0)

All simulation scripts now use **centralized global parameters** for easy tuning:

```python
# Example: Modifying parameters in sim_dgp45_mpcga.py

# HDBIC parameters
DEFAULT_C3 = 0.8           # Change penalty coefficient
DEFAULT_MAX_SET = 5        # Change max candidate paths
DEFAULT_MAX_SPLIT = 5      # Change max splits

# MTrim parameters
DEFAULT_C2_HIGH = 10.0     # Stricter trimming
DEFAULT_C2_LOW = 3.0       # Moderate trimming

# Ensemble parameters (grid search)
DEFAULT_RF_N_ESTIMATORS = [50, 100, 150]
DEFAULT_RF_MAX_DEPTH = [10, 30, 50]
DEFAULT_XGB_LEARNING_RATE = [0.2, 0.4, 0.6]

# Cross-validation
DEFAULT_CV_FOLDS = 5
DEFAULT_CV_N_ITER = 8
```

**Location**: All parameters are defined at the top of each simulation file for easy modification.

---

## Simulation Studies

The repository includes simulation scripts for various data generating processes:

### Binary Classification (DGP1-3)

- **DGP1**: Linear effects only
- **DGP2**: Indicator features (cut effects)
- **DGP3**: Quadratic + indicator features (misspecified)

### Multinomial Classification (DGP4-5)

- **DGP4**: 3-class with linear effects
- **DGP5**: 3-class with quadratic + indicator features

### Running Simulations

```bash
# Binary classification
python simulations/sim_dgp123_mpcga.py
python simulations/sim_dgp123_baseline.py

# Multinomial classification
python simulations/sim_dgp45_mpcga.py
python simulations/sim_dgp45_baseline.py
```

---

## Project Structure

```
MPCGA_clean/
├── mpcga_algorithm/              # Core algorithm
│   ├── mpcga_while.py           # ⭐ Main MPCGA algorithm
│   ├── mpcga.py                 # HDIC_Trim, Model_Trim functions
│   ├── cga.py                   # Traditional CGA baseline
│   ├── model.py                 # Prediction functions
│   ├── cut_generation_optimized.py  # Fast split-finding
│   ├── utils.py                 # Utility functions
│   ├── distributions.py         # [OPTIONAL] Distribution classes
│   └── mpcga_dist.py            # [OPTIONAL] OOP wrapper
│
├── simulations/                  # Simulation studies
│   ├── sim_dgp123_mpcga.py      # DGP1-3 MPCGA methods
│   ├── sim_dgp123_baseline.py   # DGP1-3 baseline methods
│   ├── sim_dgp45_mpcga.py       # DGP4-5 MPCGA methods
│   └── sim_dgp45_baseline.py    # DGP4-5 baseline methods
│
├── data_generation.py            # Data generating processes
├── evaluation_metrics.py         # Evaluation metrics
├── README.md                     # This file
└── requirements.txt              # Dependencies
```

---

## Performance Summary

### When to Use MPCGA

✅ **Well-specified linear models** (DGP1): Competitive with CGA+HDBIC, outperforms ML methods

✅ **Models with indicator features** (DGP2): Superior to methods limited to original features (CGA, Lasso)

✅ **Misspecified models** (DGP3): MPCGA+ML achieves best performance with parsimonious feature sets

✅ **Multi-class problems** (DGP4-5): Precise feature selection, enhances downstream ML methods

✅ **Real applications**: Compact feature sets suitable for practical constraints (e.g., sensor development)

---

## Citation

If you use this code in your research, please cite:

```
Yan-Shuo Pan, Ching-Kang Ing, Guan-Hua Huang (2025).
MPCGA: A Tree-Based Chebyshev's Greedy Algorithm.
[Manuscript in preparation]
```

---

## Contact

- **Yan-Shuo Pan**: s107024502@m107.nthu.edu.tw
- **Ching-Kang Ing**: cking@stat.nthu.edu.tw

---

## License

[To be determined]