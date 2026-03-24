# Changelog

All notable changes to this project will be documented in this file.

---

## [2026-03-24] - Simplification for Paper

### Overview
Simplified simulation parameters to match the paper settings: **c3=0.7** (HDBIC penalty) and **c2=3** (MTrim tolerance).

### Changed

#### Parameter Simplification
- **c3**: Unified to single value `0.7` (was testing both 0.8 and 1.0)
- **c2**: Unified to single value `3.0` (was testing both 3.0 and 10.0)
- Removed all `DEFAULT_C3_LOW/HIGH`, `DEFAULT_C2_LOW/HIGH` constants

#### Method Reduction
- **DGP1-3**: 10 methods -> 7 methods (True Model + CGA+HDBIC + 5 MPCGA variants)
- **DGP4-5**: 15 methods -> 6 methods (True Model + 5 MPCGA variants)
- Final method set: MPCGA+HDBIC, MPCGA-S, MPCGA+HDBIC+MTrim, MPCGA+RF, MPCGA+XGB

#### Code Quality
- Unified `cut_generation_optimized.py` as the sole cut generation module (removed dependency on `cut_generation.py`)
- Fixed True Model prediction for binary DGPs (1D probability handling)
- Updated all documentation to reflect simplified parameters

---

## [2026-01-19] - Major Refactoring Update

### 🎯 Overview
Major refactoring of simulation files to improve code quality, maintainability, and performance.

### ✨ Added

#### Global Parameter Configuration
- **DGP1-3** ([sim_dgp123_mpcga.py](simulations/sim_dgp123_mpcga.py)):
  - Added comprehensive global parameter configuration section
  - All parameters now centralized at the top of the file
  - Easy to modify: `DEFAULT_C3_LOW`, `DEFAULT_C3_HIGH`, `DEFAULT_C2`, etc.

- **DGP4-5** ([sim_dgp45_mpcga.py](simulations/sim_dgp45_mpcga.py)):
  - Added global parameter configuration section
  - Ensemble parameters now use grid search (was single values)
  - Cross-validation parameters added: `DEFAULT_CV_FOLDS`, `DEFAULT_CV_N_ITER`

#### Two-Stage Optimization for DGP4-5
- **New functions**:
  - `run_mpcga_once()`: Generate candidate paths once
  - `apply_hdbic_trim()`: Apply different c3 values to pre-generated paths
- **Performance**: ~50% time reduction when testing multiple c3 values
- **Expanded methods**: 8 methods → 15 methods
  - Now tests both c3=0.8 and c3=1.0 (was only c3=0.8)
  - All combinations of (c3, c2) parameters

#### Enhanced Method Coverage
- **DGP4-5 new methods**:
  - `MPCGA+HDBIC(c3=1.0)` - new c3 value
  - `MPCGA+HDBIC+MTrim(c3=1.0,c2=10)` - new combination
  - `MPCGA+HDBIC+MTrim(c3=1.0,c2=3)` - new combination
  - `MPCGA+RF(c3=1.0,c2=10)` - new combination
  - `MPCGA+XGB(c3=1.0,c2=10)` - new combination
  - `MPCGA+RF(c3=1.0,c2=3)` - new combination
  - `MPCGA+XGB(c3=1.0,c2=3)` - new combination

### 🔧 Changed

#### Parameter Management
- **Before**: Parameters hard-coded throughout functions
- **After**: All parameters use global constants
- **Impact**: Easy to modify and maintain

#### Ensemble Methods (DGP4-5)
- **Before**:
  ```python
  DEFAULT_RF_N_ESTIMATORS = 100  # Single value
  DEFAULT_RF_MAX_DEPTH = 5
  ```
- **After**:
  ```python
  DEFAULT_RF_N_ESTIMATORS = [50, 100, 150]  # Grid search
  DEFAULT_RF_MAX_DEPTH = [10, 30, 50]
  DEFAULT_RF_MIN_SAMPLES_SPLIT = [2, 5, 10]
  ```
- **Impact**: Better hyperparameter tuning via RandomizedSearchCV

#### Code Structure
- **DGP1-3 and DGP4-5**: Now use identical design patterns
- **Function signatures**: Consistent parameter naming and defaults
- **Documentation**: Enhanced docstrings with parameter descriptions

### 📊 Performance Improvements

#### Computational Efficiency
- **DGP4-5 Two-stage optimization**: Save ~50% computation time
- **Intelligent caching**: HDBIC and MTrim results reused
- **No redundant calculations**: Same MPCGA output used for multiple c3 values

#### Code Quality
- **Magic numbers eliminated**: 0 hard-coded values
- **Centralized configuration**: All parameters at file top
- **Consistency**: DGP1-3 and DGP4-5 aligned
- **Maintainability**: Single point of modification

### 📁 Files Updated

#### Simulation Files
- ✅ `simulations/sim_dgp123_mpcga.py` - Global parameters + code cleanup
- ✅ `simulations/sim_dgp45_mpcga.py` - Two-stage optimization + global parameters
- ✅ `simulations/sim_dgp123_baseline.py` - Latest baseline methods
- ✅ `simulations/sim_dgp45_baseline.py` - Latest baseline methods

### 🎯 Parameter Alignment

| Parameter | DGP1-3 | DGP4-5 | Status |
|-----------|--------|--------|--------|
| HDBIC c3 | 0.8, 1.0 | 0.8, 1.0 | ✅ Aligned |
| Max Set | 5 | 5 | ✅ Aligned |
| Max Split | 3 | 5 | ⚠️ Different (by design) |
| MTrim c2 | 3.0 | 3.0, 10.0 | ⚠️ Different (DGP4-5 tests more) |
| Ensemble params | Grid + CV | Grid + CV | ✅ Aligned |
| CV Folds | 5 | 5 | ✅ Aligned |

### 🧪 Testing

All changes have been tested and verified:
- ✅ DGP1-3 global parameters working correctly
- ✅ DGP4-5 two-stage optimization working correctly
- ✅ All 15 methods in DGP4-5 producing results
- ✅ Ensemble grid search + CV functioning properly

### 📚 Documentation

Additional documentation files (in development folder):
- `GLOBAL_PARAMS_SUMMARY.md` - Detailed parameter comparison
- `REFACTORING_SUMMARY.md` - Complete refactoring report

### 🎓 Usage Example

#### Modifying Parameters
```python
# Edit sim_dgp45_mpcga.py or sim_dgp123_mpcga.py

# Change HDBIC penalty
DEFAULT_C3 = 1.0  # Stricter penalty (fewer variables)

# Increase candidate paths
DEFAULT_MAX_SET = 10  # More exploration

# Adjust MTrim
DEFAULT_C2_HIGH = 15.0  # Stricter trimming
```

#### Running Simulations
```bash
# Run DGP1-3 simulation (10 methods)
python simulations/sim_dgp123_mpcga.py

# Run DGP4-5 simulation (15 methods)
python simulations/sim_dgp45_mpcga.py

# Run baseline methods
python simulations/sim_dgp123_baseline.py
python simulations/sim_dgp45_baseline.py
```

### 🔮 Future Improvements

Potential enhancements for future versions:
- [ ] Unified `global_config.py` for all simulations
- [ ] Command-line parameter overrides
- [ ] YAML/JSON configuration file support
- [ ] Parameter validation and range checking
- [ ] Automated parameter tuning suggestions

---

## Version History

### [2026-03-24] v2.1.0
- Simplified parameters to paper settings (c3=0.7, c2=3)
- Reduced methods to final paper set
- Unified cut generation module

### [2026-01-19] v2.0.0
- Major refactoring with global parameters and two-stage optimization
- 15 MPCGA methods for DGP4-5 (up from 8)
- Complete code alignment between DGP1-3 and DGP4-5

### [2026-01-14] v1.1.0
- Initial MPCGA methods implementation
- 8 MPCGA methods for DGP4-5
- 10 MPCGA methods for DGP1-3

### [2026-01-03] v1.0.0
- Initial release
- Basic MPCGA algorithm implementation
- Baseline comparison methods
