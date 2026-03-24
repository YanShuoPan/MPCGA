"""
MPCGA Methods Simulation for DGP1-3 (Binary)
This file contains only MPCGA-related methods

使用方法:
  python simulations/sim_dgp123_mpcga.py

參數設定:
  - n_iterations: 100 (預設)
  - n_jobs: 4 (使用 4 個 CPU 核心)
  - verbose: 10 (顯示進度條)
  - 6 configurations: 3 DGPs × 2 sample sizes
  - USE_CORRELATED_FEATURES: False/True (是否使用相關性特徵)
"""

# ============================================================
# CONFIGURATION: 是否使用相關性特徵
# ============================================================
USE_CORRELATED_FEATURES = True  # 改成 True 會使用相關性特徵

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from data_generation import generate_data_dgp1, generate_data_dgp2, generate_data_dgp3
from mpcga_algorithm.mpcga_while import fit_model_while
from mpcga_algorithm.cga import fit_model_cga, predict_cga
from mpcga_algorithm.mpcga import Model_Trim
from mpcga_algorithm.model import get_result
from mpcga_algorithm.cut_generation_optimized import generate_test_cut_all
from evaluation_metrics import compute_metrics, summarize_metrics, print_metrics_summary
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time
from datetime import datetime
import warnings
# Suppress all warnings (including sklearn warnings in multiprocessing)
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] XGBoost not available, XGB methods will be skipped")


# ============================================================================
# GLOBAL PARAMETER CONFIGURATION
# ============================================================================
# Modify these parameters to change settings for all MPCGA methods

# HDBIC parameters
DEFAULT_C3 = 0.7           # HDBIC penalty coefficient
DEFAULT_MAX_SET = 3        # Maximum number of candidate paths at each step
DEFAULT_MAX_SPLIT = 3      # Maximum number of splits to explore
DEFAULT_IMPORT_THRESHOLD = 0.7  # Importance threshold for variable selection
DEFAULT_PENALTY_TYPE = 'HDBIC'  # Penalty type: 'HDBIC', 'HDAIC', or 'HDHQIC'

# MTrim parameters
DEFAULT_C2 = 3.0           # MTrim c2 for trimming (used in MPCGA+MTrim)

# Ensemble parameters
DEFAULT_RF_N_ESTIMATORS = [50, 100, 150]      # RF n_estimators grid
DEFAULT_RF_MAX_DEPTH = [10, 30, 50]           # RF max_depth grid
DEFAULT_RF_MIN_SAMPLES_SPLIT = [2, 5, 10]    # RF min_samples_split grid
DEFAULT_XGB_N_ESTIMATORS = [50, 100, 150]     # XGB n_estimators grid
DEFAULT_XGB_MAX_DEPTH = [10, 30, 50]          # XGB max_depth grid
DEFAULT_XGB_LEARNING_RATE = [0.2, 0.4, 0.6]  # XGB learning_rate grid

# Cross-validation parameters
DEFAULT_CV_FOLDS = 5       # Number of CV folds
DEFAULT_CV_N_ITER = 8      # Number of RandomizedSearchCV iterations
DEFAULT_RANDOM_STATE = 42  # Random state for reproducibility

# Regression type
DEFAULT_REGRESSION_TYPE = 'binary'  # 'binary' or 'multinomial'

# ============================================================================


def build_models_from_paths(X_df, Y, unique_paths, regression_type='binary'):
    """Build models from a list of unique paths

    Args:
        X_df: DataFrame with features
        Y: response vector (binary: 0/1)
        unique_paths: list of paths (variable names)
        regression_type: 'binary' (for DGP1-3)

    Returns:
        Dictionary with models, variables, and paths
    """
    models = []
    vars_list = []
    paths_list = []

    for path in unique_paths:
        all_varnames = [name for name in path if isinstance(name, str) and name != 'beta0']

        # If no variables selected (only intercept), create intercept-only model
        if len(all_varnames) == 0:
            X_new = pd.DataFrame({'int': np.ones(len(Y))})
            lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
            lr.fit(X_new, Y)
            models.append(lr)
            vars_list.append([])
            paths_list.append(path)
            continue

        # Build features in the exact order they appear in all_varnames
        X_new_cols = []

        for var_name in all_varnames:
            if 'cut' in var_name:
                # Generate this specific cut variable
                cut_df = generate_test_cut_all(X_df, [var_name], X_df)
                if not cut_df.empty and var_name in cut_df.columns:
                    X_new_cols.append(cut_df[[var_name]])
            else:
                # Add original variable
                if var_name in X_df.columns:
                    X_new_cols.append(X_df[[var_name]])

        # Concatenate all columns in order
        if len(X_new_cols) > 0:
            X_new = pd.concat(X_new_cols, axis=1)
        else:
            X_new = pd.DataFrame()

        X_new.insert(0, 'int', 1)

        lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
        lr.fit(X_new, Y)

        models.append(lr)
        vars_list.append(all_varnames)
        paths_list.append(path)

    return {'model': models, 'main_var': vars_list, 'path': paths_list}


def get_dgp_info(dgp_name):
    """Get DGP information"""
    if dgp_name == 'DGP1':
        return {
            'description': 'Binary, Linear Effects, 5 true variables',
            'n_true': 5,
            'true_vars': [0, 1, 2, 3, 4]  # V1-V5
        }
    elif dgp_name == 'DGP2':
        return {
            'description': 'Binary, Cut + Linear, 4 true variables',
            'n_true': 4,
            'true_vars': [0, 1, 2, 3]  # V1-V4 (I{|x1|>0.5}, I{|x2|>0.5}, x3, x4)
        }
    elif dgp_name == 'DGP3':
        return {
            'description': 'Binary, Quadratic + Cut, 4 true variables',
            'n_true': 4,
            'true_vars': [0, 1, 2, 3]  # V1-V4 (x1^2, x2^2, I{x3>0.5}, I{x4>0.5})
        }
    else:
        raise ValueError(f"Unknown DGP: {dgp_name}")


def run_mpcga_once(x_train, y_train, K, max_set=None):
    """Run MPCGA to generate all candidate paths (without HDBIC trimming)

    This runs the MPCGA algorithm once and returns the raw paths before HDIC penalty.
    The results can be reused with different c3 values.

    Returns:
        cga_output: raw output from MPCGA_while with all candidate paths
        x_train_df: DataFrame version of x_train (needed for HDIC_Trim)
        p_original: original feature dimension
    """
    if max_set is None:
        max_set = DEFAULT_MAX_SET

    try:
        from mpcga_algorithm.mpcga_while import MPCGA_while

        x_train = np.array(x_train)
        p_original = x_train.shape[1]

        # Remove duplicate columns
        x_train_df = pd.DataFrame(x_train, columns=[f"V{i+1}" for i in range(x_train.shape[1])])
        x_train_df = x_train_df.T.drop_duplicates().T

        # Run MPCGA-while (no penalty applied yet)
        cga_output = MPCGA_while(x_train_df.values, y_train, K=K, max_set=max_set,
                                 import_threshold=DEFAULT_IMPORT_THRESHOLD,
                                 max_split=DEFAULT_MAX_SPLIT,
                                 regression_type=DEFAULT_REGRESSION_TYPE)

        return cga_output, x_train_df, p_original
    except Exception:
        return None, None, None


def apply_hdbic_trim(cga_output, x_train_df, y_train, x_test, p_original, c3=None):
    """Apply HDBIC trimming with specified c3 to precomputed MPCGA paths

    Args:
        cga_output: raw output from MPCGA_while
        x_train_df: DataFrame version of training data
        y_train: training labels
        x_test: test features
        p_original: original feature dimension
        c3: HDBIC penalty coefficient (default: DEFAULT_C3)

    Returns:
        predictions, selected_vars, models
    """
    if c3 is None:
        c3 = DEFAULT_C3

    try:
        from mpcga_algorithm.mpcga import HDIC_Trim

        # Apply HDIC trimming with specified c3
        output = HDIC_Trim(x_train_df.values, y_train, cga_output, c3=c3,
                          penalty_type=DEFAULT_PENALTY_TYPE, p_original=p_original)

        # Get unique trimmed paths
        trim_paths = output['trim']
        unique_paths = []
        seen_varsets = []

        for path in trim_paths:
            varnames = [name for name in path if isinstance(name, str) and name != 'beta0']
            varset = tuple(sorted(varnames))
            if varset not in seen_varsets:
                unique_paths.append(path)
                seen_varsets.append(varset)

        # Build models from unique paths
        models = build_models_from_paths(x_train_df, y_train, unique_paths)

        # Get predictions
        predictions = get_result(x_train_df.values, np.zeros(len(x_test)), x_test, models)

        # Collect selected variables
        selected_vars = []
        for vars_list in models['main_var']:
            selected_vars.extend(vars_list)
        selected_vars = list(set(selected_vars))

        return predictions, selected_vars, models
    except Exception as e:
        return None, [], None


def run_mpcga_hdbic(x_train, y_train, x_test, K, max_set=None, c3=None):
    """Run MPCGA+HDBIC (returns predictions, selected vars, and full models dict)

    This is a wrapper that runs MPCGA and applies HDBIC in one call.
    For efficiency when testing multiple c3 values, use run_mpcga_once + apply_hdbic_trim.
    """
    if max_set is None:
        max_set = DEFAULT_MAX_SET
    if c3 is None:
        c3 = DEFAULT_C3

    try:
        models = fit_model_while(
            x_train, y_train,
            K=K,
            max_set=max_set,
            import_threshold=DEFAULT_IMPORT_THRESHOLD,
            max_split=DEFAULT_MAX_SPLIT,
            c3=c3,
            penalty_type=DEFAULT_PENALTY_TYPE,
            use_mtrim=False,
            regression_type=DEFAULT_REGRESSION_TYPE
        )

        predictions = get_result(x_train, np.zeros(len(x_test)), x_test, models)

        # Collect selected variables
        selected_vars = []
        for vars_list in models['main_var']:
            selected_vars.extend(vars_list)
        selected_vars = list(set(selected_vars))

        return predictions, selected_vars, models
    except Exception as e:
        return None, [], None


def run_mpcga_hdbic_op(x_train, y_train, x_test, K, c3=None):
    """Run MPCGA-S - HDBIC with single path (one-pass greedy)

    This is similar to HDAIC(OP) but uses HDBIC criterion instead.
    OP = One-Pass, meaning greedy single-path selection.
    """
    if c3 is None:
        c3 = DEFAULT_C3

    try:
        models = fit_model_while(
            x_train, y_train,
            K=K,
            max_set=1,  # Single path (greedy)
            import_threshold=DEFAULT_IMPORT_THRESHOLD,
            max_split=0,  # No branching
            c3=c3,
            penalty_type=DEFAULT_PENALTY_TYPE,
            use_mtrim=False,
            regression_type=DEFAULT_REGRESSION_TYPE
        )

        predictions = get_result(x_train, np.zeros(len(x_test)), x_test, models)

        # Collect selected variables
        selected_vars = []
        for vars_list in models['main_var']:
            selected_vars.extend(vars_list)
        selected_vars = list(set(selected_vars))

        return predictions, selected_vars
    except Exception as e:
        return None, []


def run_mpcga_mtrim_from_hdbic(x_train, y_train, x_test, hdbic_models, c2=None):
    """
    Run MPCGA+HDBIC+MTrim by applying MTrim to existing HDBIC results

    This avoids rerunning the entire MPCGA algorithm. Instead, it takes
    the paths from MPCGA+HDBIC and applies Model_Trim.

    Args:
        x_train: training features
        y_train: training labels
        x_test: test features
        hdbic_models: models dict from MPCGA+HDBIC (contains 'path' key)
        c2: MTrim tuning parameter (default: DEFAULT_C2)

    Returns:
        predictions, selected_vars, mtrim_models
    """
    if c2 is None:
        c2 = DEFAULT_C2

    try:
        # Extract paths from HDBIC results
        hdbic_paths = hdbic_models['path']

        # Apply Model_Trim with specified c2
        mtrim_paths = Model_Trim(x_train, y_train, hdbic_paths, c2=c2, regression_type=DEFAULT_REGRESSION_TYPE)

        # Build models from trimmed paths
        x_train_df = pd.DataFrame(x_train, columns=[f"V{i+1}" for i in range(x_train.shape[1])])
        mtrim_models = build_models_from_paths(x_train_df, y_train, mtrim_paths)

        # Get predictions
        predictions = get_result(x_train, np.zeros(len(x_test)), x_test, mtrim_models)

        # Collect selected variables
        selected_vars = []
        for vars_list in mtrim_models['main_var']:
            selected_vars.extend(vars_list)
        selected_vars = list(set(selected_vars))

        return predictions, selected_vars, mtrim_models
    except Exception as e:
        return None, [], None


def run_mpcga_ensemble_from_mtrim(x_train, y_train, x_test, mtrim_models, ensemble_type='rf',
                                   cv=None, n_iter=None, random_state=None):
    """
    Train ensemble model (RF or XGB) on MPCGA+MTrim selected variables with cross-validation

    This extracts base variable indices from MTrim's selected variables (including cut variables)
    and trains a separate RF/XGB model on those base variables using CV for hyperparameter tuning.

    Args:
        x_train: training features
        y_train: training labels
        x_test: test features
        mtrim_models: models dict from MPCGA+MTrim (contains 'main_var' key)
        ensemble_type: 'rf' or 'xgb'
        cv: number of cross-validation folds (default: DEFAULT_CV_FOLDS)
        n_iter: number of RandomizedSearchCV iterations (default: DEFAULT_CV_N_ITER)
        random_state: random state for reproducibility (default: DEFAULT_RANDOM_STATE)

    Returns:
        predictions, selected_var_names
    """
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    if cv is None:
        cv = DEFAULT_CV_FOLDS
    if n_iter is None:
        n_iter = DEFAULT_CV_N_ITER
    if random_state is None:
        random_state = DEFAULT_RANDOM_STATE

    if mtrim_models is None:
        return None, []

    try:
        # Get selected variables from MTrim
        selected_vars = []
        for vars_list in mtrim_models['main_var']:
            selected_vars.extend(vars_list)
        selected_vars = list(set(selected_vars))

        if len(selected_vars) == 0:
            return None, []

        # Convert variable names to indices (extract base variable indices)
        # This correctly handles both "V1" and "V1_cut_100" formats
        selected_indices = []
        for var in selected_vars:
            if '_cut_' in var:
                # Cut variable: V1_cut_507 -> extract base V1 -> index 0
                var_idx = int(var.split('_cut_')[0][1:]) - 1
                selected_indices.append(var_idx)
            elif var.startswith('V') and var[1:].isdigit():
                # Original variable: V1 -> index 0
                var_idx = int(var[1:]) - 1
                selected_indices.append(var_idx)

        # Remove duplicates and sort
        selected_indices = sorted(list(set(selected_indices)))

        if len(selected_indices) == 0:
            return None, []

        # Extract selected features
        X_train_selected = x_train[:, selected_indices]
        X_test_selected = x_test[:, selected_indices]

        # Train ensemble model on selected features WITH CROSS-VALIDATION
        if ensemble_type == 'rf':
            # Parameter grid for RF
            param_grid = {
                'n_estimators': DEFAULT_RF_N_ESTIMATORS,
                'max_depth': DEFAULT_RF_MAX_DEPTH,
                'min_samples_split': DEFAULT_RF_MIN_SAMPLES_SPLIT
            }

            rf = RandomForestClassifier(random_state=random_state, n_jobs=1)
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

            rs = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=skf,
                scoring='accuracy',
                n_jobs=1,  # avoid oversubscription (outer loop parallel)
                random_state=random_state,
                verbose=0
            )
            rs.fit(X_train_selected, y_train)
            best_model = rs.best_estimator_

        elif ensemble_type == 'xgb':
            if not XGBOOST_AVAILABLE:
                return None, []

            # Parameter grid for XGB
            param_grid = {
                'n_estimators': DEFAULT_XGB_N_ESTIMATORS,
                'max_depth': DEFAULT_XGB_MAX_DEPTH,
                'learning_rate': DEFAULT_XGB_LEARNING_RATE
            }

            xgb = XGBClassifier(random_state=random_state, eval_metric='logloss', n_jobs=1)
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

            rs = RandomizedSearchCV(
                estimator=xgb,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=skf,
                scoring='accuracy',
                n_jobs=1,
                random_state=random_state,
                verbose=0
            )
            rs.fit(X_train_selected, y_train)
            best_model = rs.best_estimator_

        else:
            return None, []

        # Predict using the best model from CV
        y_pred = best_model.predict(X_test_selected)

        # Convert indices back to variable names for proper metrics computation
        selected_var_names = [f'V{i+1}' for i in selected_indices]
        return y_pred, selected_var_names
    except Exception as e:
        return None, []


def run_cga_hdbic(x_train, y_train, x_test, K, p):
    """Run traditional CGA+HDBIC (OLD VERSION - no cut generation)

    This uses the original CGA algorithm that only considers original variables,
    without automatic cut point generation. This provides a fair baseline comparison.
    """
    try:
        # Use original CGA with c3=1 (as in old version)
        models = fit_model_cga(x_train, y_train, K=K, c3=1, penalty_type='HDBIC')
        predictions = predict_cga(x_train, np.zeros(len(x_test)), x_test, models)

        # Collect selected variables
        selected_vars = []
        for vars_list in models['main_var']:
            selected_vars.extend(vars_list)
        selected_vars = list(set(selected_vars))

        return predictions, selected_vars
    except Exception as e:
        return None, []


def run_single_iteration_parallel(dgp_name, n_train, n_test, p, seed, iteration, methods_to_run=None):
    """
    Run a single iteration with MPCGA methods (for parallel execution)

    Args:
        dgp_name: 'DGP1', 'DGP2', or 'DGP3'
        n_train: training sample size
        n_test: test sample size
        p: number of features
        seed: random seed for this iteration
        iteration: iteration number
        methods_to_run: list of method names to run, or None for all

    Returns:
        tuple: (iteration, results_dict)
    """

    # Generate data (use global USE_CORRELATED_FEATURES)
    if dgp_name == 'DGP1':
        data = generate_data_dgp1(n_train, n_test, p, seed=seed, correlated=USE_CORRELATED_FEATURES)
    elif dgp_name == 'DGP2':
        data = generate_data_dgp2(n_train, n_test, p, seed=seed, correlated=USE_CORRELATED_FEATURES)
    elif dgp_name == 'DGP3':
        data = generate_data_dgp3(n_train, n_test, p, seed=seed, correlated=USE_CORRELATED_FEATURES)
    else:
        raise ValueError(f"Unknown DGP: {dgp_name}")

    x_train, y_train = data['x'], data['y']
    x_test, y_test = data['x_test'], data['y_test']
    true_probs_test = data.get('true_probs_test', None)

    dgp_info = get_dgp_info(dgp_name)
    true_vars = dgp_info['true_vars']
    K = int(3 * np.sqrt(n_train / np.log(p)))

    results = {}

    if methods_to_run is None:
        methods_to_run = ['all']

    def should_run(method_name):
        return 'all' in methods_to_run or method_name in methods_to_run

    # Method 0: True Model (Bayes Optimal)
    if should_run('True_Model') or should_run('all'):
        try:
            if true_probs_test is not None:
                # Predict using true probabilities
                if true_probs_test.ndim == 1:
                    # Binary: prob_test is P(Y=1), threshold at 0.5
                    y_pred_true = (true_probs_test > 0.5).astype(int)
                else:
                    # Multinomial: argmax across classes
                    y_pred_true = np.argmax(true_probs_test, axis=1)
                # True model uses all true variables - convert indices to variable names
                selected_var_names = [f'V{i+1}' for i in true_vars]
                results['True Model'] = compute_metrics(y_test, y_pred_true, selected_var_names, true_vars, p)
            else:
                print(f"  [WARNING] Iteration {iteration}: true_probs_test is None, skipping True Model")
        except Exception as e:
            print(f"  [ERROR] Iteration {iteration}: True Model failed: {e}")

    # Run MPCGA once and cache the raw paths
    cga_output = None
    x_train_df = None
    p_original = None
    hdbic_models = None
    mtrim_models = None

    # Method 1: CGA+HDBIC (traditional greedy CGA)
    if should_run('CGA'):
        try:
            y_pred, selected = run_cga_hdbic(x_train, y_train, x_test, K, p)
            if y_pred is not None:
                results['CGA+HDBIC'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception:
            pass

    # OPTIMIZATION: Run MPCGA once if any HDBIC-based method is needed
    need_mpcga = (should_run('MPCGA_HDBIC') or should_run('MPCGA_S') or
                  should_run('MPCGA_MTrim') or should_run('MPCGA_RF') or should_run('MPCGA_XGB'))

    if need_mpcga:
        try:
            cga_output, x_train_df, p_original = run_mpcga_once(x_train, y_train, K)
        except Exception:
            cga_output = None

    # Method 2: MPCGA+HDBIC
    if should_run('MPCGA_HDBIC'):
        try:
            if cga_output is not None:
                y_pred, selected, hdbic_models = apply_hdbic_trim(
                    cga_output, x_train_df, y_train, x_test, p_original, c3=DEFAULT_C3
                )
                if y_pred is not None:
                    results['MPCGA+HDBIC'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception:
            pass

    # Method 3: MPCGA-S (single path, one-pass greedy)
    if should_run('MPCGA_S'):
        try:
            y_pred, selected = run_mpcga_hdbic_op(x_train, y_train, x_test, K)
            if y_pred is not None:
                results['MPCGA-S'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception:
            pass

    # Ensure HDBIC results are available for MTrim/ensemble methods
    if hdbic_models is None and (should_run('MPCGA_MTrim') or should_run('MPCGA_RF') or should_run('MPCGA_XGB')):
        if cga_output is not None:
            try:
                _, _, hdbic_models = apply_hdbic_trim(
                    cga_output, x_train_df, y_train, x_test, p_original, c3=DEFAULT_C3
                )
            except Exception:
                hdbic_models = None

    # Method 4: MPCGA+HDBIC+MTrim
    if should_run('MPCGA_MTrim'):
        try:
            if hdbic_models is not None:
                y_pred, selected, mtrim_models = run_mpcga_mtrim_from_hdbic(x_train, y_train, x_test, hdbic_models)
                if y_pred is not None:
                    results['MPCGA+HDBIC+MTrim'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception:
            pass

    # Ensure MTrim results are available for ensemble methods
    if mtrim_models is None and (should_run('MPCGA_RF') or should_run('MPCGA_XGB')):
        if hdbic_models is not None:
            try:
                _, _, mtrim_models = run_mpcga_mtrim_from_hdbic(x_train, y_train, x_test, hdbic_models)
            except Exception:
                mtrim_models = None

    # Method 5: MPCGA+RF - Train RF on MTrim selected variables
    if should_run('MPCGA_RF'):
        try:
            if mtrim_models is not None:
                y_pred, selected = run_mpcga_ensemble_from_mtrim(x_train, y_train, x_test, mtrim_models, 'rf')
                if y_pred is not None:
                    results['MPCGA+RF'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception:
            pass

    # Method 6: MPCGA+XGB - Train XGB on MTrim selected variables
    if should_run('MPCGA_XGB'):
        try:
            if mtrim_models is not None:
                y_pred, selected = run_mpcga_ensemble_from_mtrim(x_train, y_train, x_test, mtrim_models, 'xgb')
                if y_pred is not None:
                    results['MPCGA+XGB'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception:
            pass

    return iteration, results


def run_simulation_parallel(dgp_name, n_train, n_test, p, n_iterations=100, start_seed=123,
                            methods_to_run=None, save_csv=True, n_jobs=4, verbose=10):
    """
    Run complete simulation with MPCGA methods using parallel processing

    Args:
        dgp_name: 'DGP1', 'DGP2', or 'DGP3'
        n_train: training sample size
        n_test: test sample size
        p: number of features
        n_iterations: number of simulation iterations
        start_seed: starting random seed
        methods_to_run: list of method names to run, or None for all
        save_csv: whether to save results to CSV
        n_jobs: number of parallel jobs (4 by default)
        verbose: verbosity level for joblib (0=silent, 10=progress bar)

    Returns:
        summaries: dictionary of {method_name: summary_statistics}
        all_results: dictionary of {method_name: list_of_metrics}
    """

    print("=" * 80)
    print(f"MPCGA METHODS SIMULATION: {dgp_name}")
    print("=" * 80)

    dgp_info = get_dgp_info(dgp_name)

    print(f"\nSettings:")
    print(f"  DGP: {dgp_info['description']}")
    print(f"  n_train={n_train}, n_test={n_test}, p={p}")
    print(f"  True variables: {dgp_info['n_true']} variables")
    print(f"  Iterations: {n_iterations}")
    print(f"  K = {int(3 * np.sqrt(n_train / np.log(p)))}")
    print(f"  Parallel jobs: {n_jobs}")

    # Initialize results storage (7 methods: True Model + CGA + 5 MPCGA methods)
    all_method_names = ['True Model', 'CGA+HDBIC', 'MPCGA+HDBIC', 'MPCGA-S',
                        'MPCGA+HDBIC+MTrim', 'MPCGA+RF', 'MPCGA+XGB']
    all_results = {method: [] for method in all_method_names}

    # Run iterations in parallel
    start_time = time.time()

    print(f"\nRunning {n_iterations} iterations in parallel...")
    print("Progress:")

    # Use joblib to parallelize the loop
    iteration_results_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_single_iteration_parallel)(
            dgp_name, n_train, n_test, p, start_seed + i, i+1, methods_to_run
        ) for i in range(n_iterations)
    )

    # Aggregate results from all iterations
    for iteration_num, iteration_results in iteration_results_list:
        for method, metrics in iteration_results.items():
            all_results[method].append(metrics)

    total_time = time.time() - start_time

    # Summarize results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    summaries = {}
    for method, metrics_list in all_results.items():
        if len(metrics_list) > 0:
            summary = summarize_metrics(metrics_list)
            summaries[method] = summary
            print_metrics_summary(method, summary)

    print(f"\n  Total time: {total_time:.1f}s ({total_time/n_iterations:.1f}s per iteration)")

    # Save to CSV
    if save_csv:
        import os
        # Choose output directory based on correlation setting
        output_dir = 'results_correlated' if USE_CORRELATED_FEATURES else 'results'
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{output_dir}/results_MPCGA_{dgp_name}_n{n_train}_p{p}_{timestamp}.csv'

        # Convert to DataFrame
        rows = []
        for method, metrics_list in all_results.items():
            for i, metrics in enumerate(metrics_list):
                row = {'method': method, 'iteration': i+1}
                row.update(metrics)
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"\n  Saved to: {filename}")

    return summaries, all_results


if __name__ == "__main__":
    import multiprocessing

    print(f"Starting MPCGA methods simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
    print()

    # Settings
    n_iterations = 100
    all_methods = ['CGA', 'MPCGA_HDBIC', 'MPCGA_S',
                   'MPCGA_MTrim', 'MPCGA_RF', 'MPCGA_XGB']

    print("=" * 80)
    print("MPCGA methods simulation - 6 configurations")
    print("=" * 80)
    print(f"Methods: {len(all_methods)} MPCGA variants (c3={DEFAULT_C3}, c2={DEFAULT_C2})")
    print(f"  - CGA+HDBIC, MPCGA+HDBIC, MPCGA-S")
    print(f"  - MPCGA+HDBIC+MTrim")
    print(f"  - MPCGA+RF, MPCGA+XGB")
    print()

    # Define 6 simulation configurations (3 DGPs x 2 sample sizes)
    configs = [
        {'dgp': 'DGP1', 'n_train': 300, 'n_test': 100, 'p': 600},
        {'dgp': 'DGP1', 'n_train': 600, 'n_test': 100, 'p': 1000},
        {'dgp': 'DGP2', 'n_train': 300, 'n_test': 100, 'p': 600},
        {'dgp': 'DGP2', 'n_train': 600, 'n_test': 100, 'p': 1000},
        {'dgp': 'DGP3', 'n_train': 300, 'n_test': 100, 'p': 600},
        {'dgp': 'DGP3', 'n_train': 600, 'n_test': 100, 'p': 1000}
    ]

    # Run all 6 configurations
    for i, config in enumerate(configs, 1):
        print(f"\n\n{'='*60}")
        print(f"CONFIGURATION {i}/6: {config['dgp']} (n={config['n_train']}, p={config['p']})")
        print(f"{'='*60}")

        summary, results = run_simulation_parallel(
            config['dgp'],
            n_train=config['n_train'],
            n_test=config['n_test'],
            p=config['p'],
            n_iterations=n_iterations,
            methods_to_run=all_methods,
            save_csv=True,
            n_jobs=4,
            verbose=10
        )

    print(f"\n\nAll 6 MPCGA simulations completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
