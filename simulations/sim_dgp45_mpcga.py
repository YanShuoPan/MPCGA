"""
MPCGA Methods Simulation for DGP4 and DGP5 (Multinomial)
This file contains only MPCGA-related methods

使用方法:
  python simulations/sim_dgp45_mpcga.py

參數設定:
  - n_iterations: 100 (預設)
  - n_jobs: 4 (使用 4 個 CPU 核心)
  - verbose: 10 (顯示進度條)
  - 4 configurations: 2 DGPs × 2 sample sizes
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
from data_generation import generate_data_dgp4, generate_data_dgp5
from mpcga_algorithm.mpcga_while import fit_model_while
from evaluation_metrics import compute_metrics, summarize_metrics, print_metrics_summary
from sklearn.ensemble import RandomForestClassifier
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
DEFAULT_C2 = 3.0           # MTrim c2 for trimming

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
DEFAULT_REGRESSION_TYPE = 'multinomial'  # 'binary' or 'multinomial'

# ============================================================================


def get_dgp_info(dgp_name):
    """Get DGP information"""
    if dgp_name == 'DGP4':
        return {
            'description': 'Multinomial, Linear Effects',
            'n_true': 5,
            'true_vars': [0, 1, 2, 3, 4]  # V1-V5
        }
    elif dgp_name == 'DGP5':
        return {
            'description': 'Multinomial, Nonlinear (V1^2, V2^2)',
            'n_true': 5,
            'true_vars': [0, 1, 2, 3, 4]  # V1-V5
        }
    else:
        raise ValueError(f"Unknown DGP: {dgp_name}")


def reconstruct_test_features(var_list, x_train, x_test):
    """Reconstruct test features including cut variables

    Args:
        var_list: list of variable names (e.g., ['V4_cut_333', 'V5', 'V3'])
        x_train: training data
        x_test: test data

    Returns:
        X_test: feature matrix with intercept + features in correct order
    """
    features = [np.ones(len(x_test))]  # Start with intercept

    for var_name in var_list:
        if '_cut_' in var_name:
            # Cut variable: V4_cut_333 means V4 > cutpoint at position 333
            parts = var_name.split('_cut_')
            var_idx = int(parts[0][1:]) - 1  # V4 -> index 3
            cut_idx = int(parts[1])

            # Get cut point from training data
            train_var_sorted = np.sort(x_train[:, var_idx])
            cut_point = train_var_sorted[min(cut_idx, len(train_var_sorted)-1)]

            # Apply to test data
            cut_feature = (x_test[:, var_idx] > cut_point).astype(float)
            features.append(cut_feature)
        else:
            # Original variable
            var_idx = int(var_name[1:]) - 1
            features.append(x_test[:, var_idx])

    return np.column_stack(features)


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
    except Exception as e:
        return None, None, None


def apply_hdbic_trim(cga_output, x_train_df, y_train, x_test, p_original, c3=None):
    """Apply HDBIC trimming with specified c3 to precomputed MPCGA paths

    Args:
        cga_output: raw output from MPCGA_while
        x_train_df: DataFrame version of training data
        y_train: training labels
        x_test: test features
        p_original: original feature dimension
        c3: HDBIC penalty coefficient

    Returns:
        predictions, selected_vars, result_dict (with 'model', 'main_var', 'path')
    """
    if c3 is None:
        c3 = DEFAULT_C3

    try:
        from mpcga_algorithm.mpcga import HDIC_Trim
        from sklearn.linear_model import LogisticRegression

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

        if len(unique_paths) == 0:
            return None, [], None

        # Build models from unique paths
        models = []
        vars_list = []

        for path in unique_paths:
            all_varnames = [name for name in path if isinstance(name, str) and name != 'beta0']

            # Reconstruct training features for this path
            X_train_reconstructed = reconstruct_test_features(all_varnames, x_train_df.values, x_train_df.values)

            # Fit multinomial logistic regression
            lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
            lr.fit(X_train_reconstructed, y_train)

            models.append(lr)
            vars_list.append(all_varnames)

        # Predict using ALL models and use majority voting
        from scipy.stats import mode
        all_predictions = []

        for i, (model, vars) in enumerate(zip(models, vars_list)):
            X_test_reconstructed = reconstruct_test_features(vars, x_train_df.values, x_test)
            y_pred_i = model.predict(X_test_reconstructed)
            all_predictions.append(y_pred_i)

        # Majority voting across all candidate paths
        all_predictions = np.array(all_predictions)  # shape: (n_models, n_test)
        y_pred = mode(all_predictions, axis=0, keepdims=False)[0]  # Take majority vote

        # Collect selected variables from ALL candidate paths
        selected_vars = []
        for vars in vars_list:
            selected_vars.extend(vars)
        selected_vars = list(set(selected_vars))  # Remove duplicates

        # Return in same format as before (for compatibility with MTrim functions)
        result_dict = {
            'model': models,
            'main_var': vars_list,
            'path': unique_paths
        }

        return y_pred, selected_vars, result_dict
    except Exception as e:
        return None, [], None


def run_mpcga_hdbic(x_train, y_train, x_test, K, c3=None):
    """Run MPCGA+HDBIC (wrapper for backward compatibility)

    This is a wrapper that runs MPCGA and applies HDBIC in one call.
    For efficiency when testing multiple c3 values, use run_mpcga_once + apply_hdbic_trim.
    """
    if c3 is None:
        c3 = DEFAULT_C3

    # Two-stage approach
    cga_output, x_train_df, p_original = run_mpcga_once(x_train, y_train, K)

    if cga_output is None:
        return None, [], None

    return apply_hdbic_trim(cga_output, x_train_df, y_train, x_test, p_original, c3=c3)


def run_mpcga_hdbic_op(x_train, y_train, x_test, K, c3=None):
    """Run MPCGA-S - HDBIC with single path (one-pass greedy)

    OP = One-Pass, meaning greedy single-path selection.
    This uses max_set=1 and max_split=0 to force greedy behavior.
    """
    if c3 is None:
        c3 = DEFAULT_C3

    result = fit_model_while(
        x_train, y_train,
        K=K,
        c3=c3,
        max_set=1,  # Single path (greedy)
        import_threshold=DEFAULT_IMPORT_THRESHOLD,
        max_split=0,  # No branching
        penalty_type=DEFAULT_PENALTY_TYPE,
        use_mtrim=False,
        regression_type=DEFAULT_REGRESSION_TYPE
    )

    models = result.get('model', [])
    vars_list = result.get('main_var', [])

    if len(models) == 0:
        return None, []

    # Use first model for prediction
    X_test_reconstructed = reconstruct_test_features(vars_list[0], x_train, x_test)
    y_pred = models[0].predict(X_test_reconstructed)

    # Collect selected variables from ALL candidate paths (should be just 1 for OP)
    # But we use the same logic for consistency
    selected_vars = []
    for vars in vars_list:
        selected_vars.extend(vars)
    selected_vars = list(set(selected_vars))  # Remove duplicates

    # Return variable names (not indices) for proper metrics computation
    return y_pred, selected_vars


def apply_mtrim_to_hdbic_result(hdbic_result, x_train, y_train, x_test, c2):
    """Apply MTrim to existing HDBIC candidate models

    Args:
        hdbic_result: Result dict from run_mpcga_hdbic (contains 'model', 'main_var', 'path')
        x_train: training features
        y_train: training labels
        x_test: test features
        c2: MTrim parameter

    Returns:
        y_pred: predictions on test set
        selected_vars: selected variable names
    """
    from mpcga_algorithm.mpcga import Model_Trim
    from sklearn.linear_model import LogisticRegression
    import pandas as pd

    if hdbic_result is None:
        return None, []

    # Get all candidate paths (before MTrim)
    all_paths = hdbic_result.get('path', [])

    if len(all_paths) == 0:
        return None, []

    # Apply MTrim to the candidate paths
    X_df = pd.DataFrame(x_train, columns=[f'V{i+1}' for i in range(x_train.shape[1])])
    trimmed_paths = Model_Trim(X_df.values, y_train, all_paths, c2=c2, regression_type='multinomial')

    if len(trimmed_paths) == 0:
        return None, []

    # Predict using ALL trimmed models and use majority voting
    from scipy.stats import mode
    all_predictions = []

    for path in trimmed_paths:
        path_vars = [name for name in path if isinstance(name, str) and name != 'beta0']

        # Reconstruct features for this path
        X_train_reconstructed = reconstruct_test_features(path_vars, x_train, x_train)
        X_test_reconstructed = reconstruct_test_features(path_vars, x_train, x_test)

        # Fit logistic regression for this path
        lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
        lr.fit(X_train_reconstructed, y_train)

        # Get predictions
        y_pred_i = lr.predict(X_test_reconstructed)
        all_predictions.append(y_pred_i)

    # Majority voting across all trimmed paths
    all_predictions = np.array(all_predictions)  # shape: (n_models, n_test)
    y_pred = mode(all_predictions, axis=0, keepdims=False)[0]  # Take majority vote

    # Collect selected variables from ALL trimmed paths (DGP1-3 approach)
    selected_vars = []
    for path in trimmed_paths:
        vars_in_path = [name for name in path if isinstance(name, str) and name != 'beta0']
        selected_vars.extend(vars_in_path)
    selected_vars = list(set(selected_vars))  # Remove duplicates

    return y_pred, selected_vars


def run_mpcga_mtrim(x_train, y_train, x_test, K, c3=None, c2=None):
    """Run MPCGA+HDBIC+MTrim (legacy function, now just wraps the new approach)"""
    if c3 is None:
        c3 = DEFAULT_C3
    if c2 is None:
        c2 = DEFAULT_C2

    result = fit_model_while(
        x_train, y_train,
        K=K,
        c3=c3,
        max_set=DEFAULT_MAX_SET,
        import_threshold=DEFAULT_IMPORT_THRESHOLD,
        max_split=DEFAULT_MAX_SPLIT,
        penalty_type=DEFAULT_PENALTY_TYPE,
        use_mtrim=True,
        c2=c2,
        regression_type=DEFAULT_REGRESSION_TYPE
    )

    models = result.get('model', [])
    vars_list = result.get('main_var', [])

    if len(models) == 0:
        return None, []

    X_test_reconstructed = reconstruct_test_features(vars_list[0], x_train, x_test)
    y_pred = models[0].predict(X_test_reconstructed)

    # Return variable names (not indices) for proper metrics computation
    return y_pred, vars_list[0]


def run_mpcga_mtrim_c2_3(x_train, y_train, x_test, K, c3=None):
    """Run MPCGA+HDBIC+MTrim with c2=3"""
    if c3 is None:
        c3 = DEFAULT_C3
    return run_mpcga_mtrim(x_train, y_train, x_test, K, c3=c3, c2=DEFAULT_C2)


def run_mpcga_ensemble_from_vars(selected_vars, x_train, y_train, x_test, ensemble_type='rf',
                                 cv=None, n_iter=None, random_state=None):
    """Run ensemble model (RF or XGB) on given selected variables with cross-validation

    Args:
        selected_vars: List of selected variable names from MPCGA+MTrim
        x_train: training features
        y_train: training labels
        x_test: test features
        ensemble_type: 'rf' or 'xgb'
        cv: number of cross-validation folds (default: DEFAULT_CV_FOLDS)
        n_iter: number of RandomizedSearchCV iterations (default: DEFAULT_CV_N_ITER)
        random_state: random state for reproducibility (default: DEFAULT_RANDOM_STATE)
    """
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    if cv is None:
        cv = DEFAULT_CV_FOLDS
    if n_iter is None:
        n_iter = DEFAULT_CV_N_ITER
    if random_state is None:
        random_state = DEFAULT_RANDOM_STATE

    if selected_vars is None or len(selected_vars) == 0:
        return None, []

    # Convert variable names to indices (extract base variable indices)
    selected_indices = []
    for var in selected_vars:
        if '_cut_' in var:
            # Cut variable: extract base variable index (V1_cut_507 -> V1 -> index 0)
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

        xgb = XGBClassifier(random_state=random_state, eval_metric='mlogloss', n_jobs=1)
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


def run_mpcga_ensemble(x_train, y_train, x_test, ensemble_type='rf', c2=None):
    """Run MPCGA+Ensemble (RF or XGB on MPCGA selected variables) - legacy function

    This is kept for backwards compatibility but now runs MPCGA+HDBIC+MTrim internally
    """
    if c2 is None:
        c2 = DEFAULT_C2

    # First run MPCGA+HDBIC+MTrim to get selected variables
    K = int(3 * np.sqrt(len(x_train) / np.log(x_train.shape[1])))
    y_pred_mpcga, selected_vars = run_mpcga_mtrim(x_train, y_train, x_test, K, c3=DEFAULT_C3, c2=c2)

    if y_pred_mpcga is None or len(selected_vars) == 0:
        return None, []

    # Use the new helper function
    return run_mpcga_ensemble_from_vars(selected_vars, x_train, y_train, x_test, ensemble_type)


def run_mpcga_ensemble_c2_3(x_train, y_train, x_test, ensemble_type='rf'):
    """Run MPCGA+Ensemble with c2=3"""
    return run_mpcga_ensemble(x_train, y_train, x_test, ensemble_type=ensemble_type, c2=DEFAULT_C2)


def run_single_iteration_parallel(dgp_name, n_train, n_test, p, seed, iteration, methods_to_run=None):
    """
    Run a single iteration with MPCGA methods (for parallel execution)

    Args:
        dgp_name: 'DGP4' or 'DGP5'
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
    if dgp_name == 'DGP4':
        data = generate_data_dgp4(n_train, n_test, p, seed=seed, correlated=USE_CORRELATED_FEATURES)
    elif dgp_name == 'DGP5':
        data = generate_data_dgp5(n_train, n_test, p, seed=seed, correlated=USE_CORRELATED_FEATURES)
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
    hdbic_result = None
    mtrim_selected = None

    # OPTIMIZATION: Run MPCGA once if any HDBIC-based method is needed
    need_mpcga = (should_run('MPCGA_HDBIC') or should_run('MPCGA_S') or
                  should_run('MPCGA_MTrim') or should_run('MPCGA_RF') or should_run('MPCGA_XGB'))

    if need_mpcga:
        try:
            cga_output, x_train_df, p_original = run_mpcga_once(x_train, y_train, K)
        except Exception:
            cga_output = None

    # Method 1: MPCGA+HDBIC
    if should_run('MPCGA_HDBIC'):
        try:
            if cga_output is not None:
                y_pred, selected, hdbic_result = apply_hdbic_trim(
                    cga_output, x_train_df, y_train, x_test, p_original, c3=DEFAULT_C3
                )
                if y_pred is not None:
                    results['MPCGA+HDBIC'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception:
            pass

    # Method 2: MPCGA-S (single path, one-pass greedy)
    if should_run('MPCGA_S'):
        try:
            y_pred, selected = run_mpcga_hdbic_op(x_train, y_train, x_test, K)
            if y_pred is not None:
                results['MPCGA-S'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception:
            pass

    # Ensure HDBIC results are available for MTrim/ensemble methods
    if hdbic_result is None and (should_run('MPCGA_MTrim') or should_run('MPCGA_RF') or should_run('MPCGA_XGB')):
        if cga_output is not None:
            try:
                _, _, hdbic_result = apply_hdbic_trim(
                    cga_output, x_train_df, y_train, x_test, p_original, c3=DEFAULT_C3
                )
            except Exception:
                hdbic_result = None

    # Method 3: MPCGA+HDBIC+MTrim
    if should_run('MPCGA_MTrim'):
        try:
            if hdbic_result is not None:
                y_pred, mtrim_selected = apply_mtrim_to_hdbic_result(hdbic_result, x_train, y_train, x_test, c2=DEFAULT_C2)
                if y_pred is not None:
                    results['MPCGA+HDBIC+MTrim'] = compute_metrics(y_test, y_pred, mtrim_selected, true_vars, p)
        except Exception:
            pass

    # Ensure MTrim results are available for ensemble methods
    if mtrim_selected is None and (should_run('MPCGA_RF') or should_run('MPCGA_XGB')):
        if hdbic_result is not None:
            try:
                _, mtrim_selected = apply_mtrim_to_hdbic_result(hdbic_result, x_train, y_train, x_test, c2=DEFAULT_C2)
            except Exception:
                mtrim_selected = None

    # Method 4: MPCGA+RF - Train RF on MTrim selected variables
    if should_run('MPCGA_RF'):
        try:
            y_pred, selected = run_mpcga_ensemble_from_vars(mtrim_selected, x_train, y_train, x_test, 'rf')
            if y_pred is not None:
                results['MPCGA+RF'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception:
            pass

    # Method 5: MPCGA+XGB - Train XGB on MTrim selected variables
    if should_run('MPCGA_XGB'):
        try:
            y_pred, selected = run_mpcga_ensemble_from_vars(mtrim_selected, x_train, y_train, x_test, 'xgb')
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
        dgp_name: 'DGP4' or 'DGP5'
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

    # Initialize results storage (6 methods: 1 True Model + 5 MPCGA methods)
    all_method_names = [
        'True Model',
        'MPCGA+HDBIC', 'MPCGA-S',
        'MPCGA+HDBIC+MTrim',
        'MPCGA+RF', 'MPCGA+XGB'
    ]
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
    all_methods = [
        'MPCGA_HDBIC', 'MPCGA_S',
        'MPCGA_MTrim',
        'MPCGA_RF', 'MPCGA_XGB'
    ]

    print("=" * 80)
    print("MPCGA methods simulation - 4 configurations")
    print("=" * 80)
    print(f"Methods: {len(all_methods)} MPCGA variants")
    print(f"  - MPCGA+HDBIC (c3={DEFAULT_C3}), MPCGA-S")
    print(f"  - MPCGA+HDBIC+MTrim (c2={int(DEFAULT_C2)})")
    print(f"  - MPCGA+RF, MPCGA+XGB (on MTrim selected variables)")
    print()

    # Define 4 simulation configurations
    configs = [
        {'dgp': 'DGP4', 'n_train': 400, 'n_test': 100, 'p': 200},
        {'dgp': 'DGP4', 'n_train': 600, 'n_test': 100, 'p': 300},
        {'dgp': 'DGP5', 'n_train': 400, 'n_test': 100, 'p': 200},
        {'dgp': 'DGP5', 'n_train': 600, 'n_test': 100, 'p': 300}
    ]

    # Run all 4 configurations
    for i, config in enumerate(configs, 1):
        print(f"\n\n{'='*60}")
        print(f"CONFIGURATION {i}/4: {config['dgp']} (n={config['n_train']}, p={config['p']})")
        print(f"{'='*60}")

        summary, results = run_simulation_parallel(
            config['dgp'],
            n_train=config['n_train'],
            n_test=config['n_test'],
            p=config['p'],
            n_iterations=n_iterations,
            methods_to_run=all_methods,
            save_csv=True,
            n_jobs=16,
            verbose=10
        )

    print(f"\n\nAll 4 MPCGA simulations completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
