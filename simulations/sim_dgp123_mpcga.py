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
"""

import sys
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from data_generation import generate_data_dgp1, generate_data_dgp2, generate_data_dgp3
from mpcga_algorithm.mpcga_while import fit_model_while
from mpcga_algorithm.cga import fit_model_cga, predict_cga
from mpcga_algorithm.mpcga import Model_Trim
from mpcga_algorithm.model import get_result
from mpcga_algorithm.cut_generation import generate_test_cut_all
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


def run_mpcga_once(x_train, y_train, K, max_set=5):
    """Run MPCGA to generate all candidate paths (without HDBIC trimming)

    This runs the MPCGA algorithm once and returns the raw paths before HDIC penalty.
    The results can be reused with different c3 values.

    Returns:
        cga_output: raw output from MPCGA_while with all candidate paths
        x_train_df: DataFrame version of x_train (needed for HDIC_Trim)
        p_original: original feature dimension
    """
    try:
        from mpcga_algorithm.mpcga_while import MPCGA_while

        x_train = np.array(x_train)
        p_original = x_train.shape[1]

        # Remove duplicate columns
        x_train_df = pd.DataFrame(x_train, columns=[f"V{i+1}" for i in range(x_train.shape[1])])
        x_train_df = x_train_df.T.drop_duplicates().T

        # Run MPCGA-while (no penalty applied yet)
        cga_output = MPCGA_while(x_train_df.values, y_train, K=K, max_set=max_set,
                                 import_threshold=0.7, max_split=3,
                                 regression_type='binary')

        return cga_output, x_train_df, p_original
    except Exception as e:
        return None, None, None


def apply_hdbic_trim(cga_output, x_train_df, y_train, x_test, p_original, c3=1.0):
    """Apply HDBIC trimming with specified c3 to precomputed MPCGA paths

    Args:
        cga_output: raw output from MPCGA_while
        x_train_df: DataFrame version of training data
        y_train: training labels
        x_test: test features
        p_original: original feature dimension
        c3: HDBIC penalty coefficient

    Returns:
        predictions, selected_vars, models
    """
    try:
        from mpcga_algorithm.mpcga import HDIC_Trim

        # Apply HDIC trimming with specified c3
        output = HDIC_Trim(x_train_df.values, y_train, cga_output, c3=c3,
                          penalty_type='HDBIC', p_original=p_original)

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


def run_mpcga_hdbic(x_train, y_train, x_test, K, max_set=5, c3=1.0):
    """Run MPCGA+HDBIC (returns predictions, selected vars, and full models dict)

    This is a wrapper that runs MPCGA and applies HDBIC in one call.
    For efficiency when testing multiple c3 values, use run_mpcga_once + apply_hdbic_trim.
    """
    try:
        models = fit_model_while(
            x_train, y_train,
            K=K,
            max_set=max_set,
            import_threshold=0.7,
            max_split=3,
            c3=c3,
            penalty_type='HDBIC',
            use_mtrim=False
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


def run_mpcga_hdbic_op(x_train, y_train, x_test, K, c3=0.8):
    """Run MPCGA+HDBIC(OP) - HDBIC with single path (one-pass greedy)

    This is similar to HDAIC(OP) but uses HDBIC criterion instead.
    OP = One-Pass, meaning greedy single-path selection.
    """
    try:
        models = fit_model_while(
            x_train, y_train,
            K=K,
            max_set=1,  # Single path (greedy)
            import_threshold=0.7,
            max_split=0,  # No branching
            c3=c3,
            penalty_type='HDBIC',
            use_mtrim=False,
            regression_type='binary'
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


def run_mpcga_mtrim_from_hdbic(x_train, y_train, x_test, hdbic_models, c2=1.0):
    """
    Run MPCGA+HDBIC+MTrim by applying MTrim to existing HDBIC results

    This avoids rerunning the entire MPCGA algorithm. Instead, it takes
    the paths from MPCGA+HDBIC and applies Model_Trim.

    Args:
        x_train: training features
        y_train: training labels
        x_test: test features
        hdbic_models: models dict from MPCGA+HDBIC (contains 'path' key)
        c2: MTrim tuning parameter (default: 1.0)

    Returns:
        predictions, selected_vars, mtrim_models
    """
    try:
        # Extract paths from HDBIC results
        hdbic_paths = hdbic_models['path']

        # Apply Model_Trim with specified c2
        mtrim_paths = Model_Trim(x_train, y_train, hdbic_paths, c2=c2)

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


def run_mpcga_ensemble_from_mtrim(x_train, y_train, x_test, mtrim_models, ensemble_type='rf'):
    """
    Train ensemble model (RF or XGB) on MPCGA+MTrim selected variables

    This extracts base variable indices from MTrim's selected variables (including cut variables)
    and trains a separate RF/XGB model on those base variables.

    Args:
        x_train: training features
        y_train: training labels
        x_test: test features
        mtrim_models: models dict from MPCGA+MTrim (contains 'main_var' key)
        ensemble_type: 'rf' or 'xgb'

    Returns:
        predictions, selected_var_names
    """
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

        # Train ensemble model on selected features
        if ensemble_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif ensemble_type == 'xgb':
            if not XGBOOST_AVAILABLE:
                return None, []
            model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
        else:
            return None, []

        model.fit(x_train[:, selected_indices], y_train)
        y_pred = model.predict(x_test[:, selected_indices])

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

    # Generate data
    if dgp_name == 'DGP1':
        data = generate_data_dgp1(n_train, n_test, p, seed=seed)
    elif dgp_name == 'DGP2':
        data = generate_data_dgp2(n_train, n_test, p, seed=seed)
    elif dgp_name == 'DGP3':
        data = generate_data_dgp3(n_train, n_test, p, seed=seed)
    else:
        raise ValueError(f"Unknown DGP: {dgp_name}")

    x_train, y_train = data['x'], data['y']
    x_test, y_test = data['x_test'], data['y_test']

    dgp_info = get_dgp_info(dgp_name)
    true_vars = dgp_info['true_vars']
    K = int(3 * np.sqrt(n_train / np.log(p)))

    results = {}

    if methods_to_run is None:
        methods_to_run = ['all']

    def should_run(method_name):
        return 'all' in methods_to_run or method_name in methods_to_run

    # Run MPCGA once and cache the raw paths for reuse with different c3 values
    cga_output = None
    x_train_df = None
    p_original = None
    hdbic_c08_models = None
    hdbic_c10_models = None
    mtrim_c08_models = None
    mtrim_c10_models = None

    # Method 1: CGA+HDBIC (traditional greedy CGA)
    if should_run('CGA'):
        try:
            y_pred, selected = run_cga_hdbic(x_train, y_train, x_test, K, p)
            if y_pred is not None:
                results['CGA+HDBIC'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception as e:
            pass

    # OPTIMIZATION: Run MPCGA once if any HDBIC-based method is needed
    need_mpcga = (should_run('MPCGA_HDBIC_c08') or should_run('MPCGA_HDBIC_c10') or
                  should_run('MPCGA_MTrim_c08') or should_run('MPCGA_MTrim_c10') or
                  should_run('MPCGA_RF_c08') or should_run('MPCGA_XGB_c08') or
                  should_run('MPCGA_RF_c10') or should_run('MPCGA_XGB_c10'))

    if need_mpcga:
        try:
            cga_output, x_train_df, p_original = run_mpcga_once(x_train, y_train, K)
        except Exception as e:
            cga_output = None

    # Method 2: MPCGA+HDBIC (c3=0.8) - Apply HDBIC trim to shared MPCGA result
    if should_run('MPCGA_HDBIC_c08'):
        try:
            if cga_output is not None:
                y_pred, selected, hdbic_c08_models = apply_hdbic_trim(
                    cga_output, x_train_df, y_train, x_test, p_original, c3=0.8
                )
                if y_pred is not None:
                    results['MPCGA+HDBIC(c3=0.8)'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception as e:
            pass

    # Method 3: MPCGA+HDBIC (c3=1.0) - Apply HDBIC trim to shared MPCGA result
    if should_run('MPCGA_HDBIC_c10'):
        try:
            if cga_output is not None:
                y_pred, selected, hdbic_c10_models = apply_hdbic_trim(
                    cga_output, x_train_df, y_train, x_test, p_original, c3=1.0
                )
                if y_pred is not None:
                    results['MPCGA+HDBIC(c3=1.0)'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception as e:
            pass

    # Method 4: MPCGA+HDBIC(OP) (single path, one-pass greedy)
    if should_run('MPCGA_HDBIC_OP'):
        try:
            y_pred, selected = run_mpcga_hdbic_op(x_train, y_train, x_test, K, c3=0.8)
            if y_pred is not None:
                results['MPCGA+HDBIC(OP)'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception as e:
            pass

    # For MTrim methods: ensure HDBIC results are available (from shared MPCGA)
    need_hdbic_c08 = (should_run('MPCGA_MTrim_c08') or should_run('MPCGA_RF_c08') or should_run('MPCGA_XGB_c08'))
    need_hdbic_c10 = (should_run('MPCGA_MTrim_c10') or should_run('MPCGA_RF_c10') or should_run('MPCGA_XGB_c10'))

    if hdbic_c08_models is None and need_hdbic_c08 and cga_output is not None:
        try:
            _, _, hdbic_c08_models = apply_hdbic_trim(
                cga_output, x_train_df, y_train, x_test, p_original, c3=0.8
            )
        except Exception as e:
            hdbic_c08_models = None

    if hdbic_c10_models is None and need_hdbic_c10 and cga_output is not None:
        try:
            _, _, hdbic_c10_models = apply_hdbic_trim(
                cga_output, x_train_df, y_train, x_test, p_original, c3=1.0
            )
        except Exception as e:
            hdbic_c10_models = None

    # Variables to store MTrim predictions for reuse
    mtrim_c08_pred = None
    mtrim_c08_selected = None
    mtrim_c10_pred = None
    mtrim_c10_selected = None

    # Method 5: MPCGA+HDBIC+MTrim (c3=0.8, c2=3)
    if should_run('MPCGA_MTrim_c08'):
        try:
            if hdbic_c08_models is not None:
                y_pred, selected, mtrim_c08_models = run_mpcga_mtrim_from_hdbic(x_train, y_train, x_test, hdbic_c08_models, c2=3.0)
                if y_pred is not None:
                    results['MPCGA+HDBIC+MTrim(c3=0.8)'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
                    mtrim_c08_pred = y_pred
                    mtrim_c08_selected = selected
        except Exception as e:
            pass

    # Method 6: MPCGA+HDBIC+MTrim (c3=1.0, c2=3)
    if should_run('MPCGA_MTrim_c10'):
        try:
            if hdbic_c10_models is not None:
                y_pred, selected, mtrim_c10_models = run_mpcga_mtrim_from_hdbic(x_train, y_train, x_test, hdbic_c10_models, c2=3.0)
                if y_pred is not None:
                    results['MPCGA+HDBIC+MTrim(c3=1.0)'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
                    mtrim_c10_pred = y_pred
                    mtrim_c10_selected = selected
        except Exception as e:
            pass

    # Ensure MTrim results are available for ensemble methods
    if mtrim_c08_pred is None and (should_run('MPCGA_RF_c08') or should_run('MPCGA_XGB_c08')):
        try:
            if hdbic_c08_models is not None:
                y_pred, selected, mtrim_c08_models = run_mpcga_mtrim_from_hdbic(x_train, y_train, x_test, hdbic_c08_models, c2=3.0)
                if y_pred is not None:
                    mtrim_c08_pred = y_pred
                    mtrim_c08_selected = selected
        except Exception as e:
            pass

    if mtrim_c10_pred is None and (should_run('MPCGA_RF_c10') or should_run('MPCGA_XGB_c10')):
        try:
            if hdbic_c10_models is not None:
                y_pred, selected, mtrim_c10_models = run_mpcga_mtrim_from_hdbic(x_train, y_train, x_test, hdbic_c10_models, c2=3.0)
                if y_pred is not None:
                    mtrim_c10_pred = y_pred
                    mtrim_c10_selected = selected
        except Exception as e:
            pass

    # Method 7: MPCGA+RF (c3=0.8) - Train RF on MTrim selected variables
    if should_run('MPCGA_RF_c08'):
        try:
            if mtrim_c08_models is not None:
                y_pred, selected = run_mpcga_ensemble_from_mtrim(x_train, y_train, x_test, mtrim_c08_models, 'rf')
                if y_pred is not None:
                    results['MPCGA+RF(c3=0.8)'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception as e:
            pass

    # Method 8: MPCGA+XGB (c3=0.8) - Train XGB on MTrim selected variables
    if should_run('MPCGA_XGB_c08'):
        try:
            if mtrim_c08_models is not None:
                y_pred, selected = run_mpcga_ensemble_from_mtrim(x_train, y_train, x_test, mtrim_c08_models, 'xgb')
                if y_pred is not None:
                    results['MPCGA+XGB(c3=0.8)'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception as e:
            pass

    # Method 9: MPCGA+RF (c3=1.0) - Train RF on MTrim selected variables
    if should_run('MPCGA_RF_c10'):
        try:
            if mtrim_c10_models is not None:
                y_pred, selected = run_mpcga_ensemble_from_mtrim(x_train, y_train, x_test, mtrim_c10_models, 'rf')
                if y_pred is not None:
                    results['MPCGA+RF(c3=1.0)'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception as e:
            pass

    # Method 10: MPCGA+XGB (c3=1.0) - Train XGB on MTrim selected variables
    if should_run('MPCGA_XGB_c10'):
        try:
            if mtrim_c10_models is not None:
                y_pred, selected = run_mpcga_ensemble_from_mtrim(x_train, y_train, x_test, mtrim_c10_models, 'xgb')
                if y_pred is not None:
                    results['MPCGA+XGB(c3=1.0)'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception as e:
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

    # Initialize results storage (10 MPCGA methods)
    all_method_names = ['CGA+HDBIC', 'MPCGA+HDBIC(c3=0.8)', 'MPCGA+HDBIC(c3=1.0)', 'MPCGA+HDBIC(OP)',
                        'MPCGA+HDBIC+MTrim(c3=0.8)', 'MPCGA+HDBIC+MTrim(c3=1.0)',
                        'MPCGA+RF(c3=0.8)', 'MPCGA+XGB(c3=0.8)',
                        'MPCGA+RF(c3=1.0)', 'MPCGA+XGB(c3=1.0)']
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
        os.makedirs('results', exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/results_MPCGA_{dgp_name}_n{n_train}_p{p}_{timestamp}.csv'

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
    all_methods = ['CGA', 'MPCGA_HDBIC_c08', 'MPCGA_HDBIC_c10', 'MPCGA_HDBIC_OP',
                   'MPCGA_MTrim_c08', 'MPCGA_MTrim_c10',
                   'MPCGA_RF_c08', 'MPCGA_XGB_c08', 'MPCGA_RF_c10', 'MPCGA_XGB_c10']

    print("=" * 80)
    print("MPCGA methods simulation - 6 configurations")
    print("=" * 80)
    print(f"Methods: {len(all_methods)} MPCGA variants")
    print(f"  - CGA/MPCGA core: CGA+HDBIC, MPCGA+HDBIC(c3=0.8), MPCGA+HDBIC(c3=1.0), MPCGA+HDBIC(OP) (4 methods)")
    print(f"  - MPCGA+MTrim: c3=0.8, c3=1.0 (both with c2=3) (2 methods)")
    print(f"  - MPCGA+Ensemble: RF(c3=0.8), XGB(c3=0.8), RF(c3=1.0), XGB(c3=1.0) (4 methods)")
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
            n_jobs=16,
            verbose=10
        )

    print(f"\n\nAll 6 MPCGA simulations completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
