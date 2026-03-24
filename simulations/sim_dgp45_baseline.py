"""
Baseline Methods Simulation for DGP4 and DGP5 (Multinomial)
This file contains baseline methods + True Model + CGA+HDBIC

使用方法:
  python simulations/sim_dgp4_dgp5_baseline.py

參數設定:
  - n_iterations: 100 (預設)
  - n_jobs: 16 (使用 16 個 CPU 核心)
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
from evaluation_metrics import compute_metrics, summarize_metrics, print_metrics_summary
from mpcga_algorithm.cga import fit_model_cga, predict_cga
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV

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

try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False
    print("[WARN] Boruta not available, Boruta methods will be skipped")


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


def run_lasso(x_train, y_train, x_test, cv=5, random_state=42):
    """
    Run multinomial LASSO with cross-validation (LogisticRegressionCV).

    Returns:
        y_pred, selected_indices, best_C
    """

    # C = 1 / lambda, smaller C => stronger regularization
    Cs = np.logspace(-4, 2, 20)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    lr_cv = LogisticRegressionCV(
        Cs=Cs,
        cv=skf,
        penalty='l1',
        solver='saga',
        # multi_class='multinomial' removed - automatically uses multinomial for multi-class
        scoring='accuracy',
        max_iter=5000,
        n_jobs=1,               # 外層已有 Parallel，避免 oversubscribe
        refit=True,
        random_state=random_state
    )

    lr_cv.fit(x_train, y_train)

    # coef_: (n_classes, p)
    coef = lr_cv.coef_
    non_zero_mask = np.any(np.abs(coef) > 1e-6, axis=0)
    selected_indices = np.where(non_zero_mask)[0].tolist()

    y_pred = lr_cv.predict(x_test)

    # best C (one per class, usually same; take first)
    if lr_cv.C_.ndim == 1:
        best_C = float(lr_cv.C_[0])
    else:
        best_C = float(lr_cv.C_[0, 0])

    return y_pred, selected_indices, best_C


def run_cga_hdbic(x_train, y_train, x_test, K, p):
    """Run traditional CGA+HDBIC (OLD VERSION - no cut generation)

    This uses the original CGA algorithm that only considers original variables,
    without automatic cut point generation. This provides a fair baseline comparison.

    Args:
        x_train: training features
        y_train: training labels
        x_test: test features
        K: model complexity parameter (NOT number of classes!)
            Formula: K = int(3 * sqrt(n_train / log(p)))
        p: total number of features

    Returns:
        predictions: predicted labels
        selected_vars: list of selected variable names (e.g., ['V1', 'V2', ...])
    """
    try:
        # Use original CGA with c3=1 (as in old version)
        models = fit_model_cga(x_train, y_train, K=K, c3=1, penalty_type='HDBIC')
        predictions = predict_cga(x_train, np.zeros(len(x_test)), x_test, models)

        # Collect selected variables (already in correct format from CGA)
        selected_vars = []
        for vars_list in models['main_var']:
            selected_vars.extend(vars_list)
        selected_vars = list(set(selected_vars))

        return predictions, selected_vars
    except Exception as e:
        return None, []


def run_adaptive_lasso(x_train, y_train, x_test, cv=5, random_state=42):
    """
    Run standard Adaptive LASSO (Zou 2006) with CV in both steps

    Step 1: Lasso with CV to get initial coefficient estimates
    Step 2: Adaptive Weighted Lasso with CV to select final model

    Returns:
        y_pred, selected_indices
    """
    # Step 1: Lasso with CV to get initial estimates
    Cs = np.logspace(-4, 2, 20)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    lr_lasso_cv = LogisticRegressionCV(
        Cs=Cs,
        cv=skf,
        penalty='l1',
        solver='saga',
        scoring='accuracy',
        max_iter=10000,
        tol=1e-4,
        n_jobs=1,
        refit=True,
        random_state=random_state
    )
    lr_lasso_cv.fit(x_train, y_train)

    # Get Lasso coefficients
    # For multinomial: coef_ is (n_classes, n_features)
    # Take the mean absolute value across classes for adaptive weights
    lasso_coef = np.mean(np.abs(lr_lasso_cv.coef_), axis=0)

    # Create adaptive weights (gamma=1)
    # weights = 1 / |lasso_coef|^gamma, gamma=1 is standard
    weights = 1.0 / (lasso_coef + 1e-8)

    # Step 2: Adaptive Weighted Lasso WITH CV (this is the key!)
    # Rescale features by adaptive weights
    x_train_weighted = x_train / weights
    x_test_weighted = x_test / weights

    # Do CV again on the weighted features to find optimal lambda
    lr_adaptive_cv = LogisticRegressionCV(
        Cs=Cs,
        cv=skf,
        penalty='l1',
        solver='saga',
        scoring='accuracy',
        max_iter=10000,
        tol=1e-4,
        n_jobs=1,
        refit=True,
        random_state=random_state
    )
    lr_adaptive_cv.fit(x_train_weighted, y_train)

    # Transform back to original scale
    coef_weighted = lr_adaptive_cv.coef_  # (n_classes, n_features)
    coef_original = coef_weighted / weights

    # Select features that are non-zero in ANY class
    non_zero_mask = np.any(np.abs(coef_original) > 1e-6, axis=0)
    selected_indices = np.where(non_zero_mask)[0].tolist()

    y_pred = lr_adaptive_cv.predict(x_test_weighted)
    return y_pred, selected_indices

def run_random_forest(x_train, y_train, x_test, cv=5, n_iter=8, random_state=42):
    """
    RF with RandomizedSearchCV sampled from the original discrete grid.
    Returns: y_pred, selected_indices, best_params
    """
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 30, 50],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestClassifier(random_state=random_state, n_jobs=1)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    rs = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,  # sample from the original grid
        n_iter=n_iter,
        cv=skf,
        scoring='accuracy',
        n_jobs=1,                         # avoid oversubscription (outer loop parallel)
        random_state=random_state,
        verbose=0
    )
    rs.fit(x_train, y_train)

    best_rf = rs.best_estimator_
    best_params = rs.best_params_

    importances = best_rf.feature_importances_
    selected_indices = np.where(importances > 0)[0].tolist()

    y_pred = best_rf.predict(x_test)
    return y_pred, selected_indices, best_params


def run_xgboost(x_train, y_train, x_test, cv=5, n_iter=8, random_state=42):
    """
    XGB with RandomizedSearchCV sampled from the original discrete grid.
    Returns: y_pred, selected_indices, best_params
    """
    if not XGBOOST_AVAILABLE:
        return None, [], None

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 30, 50],
        'learning_rate': [0.2, 0.4, 0.6]
    }

    xgb = XGBClassifier(random_state=random_state, eval_metric='mlogloss', n_jobs=1)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    rs = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,  # sample from the original grid
        n_iter=n_iter,
        cv=skf,
        scoring='accuracy',
        n_jobs=1,
        random_state=random_state,
        verbose=0
    )
    rs.fit(x_train, y_train)

    best_xgb = rs.best_estimator_
    best_params = rs.best_params_

    importances = best_xgb.feature_importances_
    selected_indices = np.where(importances > 0)[0].tolist()

    y_pred = best_xgb.predict(x_test)
    return y_pred, selected_indices, best_params


def run_boruta_rf(x_train, y_train, x_test, best_params=None):
    """
    Boruta+RF for feature selection using best_params from CV.
    Returns: y_pred, selected_indices
    """
    if not BORUTA_AVAILABLE:
        return None, []

    # sanitize
    Xtr = np.asarray(x_train)
    Xte = np.asarray(x_test)
    ytr = np.asarray(y_train).ravel()

    # defaults + override
    params = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }
    if best_params is not None:
        params.update(best_params)

    params["random_state"] = 42
    params["n_jobs"] = 1

    rf = RandomForestClassifier(**params)

    boruta = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        max_iter=100,
        random_state=42,
        verbose=0
    )

    try:
        boruta.fit(Xtr, ytr)
        selected_indices = np.where(boruta.support_)[0].tolist()

        rf_final = RandomForestClassifier(**params)

        if len(selected_indices) > 0:
            rf_final.fit(Xtr[:, selected_indices], ytr)
            y_pred = rf_final.predict(Xte[:, selected_indices])
        else:
            selected_indices = list(range(Xtr.shape[1]))
            rf_final.fit(Xtr, ytr)
            y_pred = rf_final.predict(Xte)

        return y_pred, selected_indices
    except Exception:
        return None, []


def run_boruta_xgb(x_train, y_train, x_test, best_params=None):
    """
    Boruta+XGB for feature selection using best_params from CV.
    Returns: y_pred, selected_indices
    """
    if not BORUTA_AVAILABLE or not XGBOOST_AVAILABLE:
        return None, []

    # sanitize
    Xtr = np.asarray(x_train)
    Xte = np.asarray(x_test)
    ytr = np.asarray(y_train).ravel()

    params = {'n_estimators': 100, 'max_depth': 30, 'learning_rate': 0.3}
    if best_params is not None:
        params.update(best_params)

    params["random_state"] = 42
    params["eval_metric"] = "mlogloss"
    params["n_jobs"] = 1

    xgb = XGBClassifier(**params)

    boruta = BorutaPy(
        estimator=xgb,
        n_estimators="auto",
        max_iter=100,
        perc=90,
        random_state=42,
        verbose=0
    )

    try:
        boruta.fit(Xtr, ytr)
        selected_indices = np.where(boruta.support_)[0].tolist()

        xgb_final = XGBClassifier(**params)

        if len(selected_indices) > 0:
            xgb_final.fit(Xtr[:, selected_indices], ytr)
            y_pred = xgb_final.predict(Xte[:, selected_indices])
        else:
            selected_indices = list(range(Xtr.shape[1]))
            xgb_final.fit(Xtr, ytr)
            y_pred = xgb_final.predict(Xte)

        return y_pred, selected_indices
    except Exception:
        return None, []

def run_single_iteration_parallel(dgp_name, n_train, n_test, p, seed, iteration, methods_to_run=None):
    """
    Run a single iteration with baseline methods (for parallel execution)

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

    # Set random seed for reproducibility in parallel execution
    np.random.seed(seed)

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

    results = {}

    if methods_to_run is None:
        methods_to_run = ['all']

    def should_run(method_name):
        return 'all' in methods_to_run or method_name in methods_to_run

    # Storage for best_params from CV
    rf_best_params = None
    xgb_best_params = None

    # Method 0: True Model (Bayes Optimal)
    if should_run('True_Model') or should_run('all'):
        try:
            if true_probs_test is not None:
                # Predict using true probabilities (multinomial: take argmax)
                y_pred_true = np.argmax(true_probs_test, axis=1)
                # True model uses all true variables - convert indices to variable names
                selected_var_names = [f'V{i+1}' for i in true_vars]
                results['True Model'] = compute_metrics(y_test, y_pred_true, selected_var_names, true_vars, p)
            else:
                print(f"  [WARNING] Iteration {iteration}: true_probs_test is None, skipping True Model")
        except Exception as e:
            print(f"  [ERROR] Iteration {iteration}: True Model failed: {e}")

    # Method 1: CGA+HDBIC (Traditional CGA without cut generation)
    if should_run('CGA_HDBIC') or should_run('all'):
        try:
            # Calculate K parameter (model complexity, NOT number of classes!)
            K = int(3 * np.sqrt(n_train / np.log(p)))
            y_pred_cga, selected_cga = run_cga_hdbic(x_train, y_train, x_test, K, p)
            if y_pred_cga is not None:
                # CGA already returns variable names in correct format
                results['CGA+HDBIC'] = compute_metrics(y_test, y_pred_cga, selected_cga, true_vars, p)
        except Exception as e:
            pass

    # Method 3: Lasso
    if should_run('Lasso'):
        try:
            y_pred, selected,_ = run_lasso(x_train, y_train, x_test)
            if y_pred is not None:
                # Convert indices to variable names
                selected_names = [f'V{i+1}' for i in selected]
                results['Lasso'] = compute_metrics(y_test, y_pred, selected_names, true_vars, p)
        except Exception as e:
            pass

    # Method 4: Adaptive Lasso
    if should_run('Adaptive_Lasso'):
        try:
            y_pred, selected_indices = run_adaptive_lasso(x_train, y_train, x_test)
            if y_pred is not None:
                # Convert indices to variable names
                selected_vars = [f'V{i+1}' for i in selected_indices]
                results['Adaptive Lasso'] = compute_metrics(y_test, y_pred, selected_vars, true_vars, p)
        except Exception as e:
            pass

    # Method 5: RF (with CV - get best_params for Boruta)
    if should_run('RF'):
        try:
            y_pred, selected_indices, rf_best_params = run_random_forest(x_train, y_train, x_test)
            if y_pred is not None:
                # Convert indices to variable names
                selected_vars = [f'V{i+1}' for i in selected_indices]
                results['RF'] = compute_metrics(y_test, y_pred, selected_vars, true_vars, p)
        except Exception as e:
            pass

    # Method 6: XGBoost (with CV - get best_params for Boruta)
    if should_run('XGB'):
        try:
            y_pred, selected_indices, xgb_best_params = run_xgboost(x_train, y_train, x_test)
            if y_pred is not None:
                # Convert indices to variable names
                selected_vars = [f'V{i+1}' for i in selected_indices]
                results['XGBoost'] = compute_metrics(y_test, y_pred, selected_vars, true_vars, p)
        except Exception as e:
            pass

    # Method 7: Boruta+RF (use best_params from RF if available)
    if should_run('Boruta_RF'):
        try:
            y_pred, selected_indices = run_boruta_rf(x_train, y_train, x_test, best_params=rf_best_params)
            if y_pred is not None:
                # Convert indices to variable names
                selected_vars = [f'V{i+1}' for i in selected_indices]
                results['Boruta+RF'] = compute_metrics(y_test, y_pred, selected_vars, true_vars, p)
        except Exception as e:
            pass

    # Method 8: Boruta+XGB (use best_params from XGBoost if available)
    if should_run('Boruta_XGB'):
        try:
            y_pred, selected_indices = run_boruta_xgb(x_train, y_train, x_test, best_params=xgb_best_params)
            if y_pred is not None:
                # Convert indices to variable names
                selected_vars = [f'V{i+1}' for i in selected_indices]
                results['Boruta+XGB'] = compute_metrics(y_test, y_pred, selected_vars, true_vars, p)
        except Exception as e:
            pass

    return iteration, results


def run_simulation_parallel(dgp_name, n_train, n_test, p, n_iterations=100, start_seed=123,
                            methods_to_run=None, save_csv=True, n_jobs=4, verbose=10):
    """
    Run complete simulation with baseline methods using parallel processing

    Args:
        dgp_name: 'DGP4' or 'DGP5'
        n_train: training sample size
        n_test: test sample size
        p: number of features
        n_iterations: number of simulation iterations
        start_seed: starting random seed
        methods_to_run: list of method names to run, or None for all
        save_csv: whether to save results to CSV
        n_jobs: number of parallel jobs (16 by default)
        verbose: verbosity level for joblib (0=silent, 10=progress bar)

    Returns:
        summaries: dictionary of {method_name: summary_statistics}
        all_results: dictionary of {method_name: list_of_metrics}
    """

    print("=" * 80)
    print(f"BASELINE METHODS SIMULATION: {dgp_name}")
    print("=" * 80)

    dgp_info = get_dgp_info(dgp_name)

    print(f"\nSettings:")
    print(f"  DGP: {dgp_info['description']}")
    print(f"  n_train={n_train}, n_test={n_test}, p={p}")
    print(f"  True variables: {dgp_info['n_true']} variables")
    print(f"  Iterations: {n_iterations}")
    print(f"  Parallel jobs: {n_jobs}")

    # Initialize results storage
    all_method_names = ['True Model', 'CGA+HDBIC', 'Lasso', 'Adaptive Lasso', 'RF', 'XGBoost', 'Boruta+RF', 'Boruta+XGB']
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
        filename = f'{output_dir}/results_BASELINE_{dgp_name}_n{n_train}_p{p}_{timestamp}.csv'

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

    print(f"Starting baseline methods simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
    print()

    # Settings
    n_iterations = 100
    all_methods = ['True_Model', 'CGA_HDBIC', 'Lasso', 'Adaptive_Lasso', 'RF', 'XGB', 'Boruta_RF', 'Boruta_XGB']

    print("=" * 80)
    print("Baseline methods simulation - 4 configurations")
    print("=" * 80)
    print(f"Methods: {len(all_methods)} methods total")
    print(f"  - True Model (Bayes Optimal) (1 method)")
    print(f"  - CGA+HDBIC (traditional CGA without cuts) (1 method)")
    print(f"  - Penalized: Lasso, Adaptive Lasso (2 methods)")
    print(f"  - Tree-based: RF, XGB, Boruta+RF, Boruta+XGB (4 methods)")
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

    print(f"\n\nAll 4 baseline simulations completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
