"""
Baseline Methods Simulation for DGP1-3 (Binary)
This file contains only baseline methods (Lasso, Adaptive Lasso, RF, XGB, Boruta)

使用方法:
  python simulations/sim_dgp123_baseline.py

參數設定:
  - n_iterations: 100 (預設)
  - n_jobs: 16 (使用 16 個 CPU 核心)
  - verbose: 10 (顯示進度條)
  - 6 configurations: 3 DGPs × 2 sample sizes
"""

import sys
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from data_generation import generate_data_dgp1, generate_data_dgp2, generate_data_dgp3
from evaluation_metrics import compute_metrics, summarize_metrics, print_metrics_summary
from sklearn.linear_model import LogisticRegression
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

try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False
    print("[WARN] Boruta not available, Boruta methods will be skipped")


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


def run_lasso(x_train, y_train, x_test):
    """Run LASSO for binary logistic regression"""
    lr = LogisticRegression(penalty='l1', solver='saga',
                            max_iter=5000, random_state=42)
    lr.fit(x_train, y_train)

    coef = lr.coef_[0]  # Binary case
    selected_indices = np.where(np.abs(coef) > 1e-6)[0].tolist()

    y_pred = lr.predict(x_test)
    return y_pred, selected_indices


def run_adaptive_lasso(x_train, y_train, x_test):
    """Run Adaptive LASSO"""
    # Step 1: Ridge regression to get weights
    lr_ridge = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000, random_state=42)
    lr_ridge.fit(x_train, y_train)
    ridge_coef = np.abs(lr_ridge.coef_[0])

    # Create weights (add small epsilon to avoid division by zero)
    weights = 1.0 / (ridge_coef + 1e-8)

    # Step 2: Weighted LASSO (approximate by rescaling features)
    x_train_weighted = x_train / weights
    x_test_weighted = x_test / weights

    lr_adaptive = LogisticRegression(penalty='l1', solver='saga', max_iter=5000, random_state=42)
    lr_adaptive.fit(x_train_weighted, y_train)

    coef_weighted = lr_adaptive.coef_[0]
    # Transform back
    coef_original = coef_weighted / weights
    selected_indices = np.where(np.abs(coef_original) > 1e-6)[0].tolist()

    y_pred = lr_adaptive.predict(x_test_weighted)
    return y_pred, selected_indices


def run_random_forest(x_train, y_train, x_test):
    """Run Random Forest - selects all features with importance > 0"""
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(x_train, y_train)

    importances = rf.feature_importances_
    # Select all features with importance > 0
    selected_indices = np.where(importances > 0)[0].tolist()

    y_pred = rf.predict(x_test)
    return y_pred, selected_indices


def run_xgboost(x_train, y_train, x_test):
    """Run XGBoost - selects all features with importance > 0"""
    if not XGBOOST_AVAILABLE:
        return None, []

    xgb = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
    xgb.fit(x_train, y_train)

    importances = xgb.feature_importances_
    # Select all features with importance > 0
    selected_indices = np.where(importances > 0)[0].tolist()

    y_pred = xgb.predict(x_test)
    return y_pred, selected_indices


def run_boruta_rf(x_train, y_train, x_test):
    """Run RF+Boruta for feature selection"""
    if not BORUTA_AVAILABLE:
        return None, []

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
    boruta = BorutaPy(rf, n_estimators=100, max_iter=100, random_state=42, verbose=0)

    try:
        boruta.fit(x_train, y_train)
        selected_indices = np.where(boruta.support_)[0].tolist()

        # Train final model with selected features
        if len(selected_indices) > 0:
            rf_final = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            rf_final.fit(x_train[:, selected_indices], y_train)
            y_pred = rf_final.predict(x_test[:, selected_indices])
        else:
            # Fallback: use all features
            selected_indices = list(range(x_train.shape[1]))
            rf_final = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            rf_final.fit(x_train, y_train)
            y_pred = rf_final.predict(x_test)

        return y_pred, selected_indices
    except Exception as e:
        return None, []


def run_boruta_xgb(x_train, y_train, x_test):
    """Run XGB+Boruta for feature selection"""
    if not BORUTA_AVAILABLE or not XGBOOST_AVAILABLE:
        return None, []

    xgb = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss', n_jobs=1)
    boruta = BorutaPy(xgb, n_estimators=100, max_iter=100, perc=90, random_state=42, verbose=0)

    try:
        boruta.fit(x_train, y_train)
        selected_indices = np.where(boruta.support_)[0].tolist()

        # Train final model with selected features
        if len(selected_indices) > 0:
            xgb_final = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
            xgb_final.fit(x_train[:, selected_indices], y_train)
            y_pred = xgb_final.predict(x_test[:, selected_indices])
        else:
            # Fallback: use all features
            selected_indices = list(range(x_train.shape[1]))
            xgb_final = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
            xgb_final.fit(x_train, y_train)
            y_pred = xgb_final.predict(x_test)

        return y_pred, selected_indices
    except Exception as e:
        return None, []


def run_single_iteration_parallel(dgp_name, n_train, n_test, p, seed, iteration, methods_to_run=None):
    """
    Run a single iteration with baseline methods (for parallel execution)

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

    results = {}

    if methods_to_run is None:
        methods_to_run = ['all']

    def should_run(method_name):
        return 'all' in methods_to_run or method_name in methods_to_run

    # Method 1: Lasso
    if should_run('Lasso'):
        try:
            y_pred, selected = run_lasso(x_train, y_train, x_test)
            if y_pred is not None:
                results['Lasso'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception as e:
            pass

    # Method 2: Adaptive Lasso
    if should_run('Adaptive_Lasso'):
        try:
            y_pred, selected = run_adaptive_lasso(x_train, y_train, x_test)
            if y_pred is not None:
                results['Adaptive Lasso'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception as e:
            pass

    # Method 3: RF
    if should_run('RF'):
        try:
            y_pred, selected_indices = run_random_forest(x_train, y_train, x_test)
            if y_pred is not None:
                # Convert indices to variable names
                selected_vars = [f'V{i+1}' for i in selected_indices]
                results['RF'] = compute_metrics(y_test, y_pred, selected_vars, true_vars, p)
        except Exception as e:
            pass

    # Method 4: XGBoost
    if should_run('XGB'):
        try:
            y_pred, selected_indices = run_xgboost(x_train, y_train, x_test)
            if y_pred is not None:
                # Convert indices to variable names
                selected_vars = [f'V{i+1}' for i in selected_indices]
                results['XGBoost'] = compute_metrics(y_test, y_pred, selected_vars, true_vars, p)
        except Exception as e:
            pass

    # Method 5: RF+Boruta
    if should_run('RF_Boruta'):
        try:
            y_pred, selected_indices = run_boruta_rf(x_train, y_train, x_test)
            if y_pred is not None:
                # Convert indices to variable names
                selected_vars = [f'V{i+1}' for i in selected_indices]
                results['RF+Boruta'] = compute_metrics(y_test, y_pred, selected_vars, true_vars, p)
        except Exception as e:
            pass

    # Method 6: XGB+Boruta
    if should_run('XGB_Boruta'):
        try:
            y_pred, selected_indices = run_boruta_xgb(x_train, y_train, x_test)
            if y_pred is not None:
                # Convert indices to variable names
                selected_vars = [f'V{i+1}' for i in selected_indices]
                results['XGB+Boruta'] = compute_metrics(y_test, y_pred, selected_vars, true_vars, p)
        except Exception as e:
            pass

    return iteration, results


def run_simulation_parallel(dgp_name, n_train, n_test, p, n_iterations=100, start_seed=123,
                            methods_to_run=None, save_csv=True, n_jobs=16, verbose=10):
    """
    Run complete simulation with baseline methods using parallel processing

    Args:
        dgp_name: 'DGP1', 'DGP2', or 'DGP3'
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
    all_method_names = ['Lasso', 'Adaptive Lasso', 'RF', 'XGBoost', 'RF+Boruta', 'XGB+Boruta']
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
        os.makedirs('results_new', exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results_new/results_BASELINE_{dgp_name}_n{n_train}_p{p}_{timestamp}.csv'

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
    all_methods = ['Lasso', 'Adaptive_Lasso', 'RF', 'XGB', 'RF_Boruta', 'XGB_Boruta']

    print("=" * 80)
    print("Baseline methods simulation - 6 configurations")
    print("=" * 80)
    print(f"Methods: {len(all_methods)} baseline methods")
    print(f"  - Penalized: Lasso, Adaptive Lasso (2 methods)")
    print(f"  - Tree-based: RF, XGB, RF+Boruta, XGB+Boruta (4 methods)")
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

    print(f"\n\nAll 6 baseline simulations completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
