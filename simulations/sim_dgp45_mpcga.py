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
"""

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


def run_mpcga_hdbic(x_train, y_train, x_test, K, c3=0.5):
    """Run MPCGA+HDBIC (returns best model and all candidate info)"""
    result = fit_model_while(
        x_train, y_train,
        K=K,
        c3=c3,
        max_set=5,
        import_threshold=0.7,
        max_split=3,
        penalty_type='HDBIC',
        use_mtrim=False,
        regression_type='multinomial'
    )

    models = result.get('model', [])
    vars_list = result.get('main_var', [])

    if len(models) == 0:
        return None, [], result

    # Reconstruct test features with cut variables (use first model for prediction)
    X_test_reconstructed = reconstruct_test_features(vars_list[0], x_train, x_test)
    y_pred = models[0].predict(X_test_reconstructed)

    # Collect selected variables from ALL candidate paths (DGP1-3 approach)
    selected_vars = []
    for vars in vars_list:
        selected_vars.extend(vars)
    selected_vars = list(set(selected_vars))  # Remove duplicates

    # Return variable names (not indices) for proper metrics computation
    # Also return the full result for reuse in MTrim
    return y_pred, selected_vars, result


def run_mpcga_hdbic_op(x_train, y_train, x_test, K, c3=0.5):
    """Run MPCGA+HDBIC(OP) - HDBIC with single path (one-pass greedy)

    OP = One-Pass, meaning greedy single-path selection.
    This uses max_set=1 and max_split=0 to force greedy behavior.
    """
    result = fit_model_while(
        x_train, y_train,
        K=K,
        c3=c3,
        max_set=1,  # Single path (greedy)
        import_threshold=0.7,
        max_split=0,  # No branching
        penalty_type='HDBIC',
        use_mtrim=False,
        regression_type='multinomial'
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

    # Take the first model after MTrim for prediction
    best_path = trimmed_paths[0]
    best_vars = [name for name in best_path if isinstance(name, str) and name != 'beta0']

    # Reconstruct training and test features with best model's variables
    X_train_reconstructed = reconstruct_test_features(best_vars, x_train, x_train)
    X_test_reconstructed = reconstruct_test_features(best_vars, x_train, x_test)

    # Fit the model on training data
    lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
    lr.fit(X_train_reconstructed, y_train)

    # Predict on test data
    y_pred = lr.predict(X_test_reconstructed)

    # Collect selected variables from ALL trimmed paths (DGP1-3 approach)
    selected_vars = []
    for path in trimmed_paths:
        vars_in_path = [name for name in path if isinstance(name, str) and name != 'beta0']
        selected_vars.extend(vars_in_path)
    selected_vars = list(set(selected_vars))  # Remove duplicates

    return y_pred, selected_vars


def run_mpcga_mtrim(x_train, y_train, x_test, K, c3=0.5, c2=10.0):
    """Run MPCGA+HDBIC+MTrim (legacy function, now just wraps the new approach)"""
    result = fit_model_while(
        x_train, y_train,
        K=K,
        c3=c3,
        max_set=5,
        import_threshold=0.7,
        max_split=3,
        penalty_type='HDBIC',
        use_mtrim=True,
        c2=c2,
        regression_type='multinomial'
    )

    models = result.get('model', [])
    vars_list = result.get('main_var', [])

    if len(models) == 0:
        return None, []

    X_test_reconstructed = reconstruct_test_features(vars_list[0], x_train, x_test)
    y_pred = models[0].predict(X_test_reconstructed)

    # Return variable names (not indices) for proper metrics computation
    return y_pred, vars_list[0]


def run_mpcga_mtrim_c2_3(x_train, y_train, x_test, K, c3=0.5):
    """Run MPCGA+HDBIC+MTrim with c2=3"""
    return run_mpcga_mtrim(x_train, y_train, x_test, K, c3=c3, c2=3.0)


def run_mpcga_ensemble_from_vars(selected_vars, x_train, y_train, x_test, ensemble_type='rf'):
    """Run ensemble model (RF or XGB) on given selected variables

    Args:
        selected_vars: List of selected variable names from MPCGA+MTrim
        x_train: training features
        y_train: training labels
        x_test: test features
        ensemble_type: 'rf' or 'xgb'
    """
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

    # Train ensemble model on selected features
    if ensemble_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    elif ensemble_type == 'xgb':
        if not XGBOOST_AVAILABLE:
            return None, []
        model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='mlogloss')
    else:
        return None, []

    model.fit(x_train[:, selected_indices], y_train)
    y_pred = model.predict(x_test[:, selected_indices])

    # Convert indices back to variable names for proper metrics computation
    selected_var_names = [f'V{i+1}' for i in selected_indices]
    return y_pred, selected_var_names


def run_mpcga_ensemble(x_train, y_train, x_test, ensemble_type='rf', c2=10.0):
    """Run MPCGA+Ensemble (RF or XGB on MPCGA selected variables) - legacy function

    This is kept for backwards compatibility but now runs MPCGA+HDBIC+MTrim internally
    """
    # First run MPCGA+HDBIC+MTrim to get selected variables
    K = int(3 * np.sqrt(len(x_train) / np.log(x_train.shape[1])))
    y_pred_mpcga, selected_vars = run_mpcga_mtrim(x_train, y_train, x_test, K, c3=0.5, c2=c2)

    if y_pred_mpcga is None or len(selected_vars) == 0:
        return None, []

    # Use the new helper function
    return run_mpcga_ensemble_from_vars(selected_vars, x_train, y_train, x_test, ensemble_type)


def run_mpcga_ensemble_c2_3(x_train, y_train, x_test, ensemble_type='rf'):
    """Run MPCGA+Ensemble with c2=3"""
    return run_mpcga_ensemble(x_train, y_train, x_test, ensemble_type=ensemble_type, c2=3.0)


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

    # Generate data
    if dgp_name == 'DGP4':
        data = generate_data_dgp4(n_train, n_test, p, seed=seed)
    elif dgp_name == 'DGP5':
        data = generate_data_dgp5(n_train, n_test, p, seed=seed)
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

    # Run MPCGA+HDBIC once and cache the result for MTrim methods
    hdbic_result = None
    hdbic_y_pred = None
    hdbic_selected = None

    # Cache MTrim results for ensemble methods
    mtrim_c10_selected = None
    mtrim_c3_selected = None

    # Method 1: MPCGA+HDBIC
    if should_run('MPCGA_HDBIC'):
        try:
            hdbic_y_pred, hdbic_selected, hdbic_result = run_mpcga_hdbic(x_train, y_train, x_test, K)
            if hdbic_y_pred is not None:
                results['MPCGA+HDBIC'] = compute_metrics(y_test, hdbic_y_pred, hdbic_selected, true_vars, p)
        except Exception as e:
            pass

    # Method 2: MPCGA+HDBIC(OP) (single path, one-pass greedy)
    if should_run('MPCGA_HDBIC_OP'):
        try:
            y_pred, selected = run_mpcga_hdbic_op(x_train, y_train, x_test, K)
            if y_pred is not None:
                results['MPCGA+HDBIC(OP)'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception as e:
            pass

    # For MTrim methods: reuse HDBIC result if available, otherwise run it
    need_hdbic = (should_run('MPCGA_MTrim_c2_10') or should_run('MPCGA_MTrim_c2_3') or
                  should_run('MPCGA_RF') or should_run('MPCGA_XGB') or
                  should_run('MPCGA_RF_c2_3') or should_run('MPCGA_XGB_c2_3'))

    if hdbic_result is None and need_hdbic:
        try:
            hdbic_y_pred, hdbic_selected, hdbic_result = run_mpcga_hdbic(x_train, y_train, x_test, K)
        except Exception as e:
            hdbic_result = None

    # Method 3: MPCGA+HDBIC+MTrim (c2=10)
    if should_run('MPCGA_MTrim_c2_10'):
        try:
            y_pred, mtrim_c10_selected = apply_mtrim_to_hdbic_result(hdbic_result, x_train, y_train, x_test, c2=10.0)
            if y_pred is not None:
                results['MPCGA+HDBIC+MTrim(c2=10)'] = compute_metrics(y_test, y_pred, mtrim_c10_selected, true_vars, p)
        except Exception as e:
            pass

    # Method 4: MPCGA+HDBIC+MTrim (c2=3)
    if should_run('MPCGA_MTrim_c2_3'):
        try:
            y_pred, mtrim_c3_selected = apply_mtrim_to_hdbic_result(hdbic_result, x_train, y_train, x_test, c2=3.0)
            if y_pred is not None:
                results['MPCGA+HDBIC+MTrim(c2=3)'] = compute_metrics(y_test, y_pred, mtrim_c3_selected, true_vars, p)
        except Exception as e:
            pass

    # Ensure MTrim results are available for ensemble methods
    if mtrim_c10_selected is None and (should_run('MPCGA_RF') or should_run('MPCGA_XGB')):
        try:
            _, mtrim_c10_selected = apply_mtrim_to_hdbic_result(hdbic_result, x_train, y_train, x_test, c2=10.0)
        except Exception as e:
            pass

    if mtrim_c3_selected is None and (should_run('MPCGA_RF_c2_3') or should_run('MPCGA_XGB_c2_3')):
        try:
            _, mtrim_c3_selected = apply_mtrim_to_hdbic_result(hdbic_result, x_train, y_train, x_test, c2=3.0)
        except Exception as e:
            pass

    # Method 5: MPCGA+RF (c2=10) - uses MTrim(c2=10) selected variables
    if should_run('MPCGA_RF'):
        try:
            y_pred, selected = run_mpcga_ensemble_from_vars(mtrim_c10_selected, x_train, y_train, x_test, 'rf')
            if y_pred is not None:
                results['MPCGA+RF'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception as e:
            pass

    # Method 6: MPCGA+XGB (c2=10) - uses MTrim(c2=10) selected variables
    if should_run('MPCGA_XGB'):
        try:
            y_pred, selected = run_mpcga_ensemble_from_vars(mtrim_c10_selected, x_train, y_train, x_test, 'xgb')
            if y_pred is not None:
                results['MPCGA+XGB'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception as e:
            pass

    # Method 7: MPCGA+RF (c2=3) - uses MTrim(c2=3) selected variables
    if should_run('MPCGA_RF_c2_3'):
        try:
            y_pred, selected = run_mpcga_ensemble_from_vars(mtrim_c3_selected, x_train, y_train, x_test, 'rf')
            if y_pred is not None:
                results['MPCGA+RF(c2=3)'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception as e:
            pass

    # Method 8: MPCGA+XGB (c2=3) - uses MTrim(c2=3) selected variables
    if should_run('MPCGA_XGB_c2_3'):
        try:
            y_pred, selected = run_mpcga_ensemble_from_vars(mtrim_c3_selected, x_train, y_train, x_test, 'xgb')
            if y_pred is not None:
                results['MPCGA+XGB(c2=3)'] = compute_metrics(y_test, y_pred, selected, true_vars, p)
        except Exception as e:
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

    # Initialize results storage (8 MPCGA methods)
    all_method_names = ['MPCGA+HDBIC', 'MPCGA+HDBIC(OP)',
                        'MPCGA+HDBIC+MTrim(c2=10)', 'MPCGA+HDBIC+MTrim(c2=3)',
                        'MPCGA+RF', 'MPCGA+XGB', 'MPCGA+RF(c2=3)', 'MPCGA+XGB(c2=3)']
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
    all_methods = ['MPCGA_HDBIC', 'MPCGA_HDBIC_OP', 'MPCGA_MTrim_c2_10', 'MPCGA_MTrim_c2_3',
                   'MPCGA_RF', 'MPCGA_XGB', 'MPCGA_RF_c2_3', 'MPCGA_XGB_c2_3']

    print("=" * 80)
    print("MPCGA methods simulation - 4 configurations")
    print("=" * 80)
    print(f"Methods: {len(all_methods)} MPCGA variants")
    print(f"  - MPCGA core: HDBIC, HDBIC(OP) (2 methods)")
    print(f"  - MPCGA+MTrim: c2=10, c2=3 (2 methods)")
    print(f"  - MPCGA+Ensemble: RF(c2=10), XGB(c2=10), RF(c2=3), XGB(c2=3) (4 methods)")
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
