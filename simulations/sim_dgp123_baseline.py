"""
Baseline Methods Simulation for DGP1-3 (Binary)
This file contains baseline methods + True Model + CGA+HDBIC

使用方法:
  python simulations/sim_dgp123_baseline.py

參數設定:
  - n_iterations: 100 (預設)
  - n_jobs: 16 (使用 16 個 CPU 核心)
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
from evaluation_metrics import compute_metrics, summarize_metrics, print_metrics_summary
from mpcga_algorithm.cga import fit_model_cga, predict_cga
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
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
def run_lasso_cv(x_train, y_train, x_test, cv=5, random_state=42):
    # 建議用 logspace 掃 C（C 越小正則越強、越稀疏）
    Cs = np.logspace(-4, 2, 20)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    lr_cv = LogisticRegressionCV(
        Cs=Cs,
        cv=skf,
        penalty='l1',
        solver='saga',
        scoring='accuracy',      # 你也可改 'neg_log_loss' / 'roc_auc'
        max_iter=5000,
        n_jobs=1,                # 你外層已經 joblib 平行了，這裡建議 1 避免 oversubscribe
        refit=True,
        random_state=random_state
    )
    lr_cv.fit(x_train, y_train)

    coef = lr_cv.coef_[0]
    selected_indices = np.where(np.abs(coef) > 1e-6)[0].tolist()

    y_pred = lr_cv.predict(x_test)

    best_C = float(lr_cv.C_[0])  # binary 情況
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
    Run standard Adaptive LASSO (Zou 2006) with CV in both steps - Binary version

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
    # For binary: coef_ is (1, n_features) or (n_features,)
    # Take absolute value for adaptive weights
    lasso_coef = np.abs(lr_lasso_cv.coef_[0]) if lr_lasso_cv.coef_.ndim == 2 else np.abs(lr_lasso_cv.coef_)

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
    coef_weighted = lr_adaptive_cv.coef_[0] if lr_adaptive_cv.coef_.ndim == 2 else lr_adaptive_cv.coef_
    coef_original = coef_weighted / weights

    # Select features that are non-zero
    selected_indices = np.where(np.abs(coef_original) > 1e-6)[0].tolist()

    y_pred = lr_adaptive_cv.predict(x_test_weighted)
    return y_pred, selected_indices



def run_random_forest(x_train, y_train, x_test, cv=5, n_iter=8, random_state=42):
    """
    Run Random Forest with RandomizedSearchCV (sampled from original grid).
    Only n_iter parameter combinations are evaluated instead of full grid.

    Returns:
        y_pred, selected_indices, best_params
    """

    # 原本的 grid（保持不變，只是改成抽樣）
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 30, 50],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestClassifier(random_state=random_state, n_jobs=1)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    rs = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,  # 👈 還是原本的 grid
        n_iter=n_iter,                    # 👈 只抽 n_iter 組
        cv=skf,
        scoring='accuracy',
        n_jobs=1,                         # 外層已 parallel，這裡保持 1
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
    Run XGBoost with RandomizedSearchCV (sampled from original grid).
    Only n_iter parameter combinations are evaluated instead of full grid.

    Returns:
        y_pred, selected_indices, best_params
    """
    if not XGBOOST_AVAILABLE:
        return None, [], None

    # 原本的 grid（保持不變，只是改成抽樣）
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 30, 50],
        'learning_rate': [0.2, 0.4, 0.6]
    }

    xgb = XGBClassifier(
        random_state=random_state,
        eval_metric='logloss',
        n_jobs=1
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    rs = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,  # 👈 還是原本 grid
        n_iter=n_iter,                    # 👈 只抽 n_iter 組
        cv=skf,
        scoring='accuracy',
        n_jobs=1,                         # 外層已 parallel，這裡保持 1
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
    Run Boruta+RF for feature selection using best_params from CV.

    Returns:
        y_pred, selected_indices
        - y_pred: np.ndarray of predicted labels on x_test (or None if failed)
        - selected_indices: list[int] of selected feature indices (empty if failed)
    """
    if not BORUTA_AVAILABLE:
        return None, []

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    # ---- sanitize inputs ----
    Xtr = np.asarray(x_train)
    Xte = np.asarray(x_test)
    ytr = np.asarray(y_train).ravel()

    # ---- default / merge params ----
    # Keep it close to your original defaults but a bit more robust.
    default_params = {
        "n_estimators": 300,      # base RF for Boruta (can be moderately large)
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }
    if best_params is not None:
        # best_params from GridSearch may not include all keys; override defaults
        default_params.update(best_params)

    # Make sure RF is reproducible and doesn't oversubscribe CPU (you already parallelize outside)
    default_params["random_state"] = 42
    default_params["n_jobs"] = 1

    # ---- Boruta base estimator ----
    rf = RandomForestClassifier(**default_params)

    # BorutaPy params:
    # - n_estimators='auto' lets Boruta decide how many trees each iteration (often better than fixed)
    # - max_iter controls how long Boruta runs; raise if you want more stable selection
    boruta = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        max_iter=100,
        perc=90,             # keep your original XGB-Boruta style strictness; adjust if too harsh
        random_state=42,
        verbose=0
    )

    try:
        boruta.fit(Xtr, ytr)
        selected_indices = np.where(boruta.support_)[0].tolist()

        # ---- Train final RF on selected features (or fallback to all) ----
        rf_final = RandomForestClassifier(**default_params)

        if len(selected_indices) > 0:
            rf_final.fit(Xtr[:, selected_indices], ytr)
            y_pred = rf_final.predict(Xte[:, selected_indices])
        else:
            # Fallback: use all features if Boruta selects none
            selected_indices = list(range(Xtr.shape[1]))
            rf_final.fit(Xtr, ytr)
            y_pred = rf_final.predict(Xte)

        return y_pred, selected_indices

    except Exception:
        return None, []

def run_boruta_xgb(x_train, y_train, x_test, best_params=None):
    """
    Run Boruta+XGB for feature selection using best_params from CV.

    Returns:
        y_pred, selected_indices
    """
    if not BORUTA_AVAILABLE or not XGBOOST_AVAILABLE:
        return None, []

    # ---- sanitize inputs ----
    Xtr = np.asarray(x_train)
    Xte = np.asarray(x_test)
    ytr = np.asarray(y_train).ravel()

    # If best_params not provided, use default params
    params = {'n_estimators': 100, 'max_depth': 30, 'learning_rate': 0.3}
    if best_params is not None:
        params.update(best_params)

    # Ensure reproducibility and avoid oversubscription (outer loop already parallel)
    params['random_state'] = 42
    params['eval_metric'] = 'logloss'
    params['n_jobs'] = 1

    # Boruta base estimator
    xgb = XGBClassifier(**params)

    # Prefer n_estimators='auto' so Boruta adapts internal tree count per iteration
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

        # Train final model with same params on selected features
        xgb_final = XGBClassifier(**params)

        if len(selected_indices) > 0:
            xgb_final.fit(Xtr[:, selected_indices], ytr)
            y_pred = xgb_final.predict(Xte[:, selected_indices])
        else:
            # Fallback: use all features
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

    # Set random seed for reproducibility in parallel execution
    np.random.seed(seed)

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
                # For binary classification, true_probs_test is 1D (prob of class 1)
                y_pred_true = (true_probs_test > 0.5).astype(int)
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
            y_pred, selected, best_C = run_lasso_cv(x_train, y_train, x_test)
            if y_pred is not None:
                # Convert indices to variable names
                selected_names = [f'V{i+1}' for i in selected]
                results['Lasso'] = compute_metrics(y_test, y_pred, selected_names, true_vars, p)
        except Exception as e:
            pass

    # Method 4: Adaptive Lasso
    if should_run('Adaptive_Lasso'):
        try:
            y_pred, selected = run_adaptive_lasso(x_train, y_train, x_test)
            if y_pred is not None:
                # Convert indices to variable names
                selected_names = [f'V{i+1}' for i in selected]
                results['Adaptive Lasso'] = compute_metrics(y_test, y_pred, selected_names, true_vars, p)
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
    print("Baseline methods simulation - 6 configurations")
    print("=" * 80)
    print(f"Methods: {len(all_methods)} methods total")
    print(f"  - True Model (Bayes Optimal) (1 method)")
    print(f"  - CGA+HDBIC (traditional CGA without cuts) (1 method)")
    print(f"  - Penalized: Lasso, Adaptive Lasso (2 methods)")
    print(f"  - Tree-based: RF, XGB, Boruta+RF, Boruta+XGB (4 methods)")
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
            n_jobs=2,
            verbose=10
        )

    print(f"\n\nAll 6 baseline simulations completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
