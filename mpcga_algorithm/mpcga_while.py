"""
MPCGA-While with Precomputed Sorting
使用預計算排序資訊加速切點選擇
"""

import numpy as np
import pandas as pd
from collections import deque
from sklearn.linear_model import LogisticRegression
from .utils import fd, fd_multinomial_aggregated, llik_multinomial
from .cut_generation_optimized import (
    precompute_cut_info,
    best_cut2_set_precomputed
)
from .distributions import Distribution, auto_distribution, create_distribution
import warnings
warnings.simplefilter('ignore')


def MPCGA_while(X, Y, K, max_set=5, import_threshold=0.7, max_split=2, regression_type='binary'):
    """Multi-Path CGA with precomputed sorting

    Args:
        X: feature matrix (n x p)
        Y: response vector (binary: 0/1, multinomial: 0,1,...,K-1)
        K: path length (number of variables to select)
        max_set: maximum number of candidate variables at each step
        import_threshold: threshold for variable importance
        max_split: maximum number of splits to explore (greedy after this)
        regression_type: 'binary' or 'multinomial'

    Returns:
        Dictionary with paths and log-likelihoods
    """
    X = np.array(X)
    Y = np.array(Y)
    n, p = X.shape

    # Determine number of classes
    if regression_type == 'multinomial':
        n_classes = len(np.unique(Y))
        # print(f"[Multinomial Regression] Detected {n_classes} classes")
    else:
        n_classes = 2

    # Pre-create X_original_df once
    X_original_df = pd.DataFrame(X, columns=[f"V{i+1}" for i in range(p)])

    # 預計算排序資訊（固定部分）
    memory_estimate_mb = (p * n * 8) / (1024 * 1024)
    # print(f"[Precomputed Sorting Mode] Precomputing sort info, estimated memory: {memory_estimate_mb:.1f} MB")

    cut_info_cache = precompute_cut_info(X)
    # print(f"  Precomputed sort info for {p} variables")

    # Initialize
    J0 = ['beta0']
    X_intercept = np.ones((n, 1))

    S = deque([(J0, X_intercept, [], 0)])
    P = []

    # Initialize logistic regression model
    # Note: warm_start不适用，因为特征维度在变化
    if regression_type == 'multinomial':
        lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
    else:
        lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')

    while S:
        J, X_current, llik_history, depth = S.popleft()

        # Fit model
        lr.fit(X_current, Y)

        # Get coefficients and compute log-likelihood
        if regression_type == 'multinomial':
            # For multinomial: coef_ is (K-1, p) or (K, p) depending on sklearn version
            # We need to reshape it to (p, K-1) for our functions
            if n_classes == 2:
                # Binary case treated as multinomial
                gamma = lr.coef_.T  # (p, 1)
            else:
                # True multinomial case
                gamma = lr.coef_.T  # (p, K-1) or (p, K)
                if gamma.shape[1] == n_classes:
                    # sklearn returns (K, p), we need (p, K-1)
                    gamma = gamma[:, 1:]  # Remove reference class

            # Compute log-likelihood using multinomial formula
            llik = llik_multinomial(Y, X_current, gamma)
        else:
            # Binary case
            gamma = lr.coef_[0]
            eta = X_current @ gamma
            llik = np.sum(eta * Y - np.log(1 + np.exp(eta)))

        llik_history_new = llik_history + [llik]

        # === Generate candidate features ===
        # 使用預計算的排序資訊，動態選擇切點
        X_cuts_df = best_cut2_set_precomputed(
            cut_info_cache, X_current, Y, gamma,
            import_threshold=import_threshold,
            max_set=max_set,
            regression_type=regression_type
        )

        if X_cuts_df.shape[1] == 0:
            if len(J) > 1:
                P.append({
                    'path': J,
                    'llik': np.array(llik_history_new)
                })
            continue

        # Add original variables to candidate pool
        X_current_df = pd.DataFrame(X_current)
        X_all_df = pd.concat([X_current_df, X_original_df, X_cuts_df], axis=1)
        X_all_df = X_all_df.T.drop_duplicates().T
        X_all = X_all_df.values

        # Current variables are at the beginning
        # After drop_duplicates, some current columns may have been removed
        # (e.g., identical cut indicators in multinomial). Use unique count.
        current_indices = list(range(X_current_df.T.drop_duplicates().shape[0]))

        # Standardization
        X_standardized = X_all.copy()

        if np.allclose(X_standardized[:, 0], 1.0) and X_all.shape[1] > 1:
            means = np.mean(X_standardized[:, 1:], axis=0)
            stds = np.std(X_standardized[:, 1:], axis=0)
            stds[stds < 1e-10] = 1.0
            X_standardized[:, 1:] = (X_standardized[:, 1:] - means) / stds

        # Compute gradient scores for variable importance
        p_all = X_all.shape[1]

        if regression_type == 'multinomial':
            # For multinomial: expand gamma to (p_all, K-1)
            if len(gamma.shape) == 1:
                # Binary case treated as multinomial: gamma is (p,)
                gamma_expanded = np.zeros(p_all)
                min_len = min(len(gamma), p_all)
                gamma_expanded[:min_len] = gamma[:min_len]
                scores = np.abs(fd(Y, X_standardized, gamma_expanded))
            else:
                # True multinomial case: gamma is (p, K-1)
                gamma_expanded = np.zeros((p_all, gamma.shape[1]))
                min_len = min(gamma.shape[0], p_all)
                gamma_expanded[:min_len, :] = gamma[:min_len, :]

                # Use aggregated gradient: sum of absolute values across classes
                scores = fd_multinomial_aggregated(Y, X_standardized, gamma_expanded)
        else:
            # Binary case
            gamma_expanded = np.zeros(p_all)
            min_len = min(len(gamma), p_all)
            gamma_expanded[:min_len] = gamma[:min_len]
            scores = np.abs(fd(Y, X_standardized, gamma_expanded))

        # Set scores of already selected features to 0
        # Note: current_indices was already computed above based on mode
        scores[current_indices] = 0

        # Select top candidates
        max_score = np.max(scores)

        if depth < max_split:
            candidate_indices = np.argsort(scores)[::-1][:min(max_set, len(scores))]
            threshold_mask = scores[candidate_indices] > import_threshold * max_score
            selected_indices = candidate_indices[threshold_mask]
        else:
            selected_indices = [np.argmax(scores)]

        selected_cols = [X_all_df.columns[i] for i in selected_indices]

        # Add to queue or paths
        for col_idx, col_name in zip(selected_indices, selected_cols):
            J_new = J + [col_name]

            if len(J_new) == K + 1:
                X_selected = X_all[:, list(current_indices) + [col_idx]]
                lr.fit(X_selected, Y)

                # Compute final log-likelihood
                if regression_type == 'multinomial':
                    if n_classes == 2:
                        gamma_final = lr.coef_.T
                    else:
                        gamma_final = lr.coef_.T
                        if gamma_final.shape[1] == n_classes:
                            gamma_final = gamma_final[:, 1:]
                    llik_final = llik_multinomial(Y, X_selected, gamma_final)
                else:
                    eta_final = X_selected @ lr.coef_[0]
                    llik_final = np.sum(eta_final * Y - np.log(1 + np.exp(eta_final)))

                P.append({
                    'path': J_new,
                    'llik': np.array(llik_history_new + [llik_final])
                })
            else:
                X_next = X_all[:, list(current_indices) + [col_idx]]
                S.append((J_new, X_next, llik_history_new, depth + 1))

    return {'path': [p['path'] for p in P], 'llik': [p['llik'] for p in P]}


def fit_model_while(X, Y, K=25, c3=0.3, max_set=3, import_threshold=0.7, max_split=2,
                    penalty_type='HDAIC', use_mtrim=False, c2=1.0, regression_type='binary'):
    """Fit MPCGA model using while-loop version

    Args:
        X: feature matrix
        Y: response vector (binary: 0/1, multinomial: 0,1,...,K-1)
        K: number of steps
        c3: penalty coefficient for HDIC
        max_set: maximum number of candidate variables at each step
        import_threshold: threshold for variable importance
        max_split: maximum number of splits to explore
        penalty_type: 'HDAIC', 'HDHQIC', or 'HDBIC'
        use_mtrim: if True, apply Model Trim (MTrim) to remove redundant models
        c2: tuning parameter for MTrim (default: 1.0)
        regression_type: 'binary' or 'multinomial'

    Returns:
        Dictionary with fitted models
    """
    from .mpcga import HDIC_Trim, Model_Trim
    from .cut_generation_optimized import generate_test_cut_all

    X = np.array(X)
    Y = np.array(Y)

    # 保存原始維度（用於 HDIC 懲罰）
    p_original = X.shape[1]

    # Remove duplicate columns
    X_df = pd.DataFrame(X, columns=[f"V{i+1}" for i in range(X.shape[1])])
    X_df = X_df.T.drop_duplicates().T

    # Run MPCGA-while
    cga_output = MPCGA_while(X_df.values, Y, K=K, max_set=max_set,
                             import_threshold=import_threshold, max_split=max_split,
                             regression_type=regression_type)

    # Apply HDIC trimming (使用原始維度)
    output = HDIC_Trim(X_df.values, Y, cga_output, c3=c3, penalty_type=penalty_type,
                         p_original=p_original)

    # Get unique trimmed paths (use 'trim' instead of 'HDIC')
    # Remove duplicates based on variable set (ignoring order and beta0)
    trim_paths = output['trim']
    unique_paths = []
    seen_varsets = []

    for path in trim_paths:
        # Extract variable names (excluding beta0)
        varnames = [name for name in path if isinstance(name, str) and name != 'beta0']
        # Create a frozenset to compare variable sets (order-independent)
        varset = frozenset(varnames)

        # Only add if this variable set hasn't been seen before
        if varset not in seen_varsets:
            unique_paths.append(path)
            seen_varsets.append(varset)

    # === Apply Model Trim (MTrim) if enabled ===
    if use_mtrim and len(unique_paths) > 1:
        print(f"\n[MTrim] Applying Model Trim with c2={c2}")
        print(f"  Candidates before MTrim: {len(unique_paths)}")

        # Apply Model_Trim function
        unique_paths = Model_Trim(X_df.values, Y, unique_paths, c2=c2, regression_type=regression_type)

        print(f"  Candidates after MTrim: {len(unique_paths)}")

    # Build models
    models = []
    vars_list = []
    paths_list = []

    for path in unique_paths:
        all_varnames = [name for name in path if isinstance(name, str) and name != 'beta0']

        # If no variables selected (only intercept), create intercept-only model
        if len(all_varnames) == 0:
            X_new = pd.DataFrame({'int': np.ones(len(Y))})
            if regression_type == 'multinomial':
                lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
            else:
                lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
            lr.fit(X_new, Y)
            models.append(lr)
            vars_list.append([])  # Empty variable list
            paths_list.append(path)
            continue

        # Build features in the exact order they appear in all_varnames
        # This is critical to ensure coefficients match variable names correctly
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

        # Fit model with appropriate regression type
        if regression_type == 'multinomial':
            lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
        else:
            lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
        lr.fit(X_new, Y)

        models.append(lr)
        vars_list.append(all_varnames)
        paths_list.append(path)

    return {'model': models, 'main_var': vars_list, 'path': paths_list}
