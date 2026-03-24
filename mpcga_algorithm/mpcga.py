"""
MPCGA (Multi-Path Coordinate Gradient Ascent) algorithm
Tree-based multi-path exploration for feature selection
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .utils import fd
from .cut_generation_optimized import best_cut2_set_precomputed as best_cut2_set, generate_test_cut_all


def CGA_tree2(X, X_current, Y, gam, likelihood, Jhat, k, K, max_set=3, import_threshold=0.8, max_split=5):
    """CGA tree algorithm (recursive multi-path exploration)

    Args:
        X: feature matrix
        X_current: current design matrix
        Y: response vector
        gam: current coefficients
        likelihood: current log-likelihood
        Jhat: selected variable names
        k: current step
        K: total number of steps
        max_set: maximum number of candidate variables
        import_threshold: threshold for variable importance
        max_split: maximum number of splits to explore

    Returns:
        List of dictionaries with path and log-likelihood
    """
    X = np.array(X)
    Y = np.array(Y)
    X_current = np.array(X_current)

    if isinstance(Jhat, list):
        Jhat_names = Jhat
    else:
        Jhat_names = [f"col_{j}" for j in Jhat if not np.isnan(j)]

    # Jhat_idx is just all current columns (0 to X_current.shape[1]-1)
    Jhat_idx = list(range(X_current.shape[1]))

    # Generate new cuts
    # gam should match the number of columns in X_current
    X_new_df = best_cut2_set(X, X_current, Y, gam,
                              import_threshold=import_threshold, max_set=max_set)

    if X_new_df.shape[1] == 0:
        return [{'path': Jhat_names, 'llik': likelihood}]

    # Include both original variables (X) and cut variables (X_new_df)
    X_current_df = pd.DataFrame(X_current)
    X_original_df = pd.DataFrame(X, columns=[f"V{i+1}" for i in range(X.shape[1])])
    X_new_full = pd.concat([X_current_df, X_original_df, X_new_df], axis=1)
    X_new_full = X_new_full.T.drop_duplicates().T
    X_new = X_new_full.values

    # Find current feature indices in new matrix
    Jhat_idx_new = list(range(X_current.shape[1]))

    # Standardize X_new (except intercept column)
    # Assume first column is intercept if it's all ones
    X_new_standardized = X_new.copy()

    if np.allclose(X_new[:, 0], 1.0) and X_new.shape[1] > 1:
        # First column is intercept, standardize columns 1 onwards
        means = np.mean(X_new[:, 1:], axis=0)
        stds = np.std(X_new[:, 1:], axis=0)
        X_new_standardized[:, 1:] = (X_new[:, 1:] - means) / stds
    else:
        # No intercept column, standardize all columns
        means = np.mean(X_new, axis=0)
        stds = np.std(X_new, axis=0)
        X_new_standardized = (X_new - means) / stds

    p = X_new.shape[1]

    # Calculate gradient - expand gam to new dimension
    gam_expanded = np.zeros(p)
    min_len = min(len(gam), p)
    gam_expanded[:min_len] = gam[:min_len]
    rq = np.abs(fd(Y, X_new_standardized, gam_expanded))
    rq[Jhat_idx_new] = 0

    max_rq = np.max(rq)

    if k <= max_split:
        approx_max_rq_idx = np.argsort(rq)[::-1][:min(max_set, len(rq))]
        import_idx = rq[approx_max_rq_idx] > import_threshold * max_rq
        potential_col_indices = approx_max_rq_idx[import_idx]
    else:
        potential_col_indices = [np.argmax(rq)]

    potential_cols = [X_new_full.columns[i] for i in potential_col_indices]

    k = k + 1

    if k >= K + 1:
        results = []
        for col in potential_cols:
            new_Jhat = Jhat_names + [col]
            col_idx = list(X_new_full.columns).index(col)
            X_selected = X_new[:, Jhat_idx_new + [col_idx]]

            lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
            lr.fit(X_selected, Y)

            eta = X_selected @ lr.coef_[0]
            llik = np.sum(eta * Y - np.log(1 + np.exp(eta)))

            new_likelihood = np.append(likelihood, llik)
            results.append({'path': new_Jhat, 'llik': new_likelihood})
        return results

    # Recursive call
    output = []
    for col in potential_cols:
        new_Jhat = Jhat_names + [col]
        col_idx = list(X_new_full.columns).index(col)
        X_selected = X_new[:, Jhat_idx_new + [col_idx]]

        lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
        lr.fit(X_selected, Y)
        gam_new = lr.coef_[0]

        eta = X_selected @ gam_new
        llik = np.sum(eta * Y - np.log(1 + np.exp(eta)))
        new_likelihood = np.append(likelihood, llik)

        result = CGA_tree2(X, X_selected, Y, gam_new, new_likelihood, new_Jhat, k, K,
                          max_set, import_threshold, max_split)
        output.extend(result)

    return output


def MPCGA(X, Y, K, max_set=5, import_threshold=0.7, max_split=2):
    """Multi-Path CGA algorithm

    Args:
        X: feature matrix
        Y: response vector
        K: number of steps
        max_set: maximum number of candidate variables
        import_threshold: threshold for variable importance
        max_split: maximum number of splits to explore

    Returns:
        Dictionary with paths and log-likelihoods
    """
    X = np.array(X)
    Y = np.array(Y)
    n = X.shape[0]

    # Initialize
    Jhat = ['beta0']
    likelihood = np.zeros(1)

    X_current = np.ones((n, 1))

    lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
    lr.fit(X_current, Y)
    gam = lr.coef_[0]

    eta = X_current @ gam
    likelihood[0] = np.sum(eta * Y - np.log(1 + np.exp(eta)))

    # Run CGA tree
    all_CGA = CGA_tree2(X, X_current, Y, gam, likelihood, Jhat, k=1, K=K,
                        max_set=max_set, import_threshold=import_threshold, max_split=max_split)

    # Extract paths and likelihoods
    paths = [item['path'] for item in all_CGA]
    lls = [item['llik'] for item in all_CGA]

    return {'path': paths, 'llik': lls}


def HDIC_Trim(X, Y, CGA_output, c3=1, penalty_type='HDAIC', p_original=None):
    """HDIC trimming for multiple paths

    Args:
        X: feature matrix
        Y: response vector
        CGA_output: output from MPCGA algorithm
        c3: penalty coefficient
        penalty_type: 'HDAIC', 'HDHQIC', or 'HDBIC'
            - HDAIC: penalty = c3 * k * log(p)
            - HDHQIC: penalty = c3 * k * log(log(n)) * log(p)
            - HDBIC: penalty = c3 * k * log(n) * log(p)
            For multinomial (K_classes > 2), the number of parameters per variable
            is multiplied by (K_classes - 1)
        p_original: optional, original number of variables (for penalty calculation)

    Returns:
        Dictionary with paths, HDIC, and trim results
    """
    X = np.array(X)
    Y = np.array(Y)
    n = X.shape[0]
    X_df = pd.DataFrame(X, columns=[f"V{i+1}" for i in range(X.shape[1])])

    paths = CGA_output['path']
    lls = CGA_output['llik']

    Jhat_hdic = []
    Jhat_trim = []

    for i in range(len(paths)):
        path = paths[i]
        likelihood = lls[i]

        # Generate features: both cut variables and original variables
        cut_names = [name for name in path if isinstance(name, str) and 'cut' in name]
        original_names = [name for name in path if isinstance(name, str) and 'cut' not in name and name != 'beta0']

        if len(cut_names) > 0:
            X_new = generate_test_cut_all(X_df, cut_names, X_df)
        else:
            X_new = pd.DataFrame()

        # Add original variables
        for orig_name in original_names:
            if orig_name in X_df.columns:
                X_new[orig_name] = X_df[orig_name]

        X_new.insert(0, 'int', 1)

        K = len(path)

        # Calculate penalty based on penalty_type
        # Use p_original if provided, otherwise use X.shape[1]
        p_for_penalty = p_original if p_original is not None else X.shape[1]

        # Use np.arange(K) for number of variables at each step
        k_vars = np.arange(K)

        if penalty_type == 'HDBIC':
            penalty = c3 * k_vars * np.log(n) * np.log(p_for_penalty)
        elif penalty_type == 'HDHQIC':
            penalty = c3 * k_vars * np.log(np.log(n)) * np.log(p_for_penalty)
        else:  # HDAIC
            penalty = c3 * k_vars * np.log(p_for_penalty)

        hdic = -2 * likelihood + penalty
        kn_hat = np.argmin(hdic)

        # Only keep HDIC selection, no trimming
        # Both HDIC and trim now return the same result (path[:kn_hat+1])
        Jhat_hdic.append(path[:kn_hat+1])
        Jhat_trim.append(path[:kn_hat+1])  # Same as HDIC, no trimming applied

    return {'path': paths, 'HDIC': Jhat_hdic, 'trim': Jhat_trim}


def MPCGA_HDIC(X, Y, K=25, c3=1.5, max_set=5, import_threshold=0.7, max_split=5, penalty_type='HDAIC'):
    """Complete MPCGA with HDIC selection

    Args:
        X: feature matrix
        Y: response vector
        K: number of steps in CGA
        c3: penalty coefficient
        max_set: maximum number of candidate variables at each step
        import_threshold: threshold for variable importance
        max_split: maximum number of splits to explore
        penalty_type: 'HDAIC' or 'HDBIC'

    Returns:
        Dictionary with HDIC and trim results
    """
    CGA_output = MPCGA(X, Y, K, max_set=max_set, import_threshold=import_threshold, max_split=max_split)
    output = HDIC_Trim(X, Y, CGA_output, c3=c3, penalty_type=penalty_type)
    return output


def Model_Trim(X, Y, trimmed_paths, c2=1.0, regression_type='binary', return_info=False):
    """Apply Model Trim (MTrim) to remove redundant models

    Removes models that are clearly inferior in terms of loss and model size.
    A model is retained if:
        loss_diff <= c2 * max(1, |J*| - |J_m|)

    where J* is the best model (minimum loss) and J_m is the candidate model.

    Args:
        X: feature matrix (n x p)
        Y: response vector (binary: 0/1, multinomial: 0,1,...,K-1)
        trimmed_paths: list of paths after HDIC trimming
        c2: tuning parameter for MTrim (default: 1.0)
            - Larger c2: more lenient, keeps more models
            - Smaller c2: stricter, keeps fewer models
        regression_type: 'binary' or 'multinomial'
        return_info: if True, return detailed information about MTrim process

    Returns:
        If return_info=False: List of paths that pass the MTrim criterion
        If return_info=True: (kept_paths, mtrim_info) where mtrim_info is a dict with:
            - n_before: number of models before MTrim
            - n_after: number of models after MTrim
            - n_removed: number of models removed
            - removal_rate: proportion of models removed
            - best_model: info about the best model
            - kept_models: list of kept model info
            - removed_models: list of removed model info
    """
    from .cut_generation_optimized import generate_test_cut_all
    from .utils import llik_multinomial
    from sklearn.linear_model import LogisticRegression

    if len(trimmed_paths) <= 1:
        # If only 0 or 1 model, no need to trim
        return trimmed_paths

    X = np.array(X)
    Y = np.array(Y)
    X_df = pd.DataFrame(X, columns=[f"V{i+1}" for i in range(X.shape[1])])

    # Detect regression type if not specified
    n_classes = len(np.unique(Y))
    if n_classes > 2 and regression_type == 'binary':
        regression_type = 'multinomial'

    # Step 1: Fit all models and compute losses
    path_losses = []
    path_sizes = []

    for path in trimmed_paths:
        varnames = [name for name in path if isinstance(name, str) and name != 'beta0']

        # Build model for this path
        if len(varnames) == 0:
            X_temp = pd.DataFrame({'int': np.ones(len(Y))})
        else:
            cut_names = [name for name in varnames if 'cut' in name]
            original_names = [name for name in varnames if 'cut' not in name]

            if len(cut_names) > 0:
                X_temp = generate_test_cut_all(X_df, cut_names, X_df)
            else:
                X_temp = pd.DataFrame()

            for orig_name in original_names:
                if orig_name in X_df.columns:
                    X_temp[orig_name] = X_df[orig_name]

            X_temp.insert(0, 'int', 1)

        # Fit and compute loss (negative log-likelihood)
        if regression_type == 'multinomial':
            lr_temp = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
            lr_temp.fit(X_temp, Y)

            # Compute multinomial log-likelihood
            gamma = lr_temp.coef_.T  # (p, K-1) or (p, K)
            if gamma.shape[1] == n_classes:
                gamma = gamma[:, 1:]  # Remove reference class
            loss = -llik_multinomial(Y, X_temp.values, gamma)
        else:
            # Binary logistic regression
            lr_temp = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
            lr_temp.fit(X_temp, Y)
            eta = X_temp.values @ lr_temp.coef_[0]
            loss = -np.sum(eta * Y - np.log(1 + np.exp(eta)))

        path_losses.append(loss)
        path_sizes.append(len(varnames))

    # Step 2: Find best model (minimum loss)
    min_loss_idx = np.argmin(path_losses)
    min_loss = path_losses[min_loss_idx]
    best_size = path_sizes[min_loss_idx]

    # Step 3: Apply MTrim criterion
    kept_paths = []
    kept_models_info = []
    removed_models_info = []

    for i, (loss, size) in enumerate(zip(path_losses, path_sizes)):
        loss_diff = loss - min_loss
        tolerance = c2 * max(1, size - best_size)

        varnames = [name for name in trimmed_paths[i] if isinstance(name, str) and name != 'beta0']

        if loss_diff <= tolerance:
            kept_paths.append(trimmed_paths[i])
            kept_models_info.append({
                'index': i,
                'loss': loss,
                'size': size,
                'loss_diff': loss_diff,
                'tolerance': tolerance,
                'variables': varnames
            })
        else:
            removed_models_info.append({
                'index': i,
                'loss': loss,
                'size': size,
                'loss_diff': loss_diff,
                'tolerance': tolerance,
                'variables': varnames
            })

    # Prepare detailed information if requested
    if return_info:
        n_before = len(trimmed_paths)
        n_after = len(kept_paths)
        n_removed = n_before - n_after

        varnames_best = [name for name in trimmed_paths[min_loss_idx] if isinstance(name, str) and name != 'beta0']

        mtrim_info = {
            'n_before': n_before,
            'n_after': n_after,
            'n_removed': n_removed,
            'removal_rate': n_removed / n_before if n_before > 0 else 0,
            'c2': c2,
            'best_model': {
                'index': min_loss_idx,
                'loss': min_loss,
                'size': best_size,
                'variables': varnames_best
            },
            'kept_models': kept_models_info,
            'removed_models': removed_models_info
        }

        return kept_paths, mtrim_info

    return kept_paths
