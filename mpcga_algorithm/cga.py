"""
CGA (Coordinate Gradient Ascent) algorithm for feature selection
Standard greedy forward selection approach for logistic regression
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .utils import fd, gam_all


def CGA(X, Y, K):
    """Coordinate Gradient Ascent algorithm

    Args:
        X: feature matrix
        Y: response vector
        K: number of steps

    Returns:
        Dictionary with path and log-likelihood
    """
    n = X.shape[0]
    X = np.array(X)
    Y = np.array(Y)

    likelihood = np.zeros(K + 1)
    X_all = np.column_stack([np.ones(n), X])
    p = X_all.shape[1]
    X_current = X_all[:, 0].reshape(-1, 1)

    Jhat = np.zeros(K + 1, dtype=int)
    Jhat[0] = 0

    # Fit initial model
    lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
    lr.fit(X_current, Y)
    gam = lr.coef_[0]

    # Calculate initial likelihood
    eta = X_current @ gam
    likelihood[0] = np.sum(eta * Y - np.log(1 + np.exp(eta)))

    for k in range(K):
        # Calculate gradient
        rq = np.abs(fd(Y, X_all, gam_all(gam, Jhat[:k+1], p)))
        rq[Jhat[:k+1]] = 0

        jmax = np.argmax(rq)
        Jhat[k + 1] = jmax

        X_current = X_all[:, Jhat[:k+2]]

        # Fit model
        lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
        lr.fit(X_current, Y)
        gam = lr.coef_[0]

        # Calculate likelihood
        eta = X_current @ gam
        likelihood[k + 1] = np.sum(eta * Y - np.log(1 + np.exp(eta)))

    return {'path': Jhat, 'llik': likelihood}


def CGA_HDIC_Trim(X, Y, CGA_output, c3=1, trim=True, penalty_type='HDAIC'):
    """CGA HDIC model selection with trimming

    Args:
        X: feature matrix
        Y: response vector
        CGA_output: output from CGA algorithm
        c3: penalty coefficient
        trim: whether to trim variables
        penalty_type: 'HDAIC', 'HDHQIC', or 'HDBIC'
            - HDAIC: penalty = c3 * k * log(p)
            - HDHQIC: penalty = c3 * k * log(log(n)) * log(p)
            - HDBIC: penalty = c3 * k * log(n) * log(p)

    Returns:
        Dictionary with path, HDIC, and trim results
    """
    X = np.array(X)
    Y = np.array(Y)
    n = X.shape[0]

    X_all = np.column_stack([np.ones(n), X])
    Jhat = CGA_output['path']
    likelihood = CGA_output['llik']
    K = len(likelihood)

    # Calculate penalty based on penalty_type
    if penalty_type == 'HDBIC':
        penalty = c3 * np.arange(K) * np.log(n) * np.log(X_all.shape[1])
    elif penalty_type == 'HDHQIC':
        penalty = c3 * np.arange(K) * np.log(np.log(n)) * np.log(X_all.shape[1])
    else:  # HDAIC
        penalty = c3 * np.arange(K) * np.log(X_all.shape[1])

    # Calculate HDIC
    hdic = -2 * likelihood + penalty
    kn_hat = np.argmin(hdic)

    if not trim:
        return {'path': CGA_output['path'], 'HDIC': Jhat[:kn_hat+1]}

    # Trimming
    benchmark = hdic[kn_hat]
    trim_pos = []

    if kn_hat > 1:
        for i in range(1, kn_hat + 1):
            JDrop = np.delete(Jhat[:kn_hat+1], i)
            X_current = X_all[:, JDrop]

            lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
            lr.fit(X_current, Y)

            eta = X_current @ lr.coef_[0]
            llik = np.sum(eta * Y - np.log(1 + np.exp(eta)))

            # Calculate penalty for dropped model
            if penalty_type == 'HDBIC':
                penalty_drop = c3 * len(JDrop) * np.log(n) * np.log(X_all.shape[1])
            elif penalty_type == 'HDHQIC':
                penalty_drop = c3 * len(JDrop) * np.log(np.log(n)) * np.log(X_all.shape[1])
            else:  # HDAIC
                penalty_drop = c3 * len(JDrop) * np.log(X_all.shape[1])

            hdic_drop = -2 * llik + penalty_drop

            if hdic_drop < benchmark:
                trim_pos.append(i)

    Jhat_trim = Jhat[:kn_hat+1]
    if len(trim_pos) > 0:
        Jhat_trim = np.delete(Jhat_trim, trim_pos)

    return {'path': CGA_output['path'], 'HDIC': Jhat[:kn_hat+1], 'trim': Jhat_trim}


def fit_model_cga(X, Y, K=25, c3=1, penalty_type='HDBIC'):
    """Fit CGA model with HDIC selection

    Args:
        X: feature matrix
        Y: response vector
        K: number of steps
        c3: penalty coefficient
        penalty_type: 'HDAIC', 'HDHQIC', or 'HDBIC'

    Returns:
        Dictionary with fitted model, selected variables, and path
    """
    X = np.array(X)
    Y = np.array(Y)
    n = X.shape[0]

    # Remove duplicate columns
    X_df = pd.DataFrame(X, columns=[f"V{i+1}" for i in range(X.shape[1])])
    X_df = X_df.T.drop_duplicates().T

    # Run CGA
    cga_output = CGA(X_df.values, Y, K=K)

    # Apply HDIC trimming
    output = CGA_HDIC_Trim(X_df.values, Y, cga_output, c3=c3, trim=True, penalty_type=penalty_type)

    # Get trimmed path (indices)
    trim_path = output['trim']

    # Build the model
    X_all = np.column_stack([np.ones(n), X_df.values])
    X_selected = X_all[:, trim_path]

    # Fit final model
    lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
    lr.fit(X_selected, Y)

    # Extract variable names (excluding intercept at index 0)
    selected_vars = []
    for idx in trim_path:
        if idx > 0:  # Skip intercept
            selected_vars.append(f"V{idx}")

    return {
        'model': [lr],
        'main_var': [selected_vars],
        'path': [trim_path],
        'trim_indices': trim_path  # Store indices for prediction
    }


def predict_cga(X_train, y_test, X_test, model_output):
    """Predict using CGA model

    Args:
        X_train: training feature matrix (used to determine column structure)
        y_test: test response vector
        X_test: test feature matrix
        model_output: output from fit_model_cga

    Returns:
        Predictions array
    """
    model = model_output['model'][0]
    trim_indices = model_output['trim_indices']

    # Build test matrix with intercept
    n_test = X_test.shape[0]
    X_test_all = np.column_stack([np.ones(n_test), X_test])
    X_test_selected = X_test_all[:, trim_indices]

    # Predict
    prob = model.predict_proba(X_test_selected)[:, 1]
    predictions = (prob > 0.5).astype(int)

    return predictions
