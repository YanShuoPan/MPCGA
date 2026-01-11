"""
Utility functions for MPCGA algorithm
Contains basic mathematical and helper functions
"""

import numpy as np
from scipy.optimize import minimize


def llik(y, G, gam):
    """Log-likelihood for logistic regression

    Args:
        y: response vector
        G: design matrix
        gam: coefficients

    Returns:
        Log-likelihood value
    """
    eta = G @ gam
    return np.sum(eta * (y == 1) - np.log(1 + np.exp(eta)))


def fd(y, G, gam):
    """First derivative (gradient) of log-likelihood

    Args:
        y: response vector
        G: design matrix
        gam: coefficients

    Returns:
        Gradient vector
    """
    eta = G @ gam
    return -G.T @ (y == 1).astype(float) + G.T @ (1 / (1 + np.exp(-eta)))


def MLE(y, G):
    """Maximum likelihood estimation using scipy

    Args:
        y: response vector
        G: design matrix

    Returns:
        Estimated coefficients
    """
    gam0 = np.concatenate([[1], np.zeros(G.shape[1] - 1)])

    def neg_llik(gamma):
        return -llik(y, G, gamma)

    res = minimize(neg_llik, gam0, method='BFGS')
    return res.x


def gam_all(gam, Jhat, p):
    """Expand gamma to full dimension

    Args:
        gam: current coefficients
        Jhat: selected indices
        p: total number of features

    Returns:
        Expanded coefficient vector
    """
    new_gam = np.zeros(p)
    if len(Jhat) == len(gam):
        new_gam[Jhat] = gam
    else:
        # If lengths don't match, use the minimum length
        min_len = min(len(Jhat), len(gam))
        new_gam[Jhat[:min_len]] = gam[:min_len]
    return new_gam


def replace_fun(arr, k, j):
    """Replace element at index k with j

    Args:
        arr: input array
        k: index to replace
        j: new value

    Returns:
        Copy of array with replaced value
    """
    arr_copy = arr.copy()
    arr_copy[k] = j
    return arr_copy


# ============================================================================
# Multinomial Logistic Regression Functions
# ============================================================================

def llik_multinomial(y, X, beta):
    """Log-likelihood for multinomial logistic regression

    Args:
        y: (n,) array of class labels (0, 1, ..., K-1)
        X: (n, p) design matrix
        beta: (p, K-1) coefficient matrix (class 0 is reference)

    Returns:
        Log-likelihood value
    """
    n, p = X.shape
    K = beta.shape[1] + 1  # Number of classes

    # Compute linear predictors for classes 1, ..., K-1
    eta = X @ beta  # (n, K-1)

    # Compute log-sum-exp for numerical stability
    # log(1 + sum(exp(eta_k))) for each sample
    # Need to include class 0 (reference, eta_0 = 0)
    eta_with_zero = np.concatenate([np.zeros((n, 1)), eta], axis=1)  # (n, K)
    max_eta = np.max(eta_with_zero, axis=1, keepdims=True)  # For numerical stability
    log_sum_exp = max_eta.flatten() + np.log(np.sum(np.exp(eta_with_zero - max_eta), axis=1))

    # Compute log-likelihood
    llik = 0
    for i in range(n):
        if y[i] == 0:  # Reference class
            llik -= log_sum_exp[i]
        else:  # Class k = 1, ..., K-1
            llik += eta[i, int(y[i]) - 1] - log_sum_exp[i]

    return llik


def fd_multinomial(y, X, beta):
    """Gradient of log-likelihood for multinomial logistic regression

    Args:
        y: (n,) array of class labels (0, 1, ..., K-1)
        X: (n, p) design matrix
        beta: (p, K-1) coefficient matrix

    Returns:
        Gradient matrix (p, K-1)
    """
    n, p = X.shape
    K = beta.shape[1] + 1

    # Compute probabilities
    eta = X @ beta  # (n, K-1)

    # Numerical stability: subtract max
    # Include class 0 (reference, eta_0 = 0)
    eta_with_zero = np.concatenate([np.zeros((n, 1)), eta], axis=1)  # (n, K)
    max_eta = np.max(eta_with_zero, axis=1, keepdims=True)
    exp_eta_all = np.exp(eta_with_zero - max_eta)  # (n, K)
    sum_exp = np.sum(exp_eta_all, axis=1)  # (n,)

    probs = exp_eta_all[:, 1:] / sum_exp.reshape(-1, 1)  # (n, K-1), exclude class 0

    # Compute gradient for each class
    grad = np.zeros((p, K - 1))
    for k in range(K - 1):
        y_k = (y == k + 1).astype(float)  # Indicator for class k+1
        grad[:, k] = X.T @ (y_k - probs[:, k])

    return grad


def fd_multinomial_aggregated(y, X, beta):
    """Aggregated gradient for variable importance in multinomial regression

    For variable selection in MPCGA, we need a single importance score per variable.
    This function computes the gradient for each class, takes absolute values,
    and sums across classes for each variable.

    Args:
        y: (n,) array of class labels (0, 1, ..., K-1)
        X: (n, p) design matrix
        beta: (p, K-1) coefficient matrix

    Returns:
        Aggregated gradient vector (p,) - sum of absolute gradients across classes
    """
    grad = fd_multinomial(y, X, beta)  # (p, K-1)

    # Sum absolute values across classes for each variable
    aggregated_grad = np.sum(np.abs(grad), axis=1)  # (p,)

    return aggregated_grad


def predict_proba_multinomial(X, beta):
    """Compute class probabilities for multinomial logistic regression

    Args:
        X: (n, p) design matrix
        beta: (p, K-1) coefficient matrix

    Returns:
        Probability matrix (n, K) where each row sums to 1
    """
    n = X.shape[0]
    K = beta.shape[1] + 1

    # Compute linear predictors
    eta = X @ beta  # (n, K-1)

    # Numerical stability
    max_eta = np.max(eta, axis=1, keepdims=True)
    exp_eta = np.exp(eta - max_eta)
    exp_0 = np.exp(-max_eta.flatten())  # For reference class
    sum_exp = exp_0 + np.sum(exp_eta, axis=1)

    # Compute probabilities
    probs = np.zeros((n, K))
    probs[:, 0] = exp_0 / sum_exp  # Reference class
    probs[:, 1:] = exp_eta / sum_exp.reshape(-1, 1)  # Other classes

    return probs
