"""
Distribution classes for MPCGA

Each distribution implements:
- fit(X, Y): Fit model and return coefficients
- log_likelihood(Y, X, coef): Compute log-likelihood
- gradient(Y, X, coef): Compute gradient
- gradient_aggregated(Y, X, coef): Compute aggregated gradient for variable importance
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings


class Distribution(ABC):
    """Abstract base class for distributions"""

    @abstractmethod
    def fit(self, X, Y):
        """
        Fit model and return coefficients

        Args:
            X: (n, p) design matrix
            Y: (n,) response vector

        Returns:
            coef: Coefficient array (shape depends on distribution)
        """
        pass

    @abstractmethod
    def log_likelihood(self, Y, X, coef):
        """
        Compute log-likelihood

        Args:
            Y: (n,) response vector
            X: (n, p) design matrix
            coef: Coefficient array

        Returns:
            float: Log-likelihood value
        """
        pass

    @abstractmethod
    def gradient(self, Y, X, coef):
        """
        Compute gradient of log-likelihood

        Args:
            Y: (n,) response vector
            X: (n, p) design matrix
            coef: Coefficient array

        Returns:
            Gradient array (same shape as coef or aggregated)
        """
        pass

    def gradient_aggregated(self, Y, X, coef):
        """
        Compute aggregated gradient for variable importance

        Default implementation: sum of absolute gradients

        Args:
            Y: (n,) response vector
            X: (n, p) design matrix
            coef: Coefficient array

        Returns:
            (p,) array: Aggregated gradient for each variable
        """
        grad = self.gradient(Y, X, coef)
        if grad.ndim == 1:
            return np.abs(grad)
        else:
            # For multinomial: sum absolute values across classes
            return np.sum(np.abs(grad), axis=1)

    def get_n_classes(self):
        """Return number of classes (for classification problems)"""
        return getattr(self, 'n_classes', 2)


class BinaryLogistic(Distribution):
    """Binary Logistic Regression"""

    def __init__(self, max_iter=1000, solver='lbfgs'):
        """
        Args:
            max_iter: Maximum iterations for solver
            solver: Solver to use ('lbfgs', 'newton-cg', etc.)
        """
        self.max_iter = max_iter
        self.solver = solver
        self.model = LogisticRegression(
            fit_intercept=False,
            max_iter=max_iter,
            solver=solver
        )
        self.n_classes = 2

    def fit(self, X, Y):
        """
        Fit binary logistic regression

        Returns:
            coef: (p,) array of coefficients
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.model.fit(X, Y)
        return self.model.coef_[0]

    def log_likelihood(self, Y, X, coef):
        """
        Binary logistic log-likelihood

        Formula: ℓ(β) = Σ [η_i * y_i - log(1 + exp(η_i))]
        where η_i = X_i @ β
        """
        eta = X @ coef
        # Numerical stability: use log1p
        return np.sum(eta * Y - np.log(1 + np.exp(eta)))

    def gradient(self, Y, X, coef):
        """
        Gradient of binary logistic log-likelihood

        Formula: ∇ℓ = X^T @ (y - p)
        where p = sigmoid(X @ β)
        """
        eta = X @ coef
        return -X.T @ Y + X.T @ (1 / (1 + np.exp(-eta)))

    def predict_proba(self, X, coef):
        """Predict probabilities"""
        eta = X @ coef
        p1 = 1 / (1 + np.exp(-eta))
        return np.column_stack([1 - p1, p1])

    def predict(self, X, coef):
        """Predict class labels"""
        return (self.predict_proba(X, coef)[:, 1] > 0.5).astype(int)


class MultinomialLogistic(Distribution):
    """Multinomial Logistic Regression"""

    def __init__(self, max_iter=1000, solver='lbfgs'):
        """
        Args:
            max_iter: Maximum iterations for solver
            solver: Solver to use ('lbfgs', 'newton-cg', etc.)
        """
        self.max_iter = max_iter
        self.solver = solver
        self.model = LogisticRegression(
            fit_intercept=False,
            max_iter=max_iter,
            solver=solver
        )
        self.n_classes = None

    def fit(self, X, Y):
        """
        Fit multinomial logistic regression

        Returns:
            coef: (p, K-1) array for K classes (reference class parameterization)
                  or (p, 1) for K=2 (binary case)
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.model.fit(X, Y)

        self.n_classes = len(np.unique(Y))

        # Handle sklearn's output format
        gamma = self.model.coef_.T  # (p, K) or (p, K-1)

        if self.n_classes == 2:
            # Binary case: sklearn returns (1, p), we return (p, 1)
            return gamma  # (p, 1)
        else:
            # Multinomial case
            if gamma.shape[1] == self.n_classes:
                # sklearn returns (K, p), we need (p, K-1)
                # Remove reference class (class 0)
                return gamma[:, 1:]  # (p, K-1)
            return gamma

    def log_likelihood(self, Y, X, coef):
        """
        Multinomial logistic log-likelihood

        Uses reference class parameterization: β_0 = 0

        Formula:
            For each sample i:
            - If y_i = 0: -log(1 + Σ exp(η_k))
            - If y_i = k: η_k - log(1 + Σ exp(η_k))
        """
        n, p = X.shape

        # Ensure coef is 2D
        if coef.ndim == 1:
            coef = coef.reshape(-1, 1)

        K = coef.shape[1] + 1  # Number of classes

        eta = X @ coef  # (n, K-1)

        # Add reference class (class 0, η_0 = 0)
        eta_with_zero = np.concatenate([np.zeros((n, 1)), eta], axis=1)  # (n, K)

        # Log-sum-exp trick for numerical stability
        max_eta = np.max(eta_with_zero, axis=1, keepdims=True)
        log_sum_exp = max_eta.flatten() + np.log(
            np.sum(np.exp(eta_with_zero - max_eta), axis=1)
        )

        # Compute log-likelihood
        llik = 0
        for i in range(n):
            if Y[i] == 0:
                # Reference class
                llik -= log_sum_exp[i]
            else:
                # Class k = 1, ..., K-1
                llik += eta[i, int(Y[i]) - 1] - log_sum_exp[i]

        return llik

    def gradient(self, Y, X, coef):
        """
        Gradient of multinomial logistic log-likelihood

        Formula: ∇_β_k ℓ = X^T @ (y_k - π_k)
        where π_k = P(Y=k|X) computed via softmax
        """
        n, p = X.shape

        # Ensure coef is 2D
        if coef.ndim == 1:
            coef = coef.reshape(-1, 1)

        K = coef.shape[1] + 1

        eta = X @ coef  # (n, K-1)
        eta_with_zero = np.concatenate([np.zeros((n, 1)), eta], axis=1)  # (n, K)

        # Compute probabilities (softmax)
        max_eta = np.max(eta_with_zero, axis=1, keepdims=True)
        exp_eta = np.exp(eta_with_zero - max_eta)
        probs = exp_eta[:, 1:] / np.sum(exp_eta, axis=1, keepdims=True)  # (n, K-1)

        # Compute gradient for each class
        grad = np.zeros((p, K - 1))
        for k in range(K - 1):
            y_k = (Y == k + 1).astype(float)  # Indicator for class k+1
            grad[:, k] = X.T @ (y_k - probs[:, k])

        # Return squeezed if only one class (K=2)
        return grad.squeeze()

    def predict_proba(self, X, coef):
        """Predict class probabilities"""
        n = X.shape[0]

        if coef.ndim == 1:
            coef = coef.reshape(-1, 1)

        K = coef.shape[1] + 1

        eta = X @ coef
        eta_with_zero = np.concatenate([np.zeros((n, 1)), eta], axis=1)

        # Softmax
        max_eta = np.max(eta_with_zero, axis=1, keepdims=True)
        exp_eta = np.exp(eta_with_zero - max_eta)
        probs = exp_eta / np.sum(exp_eta, axis=1, keepdims=True)

        return probs

    def predict(self, X, coef):
        """Predict class labels"""
        return np.argmax(self.predict_proba(X, coef), axis=1)


# Alias for backward compatibility
class Multinomial(MultinomialLogistic):
    """Alias for MultinomialLogistic"""
    pass


# Factory function for easy creation
def create_distribution(regression_type='binary', **kwargs):
    """
    Factory function to create distribution objects

    Args:
        regression_type: 'binary' or 'multinomial'
        **kwargs: Additional arguments for distribution constructor

    Returns:
        Distribution object
    """
    if regression_type == 'binary':
        return BinaryLogistic(**kwargs)
    elif regression_type == 'multinomial':
        return MultinomialLogistic(**kwargs)
    else:
        raise ValueError(f"Unknown regression_type: {regression_type}")


# Auto-detect distribution based on data
def auto_distribution(Y, **kwargs):
    """
    Automatically detect distribution based on data

    Args:
        Y: Response vector
        **kwargs: Additional arguments for distribution constructor

    Returns:
        Distribution object
    """
    n_classes = len(np.unique(Y))

    if n_classes == 2:
        return BinaryLogistic(**kwargs)
    else:
        return MultinomialLogistic(**kwargs)
