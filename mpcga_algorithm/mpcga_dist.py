"""
MPCGA with Distribution Objects - Adapter/Wrapper Layer (Optional)

IMPORTANT: This is a WRAPPER around mpcga_while.py - NOT a separate implementation.

Purpose:
    - Provides an object-oriented interface using Distribution classes
    - Internally calls mpcga_while.fit_model_while() with identical results
    - Designed for API consistency if you need custom distributions

Current Status:
    - NOT used by any simulation scripts (all use mpcga_while.py directly)
    - Tested to produce IDENTICAL results to mpcga_while.py
    - Kept for potential future extensibility

When to use:
    - If you want to define custom distribution families
    - If you prefer OOP interface over function-based interface
    - For API consistency across different distributions

When NOT to use:
    - For standard binary/multinomial logistic regression (use mpcga_while.py directly)
    - For performance-critical applications (extra wrapper layer)

Example Equivalence:
    # These two produce IDENTICAL results:

    # Method 1 (Direct - Recommended):
    from mpcga_algorithm.mpcga_while import fit_model_while
    models = fit_model_while(X, Y, K=25, regression_type='binary')

    # Method 2 (Using Distribution objects):
    from mpcga_algorithm.mpcga_dist import fit_model
    from mpcga_algorithm.distributions import BinaryLogistic
    models = fit_model(X, Y, K=25, distribution=BinaryLogistic())
    # ^ This internally calls fit_model_while() with regression_type='binary'
"""

import numpy as np
from .distributions import Distribution, create_distribution, auto_distribution
from .mpcga_while import fit_model_while as _fit_model_while_legacy


def fit_model(X, Y, K=25, distribution=None, **kwargs):
    """
    Fit MPCGA model using distribution objects

    This is the NEW recommended interface for MPCGA.

    Args:
        X: (n, p) feature matrix
        Y: (n,) response vector
        K: number of steps
        distribution: Distribution object (BinaryLogistic, MultinomialLogistic, or custom)
                     If None, auto-detect based on Y
        **kwargs: Additional arguments:
            - c3: penalty coefficient for HDIC (default: 0.3)
            - max_set: max candidate variables per step (default: 3)
            - import_threshold: variable importance threshold (default: 0.7)
            - max_split: max splits to explore (default: 2)
            - penalty_type: 'HDAIC', 'HDHQIC', or 'HDBIC' (default: 'HDAIC')
            - use_mtrim: whether to use MTrim (default: False)
            - c2: MTrim parameter (default: 1.0)

    Returns:
        Dictionary with fitted models

    Examples:
        # Binary logistic regression
        from mpcga_algorithm.distributions import BinaryLogistic
        dist = BinaryLogistic()
        result = fit_model(X, Y, K=5, distribution=dist)

        # Multinomial logistic regression
        from mpcga_algorithm.distributions import MultinomialLogistic
        dist = MultinomialLogistic()
        result = fit_model(X, Y, K=5, distribution=dist)

        # Auto-detect (recommended for standard use)
        result = fit_model(X, Y, K=5)  # Automatically detects binary/multinomial

        # Custom distribution
        class MyDistribution(Distribution):
            def fit(self, X, Y): ...
            def log_likelihood(self, Y, X, coef): ...
            def gradient(self, Y, X, coef): ...

        dist = MyDistribution()
        result = fit_model(X, Y, K=5, distribution=dist)
    """

    # Auto-detect distribution if not provided
    if distribution is None:
        distribution = auto_distribution(Y)

    # Validate distribution
    if not isinstance(distribution, Distribution):
        raise TypeError(f"distribution must be a Distribution object, got {type(distribution)}")

    # Map distribution to regression_type for legacy interface
    # This is the adaptation layer
    if distribution.get_n_classes() == 2:
        regression_type = 'binary'
    else:
        regression_type = 'multinomial'

    # Call legacy function with regression_type
    # In the future, we can replace this with a fully refactored version
    result = _fit_model_while_legacy(
        X, Y, K=K,
        regression_type=regression_type,
        **kwargs
    )

    # Store distribution object in result for future use
    result['distribution'] = distribution

    return result


# Convenience aliases
fit = fit_model
fit_mpcga = fit_model


def fit_binary(X, Y, K=25, **kwargs):
    """Convenience function for binary logistic regression"""
    from .distributions import BinaryLogistic
    return fit_model(X, Y, K=K, distribution=BinaryLogistic(), **kwargs)


def fit_multinomial(X, Y, K=25, **kwargs):
    """Convenience function for multinomial logistic regression"""
    from .distributions import MultinomialLogistic
    return fit_model(X, Y, K=K, distribution=MultinomialLogistic(), **kwargs)


# For backward compatibility: support old interface
def fit_model_while(X, Y, K=25, regression_type='binary', distribution=None, **kwargs):
    """
    Legacy interface with both old and new parameters

    Prefer using fit_model() for new code.

    Args:
        regression_type: 'binary' or 'multinomial' (legacy, will be deprecated)
        distribution: Distribution object (new, recommended)

    If both are provided, distribution takes precedence.
    """
    import warnings

    if distribution is not None and regression_type != 'binary':
        # Both provided, distribution wins
        warnings.warn(
            "Both distribution and regression_type provided. "
            "Using distribution (regression_type will be ignored).",
            DeprecationWarning
        )

    if distribution is None:
        # Use legacy interface
        if regression_type not in ['binary', 'multinomial']:
            warnings.warn(
                f"regression_type='{regression_type}' may be deprecated soon. "
                "Consider using distribution objects instead.",
                FutureWarning
            )
        distribution = create_distribution(regression_type)

    return fit_model(X, Y, K=K, distribution=distribution, **kwargs)
