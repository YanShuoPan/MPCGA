"""
Data generation functions for DGP1, DGP2, DGP3
"""

import numpy as np


def generate_correlated_features(n, p, seed=None):
    """
    Generate correlated multivariate normal features

    Covariance structure: cov(xi, xj) = 1/(|i-j|+2) for i≠j
    After generation, randomly permute the columns to randomize which
    variables become the "true" predictors in each replication.

    Args:
        n: number of samples
        p: number of features
        seed: random seed

    Returns:
        n x p array of correlated features (with columns randomly permuted)
    """
    if seed is not None:
        np.random.seed(seed)

    # Construct covariance matrix
    cov_matrix = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i == j:
                cov_matrix[i, j] = 1.0
            else:
                cov_matrix[i, j] = 1.0 / (abs(i - j) + 2)

    # Generate multivariate normal data
    # For large p, use Cholesky decomposition for efficiency
    try:
        L = np.linalg.cholesky(cov_matrix)
        z = np.random.randn(n, p)
        x = z @ L.T
    except np.linalg.LinAlgError:
        # Fallback to eigendecomposition if Cholesky fails
        x = np.random.multivariate_normal(np.zeros(p), cov_matrix, size=n)

    # Randomly permute columns to randomize which variables are "true"
    # This maintains the correlation structure while randomizing variable indices
    perm = np.random.permutation(p)
    x = x[:, perm]

    return x


def generate_data_dgp1(n_train, n_test, p, seed=123, correlated=False):
    """
    Generate data for DGP1 (Linear model)

    Model: logit(P(Y=1)) = 1 + 2*x1 + 2*x2 + 3*x3 + 4*x4 + 2*x5

    Features: Independent standard normal (iid) or correlated features

    Parameters:
    -----------
    n_train : int
        Number of training samples
    n_test : int
        Number of test samples
    p : int
        Total number of variables
    seed : int
        Random seed
    correlated : bool
        If True, generate correlated features using generate_correlated_features()
        If False, generate iid features (default)

    Returns:
    --------
    dict with keys:
        'x' : training features (n_train x p)
        'y' : training labels
        'x_test' : test features (n_test x p)
        'y_test' : test labels
        'true_indices' : indices of true variables (0-indexed)
    """
    np.random.seed(seed)

    # Generate training data - with or without correlation
    if correlated:
        x_train = generate_correlated_features(n_train, p, seed=seed)
    else:
        x_train = np.random.normal(0, 1, (n_train, p))

    # True model: 1 + 2*x1 + 2*x2 + 3*x3 + 4*x4 + 2*x5
    eta_train = 1 + 2*x_train[:, 0] + 2*x_train[:, 1] + 3*x_train[:, 2] + 4*x_train[:, 3] + 2*x_train[:, 4]
    prob_train = 1 / (1 + np.exp(-eta_train))
    y_train = (np.random.uniform(0, 1, n_train) < prob_train).astype(int)

    # Generate test data - with or without correlation
    if correlated:
        x_test = generate_correlated_features(n_test, p, seed=seed+1)
    else:
        x_test = np.random.normal(0, 1, (n_test, p))
    eta_test = 1 + 2*x_test[:, 0] + 2*x_test[:, 1] + 3*x_test[:, 2] + 4*x_test[:, 3] + 2*x_test[:, 4]
    prob_test = 1 / (1 + np.exp(-eta_test))
    y_test = (np.random.uniform(0, 1, n_test) < prob_test).astype(int)

    return {
        'x': x_train,
        'y': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'true_indices': [0, 1, 2, 3, 4],  # V1, V2, V3, V4, V5 (0-indexed)
        'true_probs_test': prob_test  # For "True" model evaluation
    }


def generate_data_dgp2(n_train, n_test, p, seed=123, correlated=False):
    """
    Generate data for DGP2 (Mixed model with cuts)

    Model: logit(P(Y=1)) = 1 + 5*I{|x1|>0.5} - 5*I{|x2|>0.5} + 3*x3 - 3*x4

    Features: Independent standard normal (iid) or correlated features

    Parameters:
    -----------
    n_train : int
        Number of training samples
    n_test : int
        Number of test samples
    p : int
        Total number of variables
    seed : int
        Random seed
    correlated : bool
        If True, generate correlated features using generate_correlated_features()
        If False, generate iid features (default)

    Returns:
    --------
    dict with keys:
        'x' : training features (n_train x p)
        'y' : training labels
        'x_test' : test features (n_test x p)
        'y_test' : test labels
        'true_indices' : indices of true variables (0-indexed)
    """
    np.random.seed(seed)

    # Generate training data - with or without correlation
    if correlated:
        x_train = generate_correlated_features(n_train, p, seed=seed)
    else:
        x_train = np.random.normal(0, 1, (n_train, p))

    # True model: 1 + 5*I{|x1|>0.5} - 5*I{|x2|>0.5} + 3*x3 - 3*x4
    eta_train = (1 +
                 5 * (np.abs(x_train[:, 0]) > 0.5).astype(float) -
                 5 * (np.abs(x_train[:, 1]) > 0.5).astype(float) +
                 3 * x_train[:, 2] -
                 3 * x_train[:, 3])

    prob_train = 1 / (1 + np.exp(-eta_train))
    y_train = (np.random.uniform(0, 1, n_train) < prob_train).astype(int)

    # Generate test data - with or without correlation
    if correlated:
        x_test = generate_correlated_features(n_test, p, seed=seed+1)
    else:
        x_test = np.random.normal(0, 1, (n_test, p))
    eta_test = (1 +
                5 * (np.abs(x_test[:, 0]) > 0.5).astype(float) -
                5 * (np.abs(x_test[:, 1]) > 0.5).astype(float) +
                3 * x_test[:, 2] -
                3 * x_test[:, 3])

    prob_test = 1 / (1 + np.exp(-eta_test))
    y_test = (np.random.uniform(0, 1, n_test) < prob_test).astype(int)

    return {
        'x': x_train,
        'y': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'true_indices': [0, 1, 2, 3],  # V1, V2, V3, V4 (0-indexed)
        'true_probs_test': prob_test  # For "True" model evaluation
    }


def generate_data_dgp3(n_train, n_test, p, seed=123, correlated=False):
    """
    Generate data for DGP3 (Nonlinear model)

    Model: logit(P(Y=1)) = 1 + 4*(x1^2-1) - 4*(x2^2-1) + 4*I{x3>0.5} - 4*I{x4>0.5}

    Features: Independent standard normal (iid) or correlated features

    Parameters:
    -----------
    n_train : int
        Number of training samples
    n_test : int
        Number of test samples
    p : int
        Total number of variables
    seed : int
        Random seed
    correlated : bool
        If True, generate correlated features using generate_correlated_features()
        If False, generate iid features (default)

    Returns:
    --------
    dict with keys:
        'x' : training features (n_train x p)
        'y' : training labels
        'x_test' : test features (n_test x p)
        'y_test' : test labels
        'true_indices' : indices of true variables (0-indexed)
    """
    np.random.seed(seed)

    # Generate training data - with or without correlation
    if correlated:
        x_train = generate_correlated_features(n_train, p, seed=seed)
    else:
        x_train = np.random.normal(0, 1, (n_train, p))

    # True model: 1 + 4*(x1^2-1) - 4*(x2^2-1) + 4*I{x3>0.5} - 4*I{x4>0.5}
    eta_train = (1 +
                 4 * (x_train[:, 0]**2 - 1) -
                 4 * (x_train[:, 1]**2 - 1) +
                 4 * (x_train[:, 2] > 0.5).astype(float) -
                 4 * (x_train[:, 3] > 0.5).astype(float))

    prob_train = 1 / (1 + np.exp(-eta_train))
    y_train = (np.random.uniform(0, 1, n_train) < prob_train).astype(int)

    # Generate test data - with or without correlation
    if correlated:
        x_test = generate_correlated_features(n_test, p, seed=seed+1)
    else:
        x_test = np.random.normal(0, 1, (n_test, p))
    eta_test = (1 +
                4 * (x_test[:, 0]**2 - 1) -
                4 * (x_test[:, 1]**2 - 1) +
                4 * (x_test[:, 2] > 0.5).astype(float) -
                4 * (x_test[:, 3] > 0.5).astype(float))

    prob_test = 1 / (1 + np.exp(-eta_test))
    y_test = (np.random.uniform(0, 1, n_test) < prob_test).astype(int)

    return {
        'x': x_train,
        'y': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'true_indices': [0, 1, 2, 3],  # V1, V2, V3, V4 (0-indexed)
        'true_probs_test': prob_test  # For "True" model evaluation
    }


def generate_data_dgp4(n_train, n_test, p, seed=123, correlated=False):
    """
    Generate data for DGP4 (Multinomial linear model)

    Model: 3-class multinomial logistic regression
    log(P(Y=k) / P(Y=0)) = beta_{k,0} + beta_{k,1}*x1 + beta_{k,2}*x2 + sum_{i=3}^5 beta_{k,i}*x_i

    Class 1: beta = (0.5, 2, 0, 2, 0, 2)  -> uses V1, V3, V5
    Class 2: beta = (0.5, 2, 2, 0, 2, 0)  -> uses V1, V2, V4

    True variables: V1, V2, V3, V4, V5 (union across classes)

    Parameters:
    -----------
    n_train : int
        Number of training samples
    n_test : int
        Number of test samples
    p : int
        Total number of variables (must be >= 5)
    seed : int
        Random seed
    correlated : bool
        If True, generate correlated features using generate_correlated_features()
        If False, generate iid features (default)

    Returns:
    --------
    dict with keys:
        'x' : training features (n_train x p)
        'y' : training labels (values: 0, 1, 2)
        'x_test' : test features (n_test x p)
        'y_test' : test labels
        'true_indices' : indices of true variables (0-indexed)
        'n_classes' : number of classes (3)
    """
    np.random.seed(seed)

    # Generate training data - with or without correlation
    if correlated:
        x_train = generate_correlated_features(n_train, p, seed=seed)
    else:
        x_train = np.random.normal(0, 1, (n_train, p))

    # Compute linear predictors for each class
    # Class 0 is reference (eta_0 = 0)
    # Class 1: 0.5 + 2*x1 + 0*x2 + 2*x3 + 0*x4 + 2*x5
    eta_1_train = 0.5 + 2*x_train[:, 0] + 0*x_train[:, 1] + 2*x_train[:, 2] + 0*x_train[:, 3] + 2*x_train[:, 4]

    # Class 2: 0.5 + 2*x1 + 2*x2 + 0*x3 + 2*x4 + 0*x5
    eta_2_train = 0.5 + 2*x_train[:, 0] + 2*x_train[:, 1] + 0*x_train[:, 2] + 2*x_train[:, 3] + 0*x_train[:, 4]

    # Compute probabilities using softmax
    exp_eta_0 = np.ones(n_train)  # exp(0) = 1
    exp_eta_1 = np.exp(eta_1_train)
    exp_eta_2 = np.exp(eta_2_train)

    sum_exp = exp_eta_0 + exp_eta_1 + exp_eta_2

    prob_0 = exp_eta_0 / sum_exp
    prob_1 = exp_eta_1 / sum_exp
    prob_2 = exp_eta_2 / sum_exp

    # Sample from multinomial distribution
    probs_train = np.column_stack([prob_0, prob_1, prob_2])
    y_train = np.array([np.random.choice([0, 1, 2], p=probs_train[i]) for i in range(n_train)])

    # Generate test data - with or without correlation
    if correlated:
        x_test = generate_correlated_features(n_test, p, seed=seed+1)
    else:
        x_test = np.random.normal(0, 1, (n_test, p))

    eta_1_test = 0.5 + 2*x_test[:, 0] + 0*x_test[:, 1] + 2*x_test[:, 2] + 0*x_test[:, 3] + 2*x_test[:, 4]
    eta_2_test = 0.5 + 2*x_test[:, 0] + 2*x_test[:, 1] + 0*x_test[:, 2] + 2*x_test[:, 3] + 0*x_test[:, 4]

    exp_eta_0_test = np.ones(n_test)
    exp_eta_1_test = np.exp(eta_1_test)
    exp_eta_2_test = np.exp(eta_2_test)

    sum_exp_test = exp_eta_0_test + exp_eta_1_test + exp_eta_2_test

    prob_0_test = exp_eta_0_test / sum_exp_test
    prob_1_test = exp_eta_1_test / sum_exp_test
    prob_2_test = exp_eta_2_test / sum_exp_test

    probs_test = np.column_stack([prob_0_test, prob_1_test, prob_2_test])
    y_test = np.array([np.random.choice([0, 1, 2], p=probs_test[i]) for i in range(n_test)])

    return {
        'x': x_train,
        'y': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'true_indices': [0, 1, 2, 3, 4],  # V1, V2, V3, V4, V5 (0-indexed)
        'n_classes': 3,
        'true_probs_test': probs_test  # For "True" model evaluation
    }


def generate_data_dgp5(n_train, n_test, p, seed=123, correlated=False):
    """
    Generate data for DGP5 (Multinomial nonlinear model)

    Model: 3-class multinomial logistic regression
    log(P(Y=k) / P(Y=0)) = beta_{k,0} + beta_{k,1}*(x1^2-1) + beta_{k,2}*(x2^2-1) + sum_{i=3}^5 beta_{k,i}*x_i

    Class 1: beta = (0.5, 2, 1, 1, 0, 2)  -> uses V1^2, V2^2, V3, V5
    Class 2: beta = (0.5, 1, 2, 0, 2, 1)  -> uses V1^2, V2^2, V4, V5

    True variables: V1, V2, V3, V4, V5 (union across classes)

    Parameters:
    -----------
    n_train : int
        Number of training samples
    n_test : int
        Number of test samples
    p : int
        Total number of variables (must be >= 5)
    seed : int
        Random seed
    correlated : bool
        If True, generate correlated features using generate_correlated_features()
        If False, generate iid features (default)

    Returns:
    --------
    dict with keys:
        'x' : training features (n_train x p)
        'y' : training labels (values: 0, 1, 2)
        'x_test' : test features (n_test x p)
        'y_test' : test labels
        'true_indices' : indices of true variables (0-indexed)
        'n_classes' : number of classes (3)
    """
    np.random.seed(seed)

    # Generate training data - with or without correlation
    if correlated:
        x_train = generate_correlated_features(n_train, p, seed=seed)
    else:
        x_train = np.random.normal(0, 1, (n_train, p))

    # Compute linear predictors for each class
    # Class 0 is reference (eta_0 = 0)
    # Class 1: 0.5 + 2*(x1^2-1) + 1*(x2^2-1) + 1*x3 + 0*x4 + 2*x5
    eta_1_train = (0.5 +
                   2*(x_train[:, 0]**2 - 1) +
                   1*(x_train[:, 1]**2 - 1) +
                   1*x_train[:, 2] +
                   0*x_train[:, 3] +
                   2*x_train[:, 4])

    # Class 2: 0.5 + 1*(x1^2-1) + 2*(x2^2-1) + 0*x3 + 2*x4 + 1*x5
    eta_2_train = (0.5 +
                   1*(x_train[:, 0]**2 - 1) +
                   2*(x_train[:, 1]**2 - 1) +
                   0*x_train[:, 2] +
                   2*x_train[:, 3] +
                   1*x_train[:, 4])

    # Compute probabilities using softmax
    exp_eta_0 = np.ones(n_train)
    exp_eta_1 = np.exp(eta_1_train)
    exp_eta_2 = np.exp(eta_2_train)

    sum_exp = exp_eta_0 + exp_eta_1 + exp_eta_2

    prob_0 = exp_eta_0 / sum_exp
    prob_1 = exp_eta_1 / sum_exp
    prob_2 = exp_eta_2 / sum_exp

    # Sample from multinomial distribution
    probs_train = np.column_stack([prob_0, prob_1, prob_2])
    y_train = np.array([np.random.choice([0, 1, 2], p=probs_train[i]) for i in range(n_train)])

    # Generate test data - with or without correlation
    if correlated:
        x_test = generate_correlated_features(n_test, p, seed=seed+1)
    else:
        x_test = np.random.normal(0, 1, (n_test, p))

    eta_1_test = (0.5 +
                  2*(x_test[:, 0]**2 - 1) +
                  1*(x_test[:, 1]**2 - 1) +
                  1*x_test[:, 2] +
                  0*x_test[:, 3] +
                  2*x_test[:, 4])

    eta_2_test = (0.5 +
                  1*(x_test[:, 0]**2 - 1) +
                  2*(x_test[:, 1]**2 - 1) +
                  0*x_test[:, 2] +
                  2*x_test[:, 3] +
                  1*x_test[:, 4])

    exp_eta_0_test = np.ones(n_test)
    exp_eta_1_test = np.exp(eta_1_test)
    exp_eta_2_test = np.exp(eta_2_test)

    sum_exp_test = exp_eta_0_test + exp_eta_1_test + exp_eta_2_test

    prob_0_test = exp_eta_0_test / sum_exp_test
    prob_1_test = exp_eta_1_test / sum_exp_test
    prob_2_test = exp_eta_2_test / sum_exp_test

    probs_test = np.column_stack([prob_0_test, prob_1_test, prob_2_test])
    y_test = np.array([np.random.choice([0, 1, 2], p=probs_test[i]) for i in range(n_test)])

    return {
        'x': x_train,
        'y': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'true_indices': [0, 1, 2, 3, 4],  # V1, V2, V3, V4, V5 (0-indexed)
        'n_classes': 3,
        'true_probs_test': probs_test  # For "True" model evaluation
    }
