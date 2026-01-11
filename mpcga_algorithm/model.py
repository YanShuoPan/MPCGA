"""
Model fitting and prediction functions for MPCGA
Handles model training, prediction, and ensemble methods
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from .mpcga import MPCGA_HDIC
from .cut_generation import generate_test_cut_all


def fit_model(X, Y, K=25, c3=0.3, max_set=3, import_threshold=0.95, max_split=2, penalty_type='HDAIC'):
    """Fit MPCGA model

    Args:
        X: feature matrix
        Y: response vector
        K: number of steps in CGA
        c3: penalty coefficient
        max_set: maximum number of candidate variables at each step
        import_threshold: threshold for variable importance
        max_split: maximum number of splits to explore
        penalty_type: 'HDAIC', 'HDHQIC', or 'HDBIC'
            - HDAIC: penalty = c3 * k * log(p)
            - HDHQIC: penalty = c3 * k * log(log(n)) * log(p)
            - HDBIC: penalty = c3 * k * log(n) * log(p)

    Returns:
        Dictionary with fitted models, variable names, and paths
    """
    X = np.array(X)
    Y = np.array(Y)

    # Remove duplicate columns
    X_df = pd.DataFrame(X, columns=[f"V{i+1}" for i in range(X.shape[1])])
    X_df = X_df.T.drop_duplicates().T

    # Run MPCGA
    output = MPCGA_HDIC(X_df.values, Y, K=K, c3=c3, max_set=max_set,
                        import_threshold=import_threshold, max_split=max_split, penalty_type=penalty_type)

    # Get unique HDIC paths (no trim)
    hdic_paths = output['HDIC']
    unique_paths = []
    for path in hdic_paths:
        # Convert to tuple of strings for comparison
        path_str = tuple(str(x) for x in path)
        if path_str not in [tuple(str(x) for x in p) for p in unique_paths]:
            unique_paths.append(path)

    models = []
    vars_list = []
    paths_list = []

    for path in unique_paths:
        # Extract variable names in selection order (preserving path order)
        all_varnames = [name for name in path if isinstance(name, str) and name != 'beta0']

        if len(all_varnames) == 0:
            continue

        # Separate cut and original variables (maintaining order)
        cut_names = [name for name in all_varnames if 'cut' in name]
        original_names = [name for name in all_varnames if 'cut' not in name]

        # Generate features
        if len(cut_names) > 0:
            X_new = generate_test_cut_all(X_df, cut_names, X_df)
        else:
            X_new = pd.DataFrame()

        # Add original variables in the order they appear in all_varnames
        for orig_name in original_names:
            if orig_name in X_df.columns:
                X_new[orig_name] = X_df[orig_name]

        X_new.insert(0, 'int', 1)

        lr = LogisticRegression(fit_intercept=False, max_iter=1000, solver='lbfgs')
        lr.fit(X_new, Y)

        models.append(lr)
        vars_list.append(all_varnames)
        paths_list.append(path)

    return {'model': models, 'main_var': vars_list, 'path': paths_list}


def predict_fun(X, Y, model):
    """Predict and calculate accuracy

    Args:
        X: feature matrix (can be DataFrame or numpy array)
        Y: response vector
        model: fitted logistic regression model

    Returns:
        Dictionary with confusion matrix, accuracy, and predictions
    """
    import warnings

    # Convert to numpy array to avoid feature name mismatch issues
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X

    # Check if model is multinomial
    if hasattr(model, 'multi_class') and model.multi_class == 'multinomial':
        # Multinomial: use argmax on probabilities
        # Suppress feature name warning since we handle feature alignment correctly
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*feature names.*')
            Ypred = model.predict(X_array)
    else:
        # Binary: use threshold 0.5
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*feature names.*')
            prob = model.predict_proba(X_array)[:, 1]
        Ypred = (prob > 0.5).astype(int)

    acc = np.mean(Ypred == Y)
    cm = confusion_matrix(Y, Ypred)

    return {'confusion': cm, 'acc': acc, 'pred': Ypred}


def get_result(X, Y, test, models):
    """Get ensemble prediction from multiple models

    Args:
        X: training feature matrix
        Y: response vector (for training data, used for compatibility)
        test: test feature matrix
        models: dictionary with fitted models and variable names

    Returns:
        Ensemble predictions (majority vote)
    """
    X_df = pd.DataFrame(X, columns=[f"V{i+1}" for i in range(X.shape[1])])
    test_df = pd.DataFrame(test, columns=[f"V{i+1}" for i in range(test.shape[1])])

    # Create dummy Y for test set (not used in prediction, only for shape)
    test_n = test.shape[0]
    Y_dummy = np.zeros(test_n)

    all_result = np.zeros((len(models['model']), test_n))

    for i in range(len(models['model'])):
        # Separate cut and original variables (preserving order)
        varnames = models['main_var'][i]

        # Handle intercept-only model (empty varnames)
        if len(varnames) == 0:
            X_new = pd.DataFrame({'int': np.ones(test.shape[0])})
            result = predict_fun(X_new, Y_dummy, models['model'][i])
            all_result[i, :] = result['pred']
            continue

        cut_names = [name for name in varnames if 'cut' in name]
        original_names = [name for name in varnames if 'cut' not in name]

        # Build features in the exact order they appear in varnames
        X_new_cols = []

        # Add intercept first
        X_new_cols.append(pd.DataFrame({'int': np.ones(test.shape[0])}))

        # Add features in the order they appear in varnames
        for var_name in varnames:
            if 'cut' in var_name:
                # Generate this specific cut variable
                cut_df = generate_test_cut_all(X_df, [var_name], test_df)
                if not cut_df.empty and var_name in cut_df.columns:
                    X_new_cols.append(cut_df[[var_name]])
            else:
                # Add original variable
                if var_name in test_df.columns:
                    X_new_cols.append(test_df[[var_name]])

        # Concatenate all columns
        if len(X_new_cols) > 1:
            X_new = pd.concat(X_new_cols, axis=1)
        else:
            X_new = X_new_cols[0]

        result = predict_fun(X_new, Y_dummy, models['model'][i])
        all_result[i, :] = result['pred']

    # Ensemble prediction
    # Check if models are multinomial
    if hasattr(models['model'][0], 'multi_class') and models['model'][0].multi_class == 'multinomial':
        # Multinomial: use majority vote across all classes
        # For each test sample, count votes for each class
        n_samples = all_result.shape[1]
        n_classes = len(np.unique(all_result))

        final_predictions = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            # Get all predictions for this sample
            votes = all_result[:, i]
            # Count votes for each class and pick the most common
            unique, counts = np.unique(votes, return_counts=True)
            final_predictions[i] = unique[np.argmax(counts)]

        result = final_predictions
    else:
        # Binary: use threshold 0.5 on mean
        result = (np.mean(all_result, axis=0) > 0.5).astype(int)

    return result
