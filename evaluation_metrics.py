"""
Evaluation metrics for simulation study
"""

import numpy as np
import re


def extract_variable_indices(selected_vars, p):
    """Extract original variable indices from selected variable names

    Args:
        selected_vars: list of variable names (e.g., ['V1', 'V2_cut_100', 'V3'])
        p: total number of features

    Returns:
        set of variable indices (0-indexed)
    """
    indices = set()

    for var in selected_vars:
        if isinstance(var, str):
            # Extract variable number from patterns like 'V1', 'V2_cut_100', etc.
            match = re.match(r'V(\d+)', var)
            if match:
                var_idx = int(match.group(1)) - 1  # Convert to 0-indexed
                if 0 <= var_idx < p:
                    indices.add(var_idx)

    return indices


def compute_metrics(y_test, y_pred, selected_vars, true_vars, p):
    """Compute all evaluation metrics

    Args:
        y_test: true test labels
        y_pred: predicted labels
        selected_vars: list of selected variable names
        true_vars: list/set of true relevant variable indices (0-indexed)
        p: total number of features

    Returns:
        dict with metrics: accuracy, var_len, exact, e_plus, include
    """
    # 1. Prediction Accuracy
    accuracy = np.mean(y_pred == y_test)

    # Extract variable indices (extracts unique base variables)
    selected_indices = extract_variable_indices(selected_vars, p)
    true_indices = set(true_vars)

    # 2. Number of selected features (count unique base variables)
    # For MPCGA: ['V1_cut_100', 'V1_cut_200', 'V2'] -> 2 unique base variables (V1, V2)
    # For other methods: indices list -> count of indices
    var_len = len(selected_indices)

    # 3. Exact recovery (E)
    exact = 1 if selected_indices == true_indices else 0

    # 4. E+i: all true variables plus i irrelevant ones
    # Calculate i = number of extra irrelevant variables
    num_true_selected = len(selected_indices & true_indices)
    num_false_selected = len(selected_indices - true_indices)

    if num_true_selected == len(true_indices):
        # All true variables are selected
        e_plus_i = num_false_selected  # i = number of irrelevant variables
    else:
        # Not all true variables are selected
        e_plus_i = -1  # Use -1 to indicate failure

    # 5. Include: number of relevant features included
    include = num_true_selected

    return {
        'accuracy': accuracy,
        'var_len': var_len,
        'exact': exact,
        'e_plus_i': e_plus_i,
        'include': include
    }


def summarize_metrics(metrics_list):
    """Summarize metrics across multiple runs

    Args:
        metrics_list: list of metric dictionaries

    Returns:
        dict with mean and std for each metric
    """
    if len(metrics_list) == 0:
        return {}

    # Extract values for each metric
    accuracies = [m['accuracy'] for m in metrics_list]
    var_lens = [m['var_len'] for m in metrics_list]
    exacts = [m['exact'] for m in metrics_list]
    e_plus_is = [m['e_plus_i'] for m in metrics_list if m['e_plus_i'] >= 0]
    includes = [m['include'] for m in metrics_list]

    summary = {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'var_len_mean': np.mean(var_lens),
        'var_len_std': np.std(var_lens),
        'exact_rate': np.mean(exacts),  # Proportion of exact recoveries
        'include_mean': np.mean(includes),
        'include_std': np.std(includes),
    }

    # E+i statistics (only for cases where all true variables are included)
    if len(e_plus_is) > 0:
        summary['e_plus_i_mean'] = np.mean(e_plus_is)
        summary['e_plus_i_std'] = np.std(e_plus_is)
        # Count proportion of each E+i value
        for i in range(6):  # E+0 to E+5
            count = sum(1 for x in e_plus_is if x == i)
            summary[f'e_plus_{i}_rate'] = count / len(metrics_list)
    else:
        summary['e_plus_i_mean'] = np.nan
        summary['e_plus_i_std'] = np.nan

    return summary


def print_metrics_summary(method_name, summary):
    """Pretty print metrics summary

    Args:
        method_name: name of the method
        summary: summary dictionary from summarize_metrics
    """
    print(f"\n{method_name}:")
    print(f"  Accuracy:    {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"  VarLen:      {summary['var_len_mean']:.2f} ± {summary['var_len_std']:.2f}")
    print(f"  Exact (E):   {summary['exact_rate']:.4f}")
    print(f"  Include:     {summary['include_mean']:.2f} ± {summary['include_std']:.2f}")

    if not np.isnan(summary.get('e_plus_i_mean', np.nan)):
        print(f"  E+i:         {summary['e_plus_i_mean']:.2f} ± {summary['e_plus_i_std']:.2f}")
        # Print E+0 to E+5 rates
        e_plus_rates = [summary.get(f'e_plus_{i}_rate', 0) for i in range(6)]
        if sum(e_plus_rates) > 0:
            print(f"  E+0 to E+5:  " + ", ".join([f"{r:.3f}" for r in e_plus_rates]))


if __name__ == "__main__":
    # Test the metrics
    print("Testing evaluation metrics...")

    # Example: true variables are V1, V2, V3 (indices 0, 1, 2)
    true_vars = [0, 1, 2]
    p = 10

    # Test case 1: Exact recovery
    selected_vars_1 = ['V1', 'V2', 'V3']
    y_test = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 1])

    metrics_1 = compute_metrics(y_test, y_pred, selected_vars_1, true_vars, p)
    print(f"\nTest 1 (Exact recovery):")
    print(f"  Selected: {selected_vars_1}")
    print(f"  Metrics: {metrics_1}")

    # Test case 2: E+2 (all true + 2 irrelevant)
    selected_vars_2 = ['V1', 'V2_cut_100', 'V3', 'V5', 'V7']
    metrics_2 = compute_metrics(y_test, y_pred, selected_vars_2, true_vars, p)
    print(f"\nTest 2 (E+2):")
    print(f"  Selected: {selected_vars_2}")
    print(f"  Metrics: {metrics_2}")

    # Test case 3: Missing one true variable
    selected_vars_3 = ['V1', 'V2', 'V5']
    metrics_3 = compute_metrics(y_test, y_pred, selected_vars_3, true_vars, p)
    print(f"\nTest 3 (Missing V3):")
    print(f"  Selected: {selected_vars_3}")
    print(f"  Metrics: {metrics_3}")

    # Test summarization
    print(f"\n\nTesting summarization:")
    metrics_list = [metrics_1, metrics_2, metrics_3]
    summary = summarize_metrics(metrics_list)
    print_metrics_summary("Test Method", summary)
