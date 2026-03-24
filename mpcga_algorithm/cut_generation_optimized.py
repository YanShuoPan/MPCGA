"""
Optimized cut generation with precomputed sorting
預計算排序資訊（固定部分），但切點選擇依賴當前模型（動態部分）

优化版本:
- v1: 共享残差计算 (避免重复矩阵乘法)
- v2: 向量化操作 (避免DataFrame多次concat)
"""

import numpy as np
import pandas as pd
from .utils import fd


def precompute_cut_info(X):
    """預計算所有變數的固定資訊（不依賴模型）

    Args:
        X: feature matrix (n x p)

    Returns:
        Dictionary with precomputed info for each variable
    """
    n, p = X.shape
    cut_info_cache = {}

    for i in range(p):
        v = X[:, i]
        col_name = f"V{i+1}"

        # 這些是固定的，可以預計算
        sorted_idx = np.argsort(v)
        v_sorted = v[sorted_idx]

        v_uni, inverse_idx = np.unique(v_sorted, return_inverse=True)
        end_points = np.bincount(inverse_idx, minlength=len(v_uni)).cumsum() - 1

        cut_info_cache[i] = {
            'name': col_name,
            'v': v,  # 原始變數（用於創建切點）
            'sorted_idx': sorted_idx,
            'v_sorted': v_sorted,
            'v_uni': v_uni,
            'end_points': end_points
        }

    return cut_info_cache


def _adjust_gam_dimension(gam, target_dim):
    """Adjust gamma dimension to match target"""
    gam = np.array(gam).flatten()

    if len(gam) == target_dim:
        return gam
    elif len(gam) < target_dim:
        return np.pad(gam, (0, target_dim - len(gam)))
    else:
        return gam[:target_dim]


def best_cut2_precomputed(cache_info, X_current, Y, gam):
    """使用預計算資訊快速計算最佳切點（依賴當前模型）

    關鍵：切點本身會根據當前模型動態選擇！

    Args:
        cache_info: 預計算的固定資訊
        X_current: current design matrix
        Y: response vector
        gam: current coefficients

    Returns:
        Dictionary with best cut matrix and value
    """
    # 從快取讀取固定資訊（避免排序）
    v = cache_info['v']
    sorted_idx = cache_info['sorted_idx']
    v_sorted = cache_info['v_sorted']
    v_uni = cache_info['v_uni']
    end_points = cache_info['end_points']
    name = cache_info['name']

    # 計算當前模型的殘差（動態）
    gam = _adjust_gam_dimension(gam, X_current.shape[1])
    residuals = -Y + 1 / (1 + np.exp(-X_current @ gam))
    residuals_sorted = residuals[sorted_idx]  # 使用預計算的 sorted_idx

    # 累積和（動態）
    cumsum_residuals = np.cumsum(residuals_sorted)
    cumsum_at_ends = cumsum_residuals[end_points]

    # 選擇最佳切點（動態 - 依賴當前殘差！）
    best_idx = np.argmax(np.abs(cumsum_at_ends))
    best_cutpoint = v_sorted[end_points[best_idx]]

    # 創建切點變數
    cut_variable = (v > best_cutpoint).astype(float).reshape(-1, 1)
    col_name = f"{name}_cut_{best_idx + 1}"

    # 計算重要性
    importance = np.abs(cut_variable.T @ residuals.reshape(-1, 1)).item()

    return {
        'm': pd.DataFrame(cut_variable, columns=[col_name]),
        'value': importance
    }


def best_cut2_set_precomputed(cut_info_cache, X_current, Y, gam,
                               import_threshold=0.8, max_set=5, regression_type='binary'):
    """使用預計算資訊快速計算多個變數的最佳切點

    **优化版本v2**:
    - v1: 残差只计算一次，避免重复的矩阵乘法 (2x加速)
    - v2: 批量构造DataFrame，避免多次concat (1.03x加速)

    适合与外层joblib并行配合使用（无冲突）

    Args:
        cut_info_cache: 預計算的固定資訊
        X_current: current design matrix
        Y: response vector
        gam: current coefficients
        import_threshold: threshold for variable importance
        max_set: maximum number of variables to select
        regression_type: 'binary' or 'multinomial'

    Returns:
        DataFrame with selected cut variables
    """
    n_vars = len(cut_info_cache)

    # === 优化: 只计算一次残差 ===
    if regression_type == 'multinomial':
        # Multinomial regression: compute residuals for each class
        # gam is (p, K-1) where K is number of classes
        # DON'T use _adjust_gam_dimension which flattens the (p, K-1) matrix!
        gam_adjusted = np.array(gam)

        if len(gam_adjusted.shape) == 1:
            # Binary case treated as multinomial
            residuals = -Y + 1 / (1 + np.exp(-X_current @ gam_adjusted))
        else:
            # True multinomial case: use actual class probability as residual
            # Compute linear predictors
            eta = X_current @ gam_adjusted  # (n, K-1)

            # Compute probabilities using softmax
            # Include class 0 (reference, eta_0 = 0)
            n = X_current.shape[0]
            K = gam_adjusted.shape[1] + 1
            eta_with_zero = np.concatenate([np.zeros((n, 1)), eta], axis=1)  # (n, K)
            max_eta = np.max(eta_with_zero, axis=1, keepdims=True)
            exp_eta_all = np.exp(eta_with_zero - max_eta)  # (n, K)
            sum_exp = np.sum(exp_eta_all, axis=1, keepdims=True)  # (n, 1)
            probs = exp_eta_all / sum_exp  # (n, K)

            # Compute residuals for each class separately
            # residual_k = I(y == k) - prob_k for each non-reference class
            # Keep as (n, K-1) matrix to avoid gradient cancellation across classes
            residuals = np.zeros((n, K-1))

            for k in range(1, K):  # Classes 1, 2, ... (skip reference class 0)
                y_k = (Y == k).astype(float)
                prob_k = probs[:, k]
                residuals[:, k-1] = y_k - prob_k  # Store in corresponding column

            # DEBUG
            # print(f"[DEBUG] Multinomial residuals: min={np.min(residuals):.3f}, max={np.max(residuals):.3f}, mean={np.mean(residuals):.3f}")
    else:
        # Binary logistic regression
        gam_adjusted = _adjust_gam_dimension(gam, X_current.shape[1])
        residuals = -Y + 1 / (1 + np.exp(-X_current @ gam_adjusted))

    # 計算每個變數的最佳切點（使用預計算資訊 + 共享残差）
    cut_results = []
    values = []

    for i in range(n_vars):
        cache_info = cut_info_cache[i]
        # 使用优化版本，传入已计算的残差
        result = best_cut2_precomputed_shared_residuals(cache_info, residuals)
        cut_results.append(result)
        values.append(result['value'])

    values = np.array(values)
    max_val = np.max(values)

    # DEBUG
    # print(f"[DEBUG] Cut values: min={np.min(values):.2f}, max={max_val:.2f}, mean={np.mean(values):.2f}")
    # print(f"[DEBUG] Threshold (import_threshold={import_threshold}): {import_threshold * max_val:.2f}")
    # print(f"[DEBUG] Values > threshold: {np.sum(values > import_threshold * max_val)}")

    # 選擇變數
    selected = values > (import_threshold * max_val)

    if np.sum(selected) > max_set:
        threshold_value = np.partition(values, -max_set)[-max_set]
        selected = values >= threshold_value

    selected_indices = np.where(selected)[0]

    if len(selected_indices) == 0:
        return pd.DataFrame()

    # === v2优化: 一次性构造DataFrame，避免多次concat ===
    if len(selected_indices) == 1:
        # 单个变量，直接返回
        return cut_results[selected_indices[0]]['m']
    else:
        # 多个变量，批量构造
        # 先提取所有NumPy数组和列名
        arrays = []
        names = []
        for idx in selected_indices:
            df = cut_results[idx]['m']
            arrays.append(df.values[:, 0])  # 提取第一列的值
            names.append(df.columns[0])      # 提取列名

        # 一次性构造DataFrame，避免多次concat
        combined = np.column_stack(arrays)
        return pd.DataFrame(combined, columns=names)


def best_cut2_precomputed_shared_residuals(cache_info, residuals):
    """使用预计算资讯和共享残差快速计算最佳切点

    这是优化版本，接收已经计算好的残差，避免重复计算

    Args:
        cache_info: 預計算的固定資訊
        residuals: 已计算的残差
            - Binary: (n,) 向量
            - Multinomial: (n, K-1) 矩陣

    Returns:
        Dictionary with best cut matrix and value
    """
    # 從快取讀取固定資訊（避免排序）
    v = cache_info['v']
    sorted_idx = cache_info['sorted_idx']
    v_sorted = cache_info['v_sorted']
    v_uni = cache_info['v_uni']
    end_points = cache_info['end_points']
    name = cache_info['name']

    # 判斷是否為 multinomial (殘差是矩陣)
    is_multinomial = len(residuals.shape) == 2

    if is_multinomial:
        # Multinomial: 對每個類別分別計算累積和，選擇總重要性最大的切點
        K_minus_1 = residuals.shape[1]

        # 計算每個類別的累積和的絕對值，然後加總
        cumsum_all_classes = None

        for k in range(K_minus_1):
            residuals_k = residuals[:, k]
            residuals_k_sorted = residuals_k[sorted_idx]

            cumsum_residuals_k = np.cumsum(residuals_k_sorted)
            cumsum_at_ends_k = cumsum_residuals_k[end_points]

            if cumsum_all_classes is None:
                # 初始化：使用第一個類別的絕對值
                cumsum_all_classes = np.abs(cumsum_at_ends_k)
            else:
                # 累加每個類別的絕對值
                cumsum_all_classes += np.abs(cumsum_at_ends_k)

        # 選擇總重要性最大的切點
        best_idx = np.argmax(cumsum_all_classes)
        best_cutpoint = v_sorted[end_points[best_idx]]

    else:
        # Binary: 原有邏輯
        residuals_sorted = residuals[sorted_idx]

        # 累積和（動態）
        cumsum_residuals = np.cumsum(residuals_sorted)
        cumsum_at_ends = cumsum_residuals[end_points]

        # 選擇最佳切點
        best_idx = np.argmax(np.abs(cumsum_at_ends))
        best_cutpoint = v_sorted[end_points[best_idx]]

    # 創建切點變數
    cut_variable = (v > best_cutpoint).astype(float).reshape(-1, 1)
    col_name = f"{name}_cut_{best_idx + 1}"

    # 計算重要性
    if is_multinomial:
        # Multinomial: 對每個類別分別計算內積，取絕對值後加總
        importance = 0
        for k in range(residuals.shape[1]):
            importance += np.abs(cut_variable.T @ residuals[:, k].reshape(-1, 1)).item()
    else:
        # Binary: 原有邏輯
        importance = np.abs(cut_variable.T @ residuals.reshape(-1, 1)).item()

    return {
        'm': pd.DataFrame(cut_variable, columns=[col_name]),
        'value': importance
    }


def _parse_cut_name(name):
    """Parse cut variable name to extract variable name and cut index

    Args:
        name: cut variable name (e.g., "V1_cut_3")

    Returns:
        Tuple of (variable_name, cut_index) or (None, None) if invalid
    """
    parts = name.split('_cut_')

    if len(parts) != 2:
        return None, None

    var_name = parts[0]
    try:
        cut_idx = int(parts[1])
        return var_name, cut_idx
    except ValueError:
        return None, None


def generate_test_cut(v, name, test):
    """Generate cut for test data based on training cut name

    Args:
        v: training variable vector
        name: cut variable name
        test: test data DataFrame

    Returns:
        DataFrame with cut variable for test data
    """
    var_name, cut_idx = _parse_cut_name(name)

    if var_name is None:
        return None

    # Get unique values and cutpoint
    v_uni = np.unique(v)

    if cut_idx > len(v_uni):
        return None

    cutpoint = v_uni[cut_idx - 1]

    # Extract test variable
    if var_name in test.columns:
        test_var = test[var_name].values
    else:
        # Try to extract by index (e.g., "V1" -> column 0)
        try:
            col_idx = int(var_name.replace('V', '')) - 1
            test_var = test.iloc[:, col_idx].values
        except (ValueError, IndexError):
            return None

    # Apply cutpoint
    cut_variable = (test_var > cutpoint).astype(float).reshape(-1, 1)
    return pd.DataFrame(cut_variable, columns=[name])


def generate_test_cut_all(X_cut, names, test):
    """Generate all cuts for test data

    Args:
        X_cut: training data DataFrame
        names: list of cut variable names
        test: test data DataFrame

    Returns:
        DataFrame with all cut variables for test data
    """
    results = []

    for name in names:
        var_name, _ = _parse_cut_name(name)

        if var_name is None:
            continue

        # Get training variable
        if var_name in X_cut.columns:
            train_var = X_cut[var_name].values
        else:
            # Try to extract by index
            try:
                col_idx = int(var_name.replace('V', '')) - 1
                train_var = X_cut.iloc[:, col_idx].values
            except (ValueError, IndexError):
                continue

        # Generate test cut
        result = generate_test_cut(train_var, name, test)
        if result is not None:
            results.append(result)

    if len(results) == 0:
        return pd.DataFrame()

    return pd.concat(results, axis=1)
