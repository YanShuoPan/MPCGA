# 關於可選模組 (distributions.py & mpcga_dist.py)

## 📌 重要說明

`distributions.py` 和 `mpcga_dist.py` 是**可選模組 (Optional Modules)**：

- ✅ **保留原因**: 展示擴展性設計，方便未來添加自訂分佈
- ❌ **目前未使用**: 所有模擬腳本直接使用 `mpcga_while.py`
- ✅ **測試通過**: 與 `mpcga_while.py` 產生完全相同的結果

---

## 🔍 這兩個模組的功能

### **distributions.py**
- 定義抽象基礎類別 `Distribution` (使用 Python ABC)
- 實現 `BinaryLogistic` 和 `MultinomialLogistic` 類別
- 提供統一的介面：`fit()`, `log_likelihood()`, `gradient()`

### **mpcga_dist.py**
- 是 `mpcga_while.py` 的**包裝器 (Wrapper)**
- 接受 `Distribution` 物件作為參數
- 內部直接呼叫 `mpcga_while.fit_model_while()`

---

## 💡 使用方式比較

### **方式 1: 直接使用 mpcga_while (推薦)**

```python
from mpcga_algorithm.mpcga_while import fit_model_while

# Binary classification
models = fit_model_while(
    X_train, y_train,
    K=25,
    regression_type='binary',
    c3=1.0,
    penalty_type='HDBIC'
)

# Multinomial classification
models = fit_model_while(
    X_train, y_train,
    K=25,
    regression_type='multinomial',
    c3=1.0,
    penalty_type='HDBIC'
)
```

**優點**:
- ✅ 直接、簡單
- ✅ 所有模擬腳本都用這個方式
- ✅ 效能最佳 (沒有額外包裝層)

---

### **方式 2: 使用 Distribution 物件 (可選)**

```python
from mpcga_algorithm.mpcga_dist import fit_model
from mpcga_algorithm.distributions import BinaryLogistic, MultinomialLogistic

# Binary classification
dist = BinaryLogistic()
models = fit_model(
    X_train, y_train,
    K=25,
    distribution=dist,
    c3=1.0,
    penalty_type='HDBIC'
)

# Multinomial classification
dist = MultinomialLogistic()
models = fit_model(
    X_train, y_train,
    K=25,
    distribution=dist,
    c3=1.0,
    penalty_type='HDBIC'
)
```

**優點**:
- ✅ 物件導向風格
- ✅ 統一的介面（如果需要自訂分佈）
- ✅ 與方式 1 產生**完全相同**的結果

**缺點**:
- ❌ 多了一層包裝，稍微慢一點點
- ❌ 目前沒有實際使用場景

---

## 🧪 測試結果

我們已經測試過兩種方式在 DGP1 (binary) 和 DGP4 (multinomial) 上的表現：

### **DGP1 (Binary)**
| 方法 | 預測 | 變數選擇 | 指標 |
|------|------|---------|------|
| mpcga_while | ✓ | ✓ | ✓ |
| mpcga_dist | ✓ | ✓ | ✓ |
| **結果** | **完全相同** | **完全相同** | **完全相同** |

### **DGP4 (Multinomial)**
| 方法 | 預測 | 變數選擇 | 指標 |
|------|------|---------|------|
| mpcga_while | ✓ | ✓ | ✓ |
| mpcga_dist | ✓ | ✓ | ✓ |
| **結果** | **完全相同** | **完全相同** | **完全相同** |

詳細測試腳本：`test_mpcga_dist_vs_while.py`

---

## 🎯 何時使用哪個？

### **使用 mpcga_while (推薦)**
- ✅ 標準的 binary/multinomial logistic regression
- ✅ 效能敏感的應用
- ✅ 遵循現有模擬腳本的做法

### **使用 mpcga_dist + distributions**
- ✅ 需要自訂分佈族 (例如 Poisson, Gaussian 等)
- ✅ 偏好物件導向介面
- ✅ 需要 API 一致性 (統一介面處理不同分佈)

### **自訂分佈範例**

如果未來需要支援其他分佈：

```python
from mpcga_algorithm.distributions import Distribution
import numpy as np

class PoissonRegression(Distribution):
    """自訂 Poisson 回歸"""

    def fit(self, X, Y):
        # 實現 Poisson 回歸的 fitting
        pass

    def log_likelihood(self, Y, X, coef):
        # 計算 Poisson log-likelihood
        eta = X @ coef
        return np.sum(Y * eta - np.exp(eta))

    def gradient(self, Y, X, coef):
        # 計算梯度
        eta = X @ coef
        return X.T @ (Y - np.exp(eta))

# 使用自訂分佈
dist = PoissonRegression()
models = fit_model(X, Y, K=25, distribution=dist)
```

---

## 📚 技術細節

### **為什麼 mpcga_dist 是包裝器？**

查看 `mpcga_dist.py` 的核心代碼：

```python
def fit_model(X, Y, K=25, distribution=None, **kwargs):
    # 1. 從 distribution 推斷 regression_type
    if distribution.get_n_classes() == 2:
        regression_type = 'binary'
    else:
        regression_type = 'multinomial'

    # 2. 直接呼叫 mpcga_while
    result = _fit_model_while_legacy(
        X, Y, K=K,
        regression_type=regression_type,
        **kwargs
    )

    return result
```

可以看到，它只是：
1. 從 `Distribution` 物件判斷是 binary 還是 multinomial
2. 呼叫 `mpcga_while.fit_model_while()`

**完全沒有改變演算法邏輯！**

---

## ✅ 總結

| 項目 | 說明 |
|------|------|
| **目前狀態** | 可選模組，未被使用 |
| **功能** | OOP 介面，內部呼叫 mpcga_while |
| **測試** | 與 mpcga_while 完全相同 |
| **保留原因** | 展示擴展性設計 |
| **建議使用** | 直接用 mpcga_while (推薦) |
| **適用場景** | 需要自訂分佈時 |

---

## 📖 相關文件

- `distributions.py` - 查看模組頂部的詳細說明
- `mpcga_dist.py` - 查看模組頂部的詳細說明
- `test_mpcga_dist_vs_while.py` - 完整的比較測試

如有疑問，請參閱各模組的 docstring 或測試腳本。
