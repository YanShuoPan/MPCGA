# MPCGA_clean 資料夾結構說明

## 📁 完整檔案樹狀圖

```
MPCGA_clean/
├── README.md                        # 項目說明文檔
├── requirements.txt                 # Python 套件依賴
├── .gitignore                       # Git 忽略檔案配置
├── STRUCTURE.md                     # 本檔案 (結構說明)
│
├── mpcga_algorithm/                 # 核心演算法 (8 個檔案)
│   ├── __init__.py                  # 套件初始化
│   ├── mpcga_while.py               # MPCGA 主演算法 (優化版)
│   ├── mpcga.py                     # HDIC_Trim, Model_Trim 等輔助函數
│   ├── cga.py                       # 傳統 CGA baseline
│   ├── model.py                     # 預測函數 (get_result)
│   ├── cut_generation_optimized.py  # 切點生成 (預計算排序版)
│   ├── distributions.py             # 分佈支援 (multinomial)
│   └── utils.py                     # 工具函數
│
├── simulations/                     # 模擬研究 (4 個檔案)
│   ├── sim_dgp123_mpcga.py          # DGP1-3 MPCGA 方法
│   ├── sim_dgp123_baseline.py       # DGP1-3 baseline 方法
│   ├── sim_dgp45_mpcga.py           # DGP4-5 MPCGA 方法 (multinomial)
│   └── sim_dgp45_baseline.py        # DGP4-5 baseline 方法
│
├── data_generation.py               # 數據生成 (DGP1-5)
└── evaluation_metrics.py            # 評估指標計算
```

**總計**: 16 個 Python 檔案 + 3 個配置/說明檔案

---

## 📋 檔案功能詳細說明

### 1️⃣ 核心演算法 (`mpcga_algorithm/`)

#### `mpcga_while.py` ⭐ 主演算法
- **主要函數**:
  - `MPCGA_while()` - 多路徑探索演算法
  - `fit_model_while()` - 完整的 MPCGA 流程
- **特色**: 使用 while 迴圈 + deque，預計算排序優化
- **用途**: 產生多條候選路徑

#### `mpcga.py` ⭐ 輔助函數
- **主要函數**:
  - `HDIC_Trim()` - 套用 HDAIC/HDBIC penalty
  - `Model_Trim()` - MTrim 路徑修剪
  - `MPCGA()` - 舊版 MPCGA (保留向後相容)
  - `CGA_tree2()` - 遞迴樹狀探索 (舊版)
- **用途**: Penalty 選擇和模型修剪

#### `cga.py` - Baseline
- **主要函數**:
  - `fit_model_cga()` - 傳統單路徑 CGA
  - `predict_cga()` - CGA 預測
- **用途**: 提供傳統 CGA baseline 比較

#### `model.py` - 預測
- **主要函數**:
  - `get_result()` - 多模型集成預測 (投票)
- **用途**: 所有方法的預測函數

#### `cut_generation_optimized.py` - 切點生成
- **主要函數**:
  - `precompute_cut_info()` - 預計算排序資訊
  - `best_cut2_set_precomputed()` - 快速切點選擇
- **用途**: 自動生成最佳切點

#### `distributions.py` - 分佈支援
- **主要類別**:
  - `Distribution` - 分佈基礎類別
- **用途**: 支援 multinomial regression (DGP4-5)

#### `utils.py` - 工具函數
- **主要函數**:
  - `fd()` - First difference (binary)
  - `fd_multinomial_aggregated()` - First difference (multinomial)
- **用途**: 數學計算輔助

---

### 2️⃣ 模擬研究 (`simulations/`)

#### `sim_dgp123_mpcga.py` - DGP1-3 MPCGA
- **方法數量**: 10 個
- **配置**: 6 種 (DGP1-3 × n=300/600)
- **迭代次數**: 100
- **包含方法**:
  1. CGA+HDBIC
  2. MPCGA+HDBIC(c3=0.8)
  3. MPCGA+HDBIC(c3=1.0)
  4. MPCGA+HDAIC(OP)
  5. MPCGA+HDBIC+MTrim(c3=0.8)
  6. MPCGA+HDBIC+MTrim(c3=1.0)
  7-10. MPCGA+RF/XGB (c3=0.8/1.0)

#### `sim_dgp123_baseline.py` - DGP1-3 Baseline
- **方法數量**: 6 個
- **配置**: 6 種
- **包含方法**:
  1. Lasso
  2. Adaptive Lasso
  3. Random Forest
  4. XGBoost
  5. RF + Boruta
  6. XGB + Boruta

#### `sim_dgp45_mpcga.py` - DGP4-5 MPCGA
- **特色**: Multinomial classification (3 classes)
- **方法**: 類似 DGP1-3

#### `sim_dgp45_baseline.py` - DGP4-5 Baseline
- **特色**: Multinomial baseline methods

---

### 3️⃣ 數據與評估

#### `data_generation.py` - 數據生成
- **函數**:
  - `generate_data_dgp1()` - 線性效應
  - `generate_data_dgp2()` - Cut + 線性
  - `generate_data_dgp3()` - 二次 + Cut
  - `generate_data_dgp4()` - Multinomial, Cut + 線性
  - `generate_data_dgp5()` - Multinomial, 二次 + Cut

#### `evaluation_metrics.py` - 評估指標
- **函數**:
  - `compute_metrics()` - 計算單次指標
  - `summarize_metrics()` - 匯總多次結果
  - `print_metrics_summary()` - 顯示結果
- **指標**:
  - Accuracy, Exact rate, Variable length
  - E+I (excess + incorrect), Include rate

---

## 🎯 與原始資料夾的差異

### ✅ 保留的檔案 (16 個核心檔案)
- 所有必要的演算法和模擬檔案
- 功能完整，可以直接運行

### ❌ 移除的檔案類型
- 備份檔案 (`*_backup.py`, `*_old.py`)
- 舊版本資料夾 (`old_versions/`)
- 實驗性檔案 (`cut_generation.py`, `mpcga_dist.py`)
- 測試腳本 (`test_*.py`, `compare_*.py`, `verify_*.py`)
- 臨時分析腳本 (`summarize_*.py`, `show_*.py`)

### ⭐ 新增的檔案 (3 個)
- `README.md` - 完整的使用說明
- `requirements.txt` - 套件依賴清單
- `.gitignore` - Git 配置
- `STRUCTURE.md` - 本檔案

---

## 🚀 使用方式

### 1. 安裝依賴
```bash
cd MPCGA_clean
pip install -r requirements.txt
```

### 2. 運行模擬
```bash
# DGP1-3 MPCGA 方法
python simulations/sim_dgp123_mpcga.py

# DGP1-3 Baseline 方法
python simulations/sim_dgp123_baseline.py
```

### 3. 使用 MPCGA 演算法
```python
from mpcga_algorithm.mpcga_while import fit_model_while
# ... (詳見 README.md)
```

---

## 📊 檔案大小統計

- **核心演算法**: 8 個檔案
- **模擬腳本**: 4 個檔案
- **數據/評估**: 2 個檔案
- **文檔**: 4 個檔案 (README, STRUCTURE, requirements, .gitignore)
- **總計**: 18 個檔案

**對比原始資料夾**: 從 40+ 檔案精簡到 18 個核心檔案 (減少約 55%)

---

## ✅ 清理完成檢查清單

- [x] 移除所有備份檔案
- [x] 移除舊版本資料夾
- [x] 移除實驗性/測試檔案
- [x] 保留所有核心功能
- [x] 添加完整文檔 (README)
- [x] 添加依賴清單 (requirements.txt)
- [x] 添加 Git 配置 (.gitignore)
- [x] 結構清晰，易於理解

**此資料夾可直接上傳至 GitHub！** ✅
