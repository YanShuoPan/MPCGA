"""
MPCGA (Multi-Path Coordinate Gradient Ascent) Package
A tree-based feature selection algorithm for logistic regression
"""

# Import from utils
from .utils import llik, fd, MLE, gam_all, replace_fun

# Import from cut_generation_optimized
from .cut_generation_optimized import (
    precompute_cut_info,
    best_cut2_precomputed,
    best_cut2_set_precomputed,
    generate_test_cut,
    generate_test_cut_all
)

# Import from cga
from .cga import CGA, CGA_HDIC_Trim

# Import from mpcga
from .mpcga import CGA_tree2, MPCGA, HDIC_Trim, MPCGA_HDIC, Model_Trim

# Import from mpcga_while
from .mpcga_while import MPCGA_while, fit_model_while

# Import from model
from .model import fit_model, predict_fun, get_result

# Define what's available when using "from mpcga import *"
__all__ = [
    # Utils
    'llik', 'fd', 'MLE', 'gam_all', 'replace_fun',

    # Cut generation (optimized)
    'precompute_cut_info', 'best_cut2_precomputed', 'best_cut2_set_precomputed',
    'generate_test_cut', 'generate_test_cut_all',

    # CGA
    'CGA', 'CGA_HDIC_Trim',

    # MPCGA
    'CGA_tree2', 'MPCGA', 'HDIC_Trim', 'MPCGA_HDIC', 'Model_Trim',

    # MPCGA-While
    'MPCGA_while', 'fit_model_while',

    # Model
    'fit_model', 'predict_fun', 'get_result'
]

__version__ = '1.0.0'
__author__ = 'MPCGA Team'
