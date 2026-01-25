# tests/test_shapes.py
import numpy as np
import pandas as pd

from core.factors import make_lagged_factor_matrix
from core.opt_cvx import estimate_C_via_cvx


def test_lag_matrix_shapes():
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    factors = pd.DataFrame({"f1": np.random.randn(50), "f2": np.random.randn(50)}, index=idx)
    X, aligned_idx, names = make_lagged_factor_matrix(factors, p=2)
    assert X.shape[0] == 2 * (2 + 1)  # m = k*(p+1)
    assert X.shape[1] == len(aligned_idx)
    assert len(names) == X.shape[0]


def test_cvx_C_shape():
    n, k, p = 3, 2, 1
    m = k * (p + 1)
    T = 60
    X = np.random.randn(m, T)
    C_true = np.random.randn(n, m)
    Y = C_true @ X + 0.01 * np.random.randn(n, T)

    C_hat = estimate_C_via_cvx(Y, X, lam_ridge=1e-3)
    assert C_hat.shape == (n, m)
