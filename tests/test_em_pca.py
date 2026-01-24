import numpy as np
from core.models.em_pca import estimate_em_pca


def _orth(A: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(A)
    return Q


def _subspace_dist(Q1: np.ndarray, Q2: np.ndarray) -> float:
    # principal-angle based distance proxy
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.sqrt(1.0 - s.min() ** 2))


def test_em_pca_recovers_pca_subspace():
    rng = np.random.default_rng(42)
    n, T, m = 25, 400, 5

    C_true = rng.normal(size=(n, m))
    X_true = rng.normal(size=(m, T))
    Y = C_true @ X_true + 0.05 * rng.normal(size=(n, T))

    res = estimate_em_pca(Y, m, seed=0, iterations=500, tol=1e-8)

    # PCA via SVD on centered Y
    Yc = Y - Y.mean(axis=1, keepdims=True)
    U, _, _ = np.linalg.svd(Yc, full_matrices=False)
    C_pca = U[:, :m]

    d = _subspace_dist(_orth(res.C), _orth(C_pca))
    assert d < 0.1, f"Subspace distance too large: {d}"
    assert res.sse > 0
    assert res.n_iter >= 1
