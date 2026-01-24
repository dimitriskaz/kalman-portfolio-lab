from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class EMPCAResult:
    C: np.ndarray      # (n, m) loading matrix
    X: np.ndarray      # (m, T) latent factors / projected data
    mu: np.ndarray     # (n, 1) mean vector used to center Y
    sse: float         # sum of squared errors (matches MATLAB listing)
    n_iter: int


def estimate_em_pca(
    Y: np.ndarray,
    m: int,
    *,
    iterations: int = 250,
    tol: float = 1e-6,
    seed: int | None = 0,
) -> EMPCAResult:
    """
    EM-PCA (E/M steps) ported from your MATLAB code (Listing 9.10 in the thesis).
    Y is expected to be shape (n, T) matching the thesis convention.
    Returns C, X, mean vector, SSE, and iterations used.

    Notes:
      - This algorithm finds the PCA principal subspace; C is identifiable up to rotation.
      - 'sse' here is sum of squared reconstruction errors, matching your MATLAB "MSE" line.
    """
    if Y.ndim != 2:
        raise ValueError("Y must be 2D (n, T).")
    n, T = Y.shape
    if m < 1 or m > min(n, T):
        raise ValueError(f"m must be between 1 and min(n,T)={min(n,T)}.")

    # mu = mean(Y,2)  => mean across time
    mu = np.mean(Y, axis=1, keepdims=True)
    Yc = Y - mu

    rng = np.random.default_rng(seed)
    C = rng.random((n, m), dtype=float)

    prev = float("inf")
    sse = prev

    for i in range(1, iterations + 1):
        # E-step: X = (C' C) \ (C' Y)
        CtC = C.T @ C
        CtY = C.T @ Yc
        X = np.linalg.solve(CtC, CtY)

        # M-step: C = (Y X') / (X X')
        XXt = X @ X.T
        YXt = Yc @ X.T
        C = (np.linalg.solve(XXt.T, YXt.T)).T

        # Reconstruction error
        E = Yc - (C @ X)
        sse = float(np.dot(E.ravel(), E.ravel()))

        # stopping condition (relative improvement)
        if np.isfinite(prev) and abs(prev - sse) < max(1.0, sse) * tol:
            return EMPCAResult(C=C, X=X, mu=mu, sse=sse, n_iter=i)
        prev = sse

    return EMPCAResult(C=C, X=X, mu=mu, sse=sse, n_iter=iterations)
