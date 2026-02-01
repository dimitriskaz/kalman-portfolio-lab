# core/cestimate_em_pca.py
from __future__ import annotations

import numpy as np


def cestimate_em_pca(
    Y: np.ndarray,            # (n x T)
    m: int,                   # target latent dimension
    iterations: int = 250,
    tol: float = 1e-6,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Exact translation of MATLAB Listing 9.10: Cestimate_EM_PCA(Y, m)

    Inputs:
      Y: n x T data matrix
      m: number of dynamic factors (target dimension)
    Outputs:
      C: n x m factor loading matrix
      X: m x T projected data matrix (latent factors)
      mu: n x 1 mean vector
      MSE: mean square error (scalar)
    """
    if seed is not None:
        np.random.seed(seed)

    n, T = Y.shape
    if m <= 0 or m > n:
        raise ValueError("m must satisfy 1 <= m <= n")

    # mu = mean(Y,2)
    mu = np.mean(Y, axis=1, keepdims=True)

    # Y = bsxfun(@minus,Y,mu)
    Yc = Y - mu

    # C = rand(n,m)
    C = np.random.rand(n, m)

    MSE = float("inf")

    for _ in range(iterations):
        last = MSE

        # E-step: X = (C'*C)\(C'*Y)
        CtC = C.T @ C
        ridge = 1e-10 * np.eye(m)
        X = np.linalg.solve(CtC + ridge, C.T @ Yc)  # (m x T)

        # M-step: C = (Y*X')/(X*X')
        XXt = X @ X.T
        C = (Yc @ X.T) @ np.linalg.inv(XXt + ridge)  # (n x m)

        # E = Y - C*X
        E = Yc - C @ X

        # MSE = mean(dot(E(:),E(:)));
        MSE = float(np.mean(E.ravel() @ E.ravel()))

        if abs(last - MSE) < MSE * tol:
            break

    return C, X, mu, MSE
