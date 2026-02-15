from __future__ import annotations

import numpy as np
import cvxpy as cp


def cestimate_opt(
    Yhat: np.ndarray,   # (n x Tm1)
    Xhat: np.ndarray,   # (m x Tm1)
    A: np.ndarray,      # (m x m)
    B: np.ndarray,      # (m x k)
    u: np.ndarray,      # (k x Tm1)
    m: int,
    n: int,
    k: int,
) -> np.ndarray:
    """
    Python translation of MATLAB:
      function Cest = Cestimate_Opt(Yhat,Xhat,A,B,u,m,n,k)

    Implements the stacked-matrix construction and solves:

      minimize norm(E)
      where E = (Yhat - C*Xhat - D*u)

    Notes:
    - MATLAB CVX `norm(E)` for a matrix is the spectral norm by default.
      We match that with cvxpy: cp.norm(E, 2).
    """
    # --- Shape checks ---
    if Yhat.shape[0] != n:
        raise ValueError(f"Yhat must be (n x T), got {Yhat.shape}")
    if Xhat.shape[0] != m:
        raise ValueError(f"Xhat must be (m x T), got {Xhat.shape}")
    if u.shape[0] != k:
        raise ValueError(f"u must be (k x T), got {u.shape}")
    if A.shape != (m, m):
        raise ValueError(f"A must be (m x m), got {A.shape}")
    if B.shape != (m, k):
        raise ValueError(f"B must be (m x k), got {B.shape}")

    Tm1 = Yhat.shape[1]
    if Xhat.shape[1] != Tm1 or u.shape[1] != Tm1:
        raise ValueError("Yhat, Xhat, u must share the same number of columns (time steps).")

    # --- Stacking matrices  ---
    # A_hat = [A; zeros((n-1)*m, m)]
    A_hat = np.vstack([A, np.zeros(((n - 1) * m, m))])  # (n*m x m)

    # B_hat = kron(eye(n), B)
    B_hat = np.kron(np.eye(n), B)  # (n*m x n*k)

    # K_hat construction:
    # K_hat = [zeros(m,(n-1)*m) zeros(m,m);
    #          kron(eye(n-1),A) zeros((n-1)*m,m)];
    top = np.hstack([np.zeros((m, (n - 1) * m)), np.zeros((m, m))])  # (m x n*m)
    bottom = np.hstack([np.kron(np.eye(n - 1), A), np.zeros(((n - 1) * m, m))])  # ((n-1)*m x n*m)
    K_hat = np.vstack([top, bottom])  # (n*m x n*m)

    k_hat = np.eye(n * m) - K_hat  # (n*m x n*m)

    # A_bar = k_hat \ A_hat
    A_bar = np.linalg.solve(k_hat, A_hat)  # (n*m x m)

    # B_bar = k_hat \ B_hat
    B_bar = np.linalg.solve(k_hat, B_hat)  # (n*m x n*k)

    # D_hat = A_bar' * B_bar
    D_hat = A_bar.T @ B_bar  # (m x n*k)

    # D = D_hat'; D = D([1:n],[1:k])
    D = D_hat.T  # (n*k x m)
    D = D[:n, :k]  # (n x k)

    # --- CVX optimization ---
    C = cp.Variable((n, m))

    E = Yhat - C @ Xhat - D @ u  # (n x Tm1)

    # CVX: minimize(norm(E))
    # Match default matrix norm => spectral norm (largest singular value)
    objective = cp.Minimize(cp.norm(E, 2))
    prob = cp.Problem(objective)

    # Use OSQP when possible, otherwise fallback to SCS
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if C.value is None:
        raise RuntimeError("CVX optimization failed: C has no value.")

    return np.array(C.value, dtype=float)
