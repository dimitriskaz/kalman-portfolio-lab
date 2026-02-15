# core/kf_real.py
from __future__ import annotations

import numpy as np

from core.cestimate_opt import cestimate_opt


def build_A_B(k: int, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
      m = k*(p+1)
      A=[zeros(k,m-k),zeros(k,k); eye(k*p),zeros(m-k,k)];
      B=eye(m,k);
    """
    m = k * (p + 1)

    A_top = np.hstack([np.zeros((k, m - k)), np.zeros((k, k))])  # (k x m)
    if p > 0:
        A_bottom = np.hstack([np.eye(k * p), np.zeros((m - k, k))])  # ((m-k) x m)
        A = np.vstack([A_top, A_bottom])
    else:
        A = A_top

    B = np.eye(m, k)  # inject u into top k states
    return A, B


def build_state_X_from_u(u: np.ndarray, k: int, p: int) -> np.ndarray:
    """
    Construct X using the same shift-register model:
      X(:,1)=[u(:,1); zeros(k*p,1)];
      X(:,t)=A*X(:,t-1)+B*u(:,t-1);
    """
    k2, T = u.shape
    if k2 != k:
        raise ValueError("u must be (k x T)")

    m = k * (p + 1)
    A, B = build_A_B(k, p)

    X = np.zeros((m, T))
    X[:, 0] = np.concatenate([u[:, 0], np.zeros(k * p)]) if p > 0 else u[:, 0].copy()

    for t in range(1, T):
        X[:, t] = A @ X[:, t - 1] + B @ u[:, t - 1]

    return X


def kalman_with_C_est(
    Y: np.ndarray,  # (n x T)
    u: np.ndarray,  # (k x T)
    k: int,
    p: int,
    noise_scale: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Real-data version of KF_Opt focusing on C_est path (Approach 1).
    - No synthetic generation
    - Uses observed Y and observed factors u
    """
    n, T = Y.shape
    k2, T2 = u.shape
    if T != T2 or k2 != k:
        raise ValueError("Shape mismatch: Y is (n x T), u must be (k x T) with same T")

    m = k * (p + 1)
    A, B = build_A_B(k, p)

    # Build X from u
    X = build_state_X_from_u(u, k, p)  # (m x T)

    # Build E (measurement noise covariance)
    # MATLAB makes E from random noise; for real data we estimate scale from Y variance
    y_var = np.var(Y, axis=1) + 1e-12
    E = np.diag(noise_scale * y_var)  # (n x n), SPD-ish

    # u_hat / Yhat / Xhat use T-1 columns
    Yhat = Y[:, :-1]
    Xhat = X[:, :-1]
    uhat = u[:, :-1]

    # Estimate C (Approach 1 exact translation)
    C_est = cestimate_opt(Yhat, Xhat, A, B, uhat, m=m, n=n, k=k)

    # Kalman recursion (estimated C only)
    Q = np.eye(k)          
    R = np.eye(m)         

    Sigma = np.zeros((m, m, T))
    Sigma[:, :, 0] = R

    X_opt = np.zeros((m, T))
    X_opt_t = np.zeros((m, T - 1))

    Y_pred = np.zeros((n, T - 1))

    for t in range(1, T):
        S = C_est @ Sigma[:, :, t - 1] @ C_est.T + E
        L = Sigma[:, :, t - 1] @ C_est.T @ np.linalg.inv(S)

        Sigma[:, :, t] = (
            A @ Sigma[:, :, t - 1] @ A.T
            + B @ Q @ B.T
            - A @ L @ C_est @ Sigma[:, :, t - 1] @ A.T
        )

        innov = Y[:, t - 1] - (C_est @ X_opt[:, t - 1])
        X_opt[:, t] = A @ (L @ innov) + A @ X_opt[:, t - 1]
        X_opt_t[:, t - 1] = X_opt[:, t - 1] + L @ innov

        # predicted measurement for time t-1 (one-step filter-style)
        Y_pred[:, t - 1] = C_est @ X_opt_t[:, t - 1]

    return {
        "C_est": C_est,
        "X": X,
        "Y_pred": Y_pred,      # (n x T-1) predicted returns aligned to Y[:, :-1]
        "Y_true": Y[:, :-1],   # matching segment
    }
