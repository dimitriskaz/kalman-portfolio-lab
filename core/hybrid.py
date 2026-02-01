from __future__ import annotations

import numpy as np
import cvxpy as cp

from core.empca import ppca_em
from core.kalman_std import kalman_filter_standard, fit_transition_ols
from core.kf_real import build_A_B, build_state_X_from_u


def estimate_C_cvx_fro(Y: np.ndarray, F: np.ndarray, lam_ridge: float = 0.0) -> np.ndarray:
    """
    Solve: min ||Y - C F||_F + lam ||C||_F^2
    Y: (n x T), F: (d x T), C: (n x d)
    """
    n, T = Y.shape
    d, T2 = F.shape
    if T != T2:
        raise ValueError("Y and F must share same T.")

    C = cp.Variable((n, d))
    E = Y - C @ F
    obj = cp.norm(E, "fro")
    if lam_ridge > 0:
        obj = obj + lam_ridge * cp.sum_squares(C)

    prob = cp.Problem(cp.Minimize(obj))
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if C.value is None:
        raise RuntimeError("CVX failed to estimate C in hybrid.")
    return np.array(C.value, dtype=float)


def run_hybrid(
    Y: np.ndarray,      # (n x T)
    u: np.ndarray,      # (k x T)
    p: int,
    q: int,
    empca_max_iter: int = 200,
    empca_tol: float = 1e-6,
    lam_ridge: float = 0.0,
    seed: int | None = 0,
) -> dict[str, np.ndarray]:
    """
    Hybrid approach:
      - Observable lag-state X_obs from u (Approach 1 state structure)
      - Latent factors Z from EM–PCA on Y (Approach 2)
      - F = [X_obs; Z]
      - Estimate C via convex optimization
      - Fit A,Q from F and run Kalman
    """
    n, T = Y.shape
    k, T2 = u.shape
    if T != T2:
        raise ValueError("Y and u must share same T.")
    if q <= 0 or q >= n:
        raise ValueError("q must be in [1, n-1].")

    # Observable state from u
    X_obs = build_state_X_from_u(u=u, k=k, p=p)  # (m x T), m=k*(p+1)

    # Latent factors from EM–PCA
    pp = ppca_em(Y, q=q, max_iter=empca_max_iter, tol=empca_tol, seed=seed)
    Z = pp["Z"]  # (q x T)

    # Combined factor/state
    F = np.vstack([X_obs, Z])  # (d x T)

    # Estimate measurement matrix
    C_est = estimate_C_cvx_fro(Y, F, lam_ridge=lam_ridge)  # (n x d)

    # Residual-based measurement noise
    resid = Y - C_est @ F
    R = np.cov(resid, bias=True) + 1e-10 * np.eye(n)

    # Fit dynamics from combined factor series
    A, Q = fit_transition_ols(F)

    out = kalman_filter_standard(Y, A=A, C=C_est, Q=Q, R=R, x0=F[:, 0], P0=np.eye(F.shape[0]))

    return {
        "C_est": C_est,
        "F": F,
        "X_obs": X_obs,
        "Z": Z,
        "A": A,
        "Q": Q,
        "R": R,
        "Y_pred": out["y_pred"],
        "Y_true": Y,
    }
