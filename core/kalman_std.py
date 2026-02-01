from __future__ import annotations

import numpy as np


def kalman_filter_standard(
    Y: np.ndarray,   # (n x T)
    A: np.ndarray,   # (d x d)
    C: np.ndarray,   # (n x d)
    Q: np.ndarray,   # (d x d)
    R: np.ndarray,   # (n x n)
    x0: np.ndarray | None = None,  # (d,)
    P0: np.ndarray | None = None,  # (d x d)
) -> dict[str, np.ndarray]:
    """
    Standard discrete-time Kalman filter:
      x_t = A x_{t-1} + w_t,   w_t ~ N(0,Q)
      y_t = C x_t     + v_t,   v_t ~ N(0,R)

    Returns:
      x_filt: (d x T) x_{t|t}
      x_pred: (d x T) x_{t|t-1}
      y_pred: (n x T) yhat_t = C x_{t|t-1}
      innov : (n x T) y_t - yhat_t
    """
    n, T = Y.shape
    d = A.shape[0]

    if A.shape != (d, d):
        raise ValueError("A must be square (d x d).")
    if C.shape != (n, d):
        raise ValueError("C must be (n x d).")
    if Q.shape != (d, d):
        raise ValueError("Q must be (d x d).")
    if R.shape != (n, n):
        raise ValueError("R must be (n x n).")

    x = np.zeros((d, 1)) if x0 is None else x0.reshape(d, 1)
    P = np.eye(d) if P0 is None else P0.copy()

    I = np.eye(d)

    x_filt = np.zeros((d, T))
    x_pred = np.zeros((d, T))
    y_pred = np.zeros((n, T))
    innov = np.zeros((n, T))

    for t in range(T):
        # predict
        xp = A @ x
        Pp = A @ P @ A.T + Q
        x_pred[:, t] = xp.ravel()

        yp = C @ xp
        y_pred[:, t] = yp.ravel()

        # update
        y = Y[:, t].reshape(n, 1)
        nu = y - yp
        innov[:, t] = nu.ravel()

        S = C @ Pp @ C.T + R
        S = 0.5 * (S + S.T)  # symmetrize
        K = Pp @ C.T @ np.linalg.inv(S)

        x = xp + K @ nu
        P = (I - K @ C) @ Pp
        P = 0.5 * (P + P.T)

        x_filt[:, t] = x.ravel()

    return {"x_filt": x_filt, "x_pred": x_pred, "y_pred": y_pred, "innov": innov}


def fit_transition_ols(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit A,Q from a state time series X (d x T):
      x_t ≈ A x_{t-1} + noise
    """
    d, T = X.shape
    if T < 2:
        raise ValueError("Need at least 2 time steps to fit transition.")

    X_prev = X[:, :-1]
    X_next = X[:, 1:]

    Xt = X_prev @ X_prev.T
    ridge = 1e-8 * np.eye(d)
    A = (X_next @ X_prev.T) @ np.linalg.inv(Xt + ridge)

    resid = X_next - A @ X_prev
    Q = np.cov(resid, bias=True) + 1e-10 * np.eye(d)
    return A, Q
