# core/approach2_real.py
from __future__ import annotations

import numpy as np

from core.cestimate_em_pca import cestimate_em_pca


def build_A_B(k: int, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Same as MATLAB:
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

    B = np.eye(m, k)
    return A, B


def run_approach2_em_pca_on_yahoo(
    Y: np.ndarray,      # (n x T) real returns from Yahoo
    k: int,             # number of "base factors" (paper uses this)
    p: int,             # lag order
    noise_scale: float = 0.5,
    em_iterations: int = 250,
    em_tol: float = 1e-6,
    seed: int | None = 0,
) -> dict[str, np.ndarray]:
    """
    Paper-faithful Approach 2 for REAL data.

    Steps (matching KF_EM structure):
      1) Define m = k*(p+1), build A,B,Q,R
      2) Estimate C_est using Cestimate_EM_PCA(Y, m)
      3) Run Kalman recursion using C_est (same structure as MATLAB)
      4) Return predicted returns series and portfolio series for plotting

    Notes on noise covariance E:
      - MATLAB builds E from random noise e*e' and symmetrizes.
      - For real data we build a stable SPD E using per-asset variance scaled by noise_scale.
    """
    if seed is not None:
        np.random.seed(seed)

    n, T = Y.shape
    if T < 10:
        raise ValueError("Need more observations (T) to run the filter.")
    if k <= 0:
        raise ValueError("k must be >= 1")
    if p < 0:
        raise ValueError("p must be >= 0")

    m = k * (p + 1)
    if m > n:
        # The EM-PCA routine in the paper sets m as "dimension of target space".
        # It is valid to have m <= n. If m > n, the inverses become unstable.
        raise ValueError(f"Invalid setting: m=k*(p+1)={m} exceeds n={n}. Reduce k or p.")

    A, B = build_A_B(k, p)

    # Q=eye(k), R=eye(m) per MATLAB
    Q = np.eye(k)
    R = np.eye(m)

    # Build E (n x n) measurement noise covariance (real-data replacement)
    y_var = np.var(Y, axis=1) + 1e-12
    E = np.diag(noise_scale * y_var)
    E = 0.5 * (E + E.T)

    # EM–PCA estimate of C and latent X from Y
    C_est, X_est, mu, mse = cestimate_em_pca(
        Y=Y,
        m=m,
        iterations=em_iterations,
        tol=em_tol,
        seed=seed,
    )

    # IMPORTANT:
    # C_est was estimated on centered data (Y - mu), so we also filter in centered space.
    Yc = Y - mu

    # Kalman recursion (estimated C) — translate MATLAB KF_EM loop
    Sigma = np.zeros((m, m, T))
    Sigma[:, :, 0] = R

    X_opt = np.zeros((m, T))
    X_opt_t = np.zeros((m, T))

    # Store predicted returns (centered), then add mu back
    Yc_pred = np.zeros((n, T))

    for t in range(1, T):
        # L = Sigma(:,:,t-1)*C_est'*inv(C_est*Sigma(:,:,t-1)*C_est'+E)
        S = C_est @ Sigma[:, :, t - 1] @ C_est.T + E
        L = Sigma[:, :, t - 1] @ C_est.T @ np.linalg.inv(S)

        # Sigma(:,:,t)=A*Sigma(:,:,t-1)*A'+B*Q*B'-A*L*C_est*Sigma(:,:,t-1)*A'
        Sigma[:, :, t] = (
            A @ Sigma[:, :, t - 1] @ A.T
            + B @ Q @ B.T
            - A @ L @ C_est @ Sigma[:, :, t - 1] @ A.T
        )

        # X_opt(:,t)=A*L*(Y(:,t-1)-C_est*X_opt(:,t-1))+A*X_opt(:,t-1);
        innov = Yc[:, t - 1] - (C_est @ X_opt[:, t - 1])
        X_opt[:, t] = A @ (L @ innov) + A @ X_opt[:, t - 1]

        # X_opt_t(:,t-1)=X_opt(:,t-1)+L*(Y(:,t-1)-C_est*X_opt(:,t-1));
        X_opt_t[:, t - 1] = X_opt[:, t - 1] + L @ innov

        # Predicted centered return at (t-1) based on updated state estimate
        Yc_pred[:, t - 1] = C_est @ X_opt_t[:, t - 1]

    # Add mean back for return-level predictions
    Y_pred = Yc_pred + mu

    return {
        "C_est": C_est,
        "X_est": X_est,
        "mu": mu,
        "mse": np.array([mse]),
        "Y_true": Y,
        "Y_pred": Y_pred,
        "k": np.array([k]),
        "p": np.array([p]),
        "m": np.array([m]),
    }
