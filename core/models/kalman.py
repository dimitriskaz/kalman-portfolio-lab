from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from core.models.em_pca import estimate_em_pca


@dataclass
class KFRealDataResult:
    y_actual: float          # scalar norm of final actual return vector
    y_estimate: float        # scalar norm of final estimated return vector
    e_c_est: float           # normalized error
    y_estimated_vec: np.ndarray  # (n,) estimated return vector at final step
    c_est: np.ndarray        # (n, m) estimated factor loading matrix
    x_opt_t: np.ndarray      # (m, T-1) posterior state estimates (per MATLAB indexing)
    sigma: np.ndarray        # (m, m) final covariance matrix


def _build_state_matrices(k: int, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Replicates the MATLAB construction:

      m = k*(p+1)
      A = [zeros(k,m-k), zeros(k,k);
           eye(k*p),     zeros(m-k,k)]
      B = eye(m,k)
      Q = eye(k)
      R = eye(m)

    as shown in KF_RealData / KF_EM. :contentReference[oaicite:5]{index=5}
    """
    m = k * (p + 1)

    # A is (m x m)
    # top block: k rows, m columns = [0_(k x (m-k)) , 0_(k x k)]
    top = np.hstack([np.zeros((k, m - k)), np.zeros((k, k))])

    # bottom block: (m-k) rows
    # [I_(k*p), 0_((m-k) x k)]
    if p > 0:
        bottom = np.hstack([np.eye(k * p), np.zeros((m - k, k))])
    else:
        # if p=0 then m=k and bottom has 0 rows
        bottom = np.zeros((0, m))

    A = np.vstack([top, bottom])  # (m,m)

    # B is (m x k) identity-in-top block in MATLAB: eye(m,k)
    B = np.eye(m, k)

    Q = np.eye(k)
    R = np.eye(m)
    return A, B, Q, R, m


def _spd_noise_cov(n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    """
    MATLAB:
      e = noise*randn(n);
      E = e*e';
      E = E + E;

    Note: e is (n x n) in MATLAB when randn(n) called with single arg,
    so E becomes (n x n). :contentReference[oaicite:6]{index=6}
    """
    e = noise * rng.standard_normal((n, n))
    E = e @ e.T
    E = E + E
    return E


def kf_real_data_em(
    Y: np.ndarray,
    k: int,
    p: int,
    *,
    noise: float = 0.3,
    seed: int = 0,
    em_iterations: int = 250,
    em_tol: float = 1e-6,
) -> KFRealDataResult:
    """
    Python port of MATLAB KF_RealData that:
      - estimates C_est with EM-PCA
      - runs the Kalman recursion using C_est
      - returns scalar norms + normalized error

    Y: (n, T) asset return matrix (thesis convention).
    k: number of (observable) factors
    p: lag order
    """
    if Y.ndim != 2:
        raise ValueError("Y must be 2D with shape (n, T).")
    n, T = Y.shape
    if T < 2:
        raise ValueError("Need at least 2 time steps (T>=2).")
    if k < 1 or p < 0:
        raise ValueError("k must be >=1 and p must be >=0.")

    rng = np.random.default_rng(seed)

    # Build A, B, Q, R and m = k*(p+1)
    A, B, Q, R, m = _build_state_matrices(k, p)
    
    # Validate m doesn't exceed what EM-PCA can handle
    max_m = min(n, T)
    if m > max_m:
        raise ValueError(
            f"Invalid k,p: m=k*(p+1)={m} but must be <= min(n,T)={max_m}. "
            f"Fix by reducing k or p, increasing number of assets (n), or using more time points (T)."
        )

    # Measurement noise covariance E (n x n), from MATLAB code :contentReference[oaicite:7]{index=7}
    E = _spd_noise_cov(n=n, noise=noise, rng=rng)

    # Estimate factor loading matrix using EM-PCA (Approach 2) :contentReference[oaicite:8]{index=8}
    # In MATLAB they call: [C_est, X_est, mu, mse] = Cestimate_EM_PCA(Y, m);
    em_res = estimate_em_pca(Y, m, iterations=em_iterations, tol=em_tol, seed=seed)
    C_est = em_res.C  # (n, m)

    # Initialisation consistent with MATLAB:
    # X_opt_C(:,1)=0*X(:,1)  (we don't need X itself for this recursion)
    # X_opt_t_C(:,1)=0*X(:,1)
    # Sigma_C(:,:,1)=R;
    # Loop for t=2:T
    X_opt = np.zeros((m, T), dtype=float)      # store X_opt_C(:,t)
    X_opt_t = np.zeros((m, T), dtype=float)    # store X_opt_t_C(:,t-1) pattern
    Sigma = np.zeros((m, m, T), dtype=float)
    Sigma[:, :, 0] = R

    y_estimated_vec = np.zeros((n,), dtype=float)

    # Helper: solve (C_est * Sigma * C_est^T + E) without explicit inverse
    for t in range(1, T):  # Python t=1..T-1 corresponds to MATLAB t=2..T
        Sigma_prev = Sigma[:, :, t - 1]

        S = C_est @ Sigma_prev @ C_est.T + E  # (n,n)
        # L = Sigma_prev * C_est' * inv(S)
        # compute inv(S) via solve: inv(S) = solve(S, I)
        Sinv = np.linalg.solve(S, np.eye(n))
        L = Sigma_prev @ C_est.T @ Sinv  # (m,n)

        # Sigma update:
        # Sigma = A*Sigma_prev*A' + B*Q*B' - A*L*C_est*Sigma_prev*A'
        Sigma[:, :, t] = (
            A @ Sigma_prev @ A.T
            + B @ Q @ B.T
            - A @ L @ C_est @ Sigma_prev @ A.T
        )

        # X_opt update:
        # X_opt(:,t)=A*L*(Y(:,t-1)-C_est*X_opt(:,t-1))+A*X_opt(:,t-1);
        innovation = Y[:, t - 1] - (C_est @ X_opt[:, t - 1])
        X_opt[:, t] = A @ (L @ innovation) + (A @ X_opt[:, t - 1])

        # X_opt_t(:,t-1)=X_opt(:,t-1)+L*(Y(:,t-1)-C_est*X_opt(:,t-1));
        X_opt_t[:, t - 1] = X_opt[:, t - 1] + (L @ innovation)

        # If final step, estimate Y using C_est and posterior state
        if t == T - 1:
            # MATLAB: Y_estimated = C_est * X_opt_t_C(:,t-1);
            y_estimated_vec = C_est @ X_opt_t[:, t - 1]

    # MATLAB output scalars (norms) :contentReference[oaicite:9]{index=9}
    y_actual = float(np.linalg.norm(Y[:, -1]))
    y_estimate = float(np.linalg.norm(y_estimated_vec))
    denom = max(1e-12, y_actual)
    e_c_est = float(np.linalg.norm(y_actual - y_estimate) / denom)

    return KFRealDataResult(
        y_actual=y_actual,
        y_estimate=y_estimate,
        e_c_est=e_c_est,
        y_estimated_vec=y_estimated_vec,
        c_est=C_est,
        x_opt_t=X_opt_t[:, : (T - 1)],
        sigma=Sigma[:, :, -1],
    )
