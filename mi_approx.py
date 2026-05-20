import numpy as np
from math import comb
from scipy.special import gammaln, digamma, polygamma, logsumexp
import pandas as pd

EPS = 1e-300

def empirical_joint(samples_x: np.ndarray, samples_y: np.ndarray, m: int) -> np.ndarray:
    counts = np.zeros((m, m), dtype=float)
    np.add.at(counts, (samples_x, samples_y), 1.0)
    return counts / counts.sum()


def sample_from_joint(Pxy, n, rng: np.random.Generator):
    Pxy = normalize(Pxy)
    m, l = Pxy.shape
    flat = Pxy.ravel()
    idx = rng.choice(len(flat), size=n, replace=True, p=flat)
    x = idx // l
    y = idx % l
    return x, y

def normalize(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    s = P.sum()
    if s <= 0:
        raise ValueError("Distribution must have positive total mass.")
    return P / s


def entropy(P: np.ndarray, base = np.e):
    """Shannon entropy of a discrete distribution."""
    p = np.asarray(P, dtype=float).ravel()
    p = p[p > 0]
    if base == np.e:
        return float(-np.sum(p * np.log(p)))
    return float(-np.sum(p * np.log(p)) / np.log(base))


def mutual_information(Pxy: np.ndarray, base = np.e):
    """Exact mutual information for a joint distribution Pxy."""
    Pxy = normalize(Pxy)
    px = Pxy.sum(axis=1)
    py = Pxy.sum(axis=0)
    return entropy(px, base=base) + entropy(py, base=base) - entropy(Pxy, base=base)


def plugin_mi_from_samples(samples_x, samples_y, m, base = np.e):
    Phat = empirical_joint(samples_x, samples_y, m)
    return mutual_information(Phat, base=base)


def miller_madow_mi_from_samples(samples_x, samples_y, m, base = np.e):
    """
    Miller-Madow correction for MI:

        I_MM = I_plugin + [(K_xy - K_x - K_y + 1) / (2n)]

    in nats. Divide by log(base) if using another base.

    Here K_x, K_y, K_xy are observed support sizes.
    """
    n = len(samples_x)
    counts = np.zeros((m, m), dtype=int)
    np.add.at(counts, (samples_x, samples_y), 1)

    kxy = np.sum(counts > 0)
    kx = np.sum(counts.sum(axis=1) > 0)
    ky = np.sum(counts.sum(axis=0) > 0)

    correction = (kxy - kx - ky + 1) / (2.0 * n)
    if base != np.e:
        correction /= np.log(base)

    return plugin_mi_from_samples(samples_x, samples_y, m, base=base) + correction

def expected_entropy_dirichlet_posterior(counts, beta):
    """
    Posterior expected entropy under a symmetric Dirichlet(beta) prior.

    H is returned in nats.

    For posterior parameters alpha_i = n_i + beta,

        E[H(p) | counts, beta]
        =
        psi(A + 1)
        -
        sum_i alpha_i / A * psi(alpha_i + 1)

    where A = sum_i alpha_i.
    """
    counts = np.asarray(counts, dtype=float)
    K = len(counts)
    alpha = counts + beta
    A = np.sum(alpha)

    return float(
        digamma(A + 1.0)
        - np.sum((alpha / A) * digamma(alpha + 1.0))
    )


def expected_entropy_symmetric_dirichlet_prior(K, beta):
    """
    Expected entropy under a symmetric Dirichlet(beta) prior over K bins.

    This is the prior mean entropy xi(beta).
    """
    return float(digamma(K * beta + 1.0) - digamma(beta + 1.0))


def d_expected_entropy_prior_d_beta(K, beta):
    """
    Derivative of xi(beta) = E[H(p)] under symmetric Dirichlet(beta).

    xi'(beta) = K * psi_1(K beta + 1) - psi_1(beta + 1)

    The NSB prior is proportional to |xi'(beta)|.
    """
    return float(K * polygamma(1, K * beta + 1.0) - polygamma(1, beta + 1.0))

def log_dirichlet_multinomial_evidence_symmetric(
    counts: np.ndarray,
    beta: float,
) -> float:
    """
    Log marginal likelihood of counts under symmetric Dirichlet(beta),
    up to the multinomial coefficient.

    The multinomial coefficient does not depend on beta and cancels in
    posterior weighting over beta.

    log p(counts | beta)
    =
    log Gamma(K beta) - log Gamma(N + K beta)
    + sum_i [log Gamma(n_i + beta) - log Gamma(beta)].
    """
    counts = np.asarray(counts, dtype=float)
    K = len(counts)
    N = np.sum(counts)

    return float(
        gammaln(K * beta)
        - gammaln(N + K * beta)
        + np.sum(gammaln(counts + beta) - gammaln(beta))
    )


def nsb_entropy_from_counts(counts, base = np.e, beta_grid=None, return_diagnostics = False,):
    """
    NSB entropy estimate for a discrete distribution with known support.

    Parameters
    ----------
    counts:
        Counts over the full known alphabet. Include zero-count bins.
    base:
        np.e for nats, 2 for bits.
    beta_grid:
        Optional grid over symmetric Dirichlet concentration beta.
        If None, a default log-spaced grid is used.
    return_diagnostics:
        If True, return estimate and integration diagnostics.

    Returns
    -------
    H_nsb:
        NSB entropy estimate.
    """
    counts = np.asarray(counts, dtype=float)
    K = len(counts)
    N = np.sum(counts)

    if K <= 1:
        return 0.0 if not return_diagnostics else (0.0, {})

    if N <= 0:
        raise ValueError("Counts must sum to a positive value.")

    if beta_grid is None:
        # Wide grid. You can tune this if needed.
        # Small beta handles sparse distributions; large beta approaches uniform.
        beta_grid = np.logspace(-8, 4, 1200)

    log_beta = np.log(beta_grid)

    log_weights = []
    H_beta = []

    for beta in beta_grid:
        xi_prime = abs(d_expected_entropy_prior_d_beta(K, beta))

        # If derivative is numerically zero, skip this point.
        if not np.isfinite(xi_prime) or xi_prime <= 0:
            log_weights.append(-np.inf)
            H_beta.append(np.nan)
            continue

        log_evidence = log_dirichlet_multinomial_evidence_symmetric(counts, beta)

        # Integrating over beta on a log-spaced grid means d beta = beta d log beta.
        # Therefore include + log(beta) in the quadrature weight.
        lw = log_evidence + np.log(xi_prime) + np.log(beta)

        log_weights.append(lw)
        H_beta.append(expected_entropy_dirichlet_posterior(counts, beta))

    log_weights = np.asarray(log_weights)
    H_beta = np.asarray(H_beta)

    valid = np.isfinite(log_weights) & np.isfinite(H_beta)
    if valid.sum() < 5:
        # Fallback to plugin if integration fails.
        p = counts / counts.sum()
        p = p[p > 0]
        H = -np.sum(p * np.log(p))
        if base != np.e:
            H /= np.log(base)
        return H if not return_diagnostics else (H, {"fallback": True})

    log_weights = log_weights[valid]
    H_beta = H_beta[valid]
    beta_grid_valid = beta_grid[valid]

    weights = np.exp(log_weights - logsumexp(log_weights))
    H_nsb = float(np.sum(weights * H_beta))

    if base != np.e:
        H_nsb /= np.log(base)

    if return_diagnostics:
        diagnostics = {
            "fallback": False,
            "beta_grid": beta_grid_valid,
            "weights": weights,
            "posterior_mean_beta": float(np.sum(weights * beta_grid_valid)),
            "posterior_entropy_sd": float(np.sqrt(np.sum(weights * (H_beta - np.sum(weights * H_beta)) ** 2))),
        }
        return H_nsb, diagnostics

    return H_nsb

def nsb_mi_from_counts_joint(counts_xy, base = np.e):
    """
    NSB mutual information estimate from a full joint count table.

    Uses:
        I_NSB = H_NSB(X) + H_NSB(Y) - H_NSB(X,Y)

    The joint support is assumed to be the full shape of counts_xy.
    """
    counts_xy = np.asarray(counts_xy, dtype=float)

    counts_x = counts_xy.sum(axis=1)
    counts_y = counts_xy.sum(axis=0)

    Hx = nsb_entropy_from_counts(counts_x, base=base)
    Hy = nsb_entropy_from_counts(counts_y, base=base)
    Hxy = nsb_entropy_from_counts(counts_xy.ravel(), base=base)

    return float(Hx + Hy - Hxy)


def nsb_mi_from_samples(samples_x, samples_y, m, base = np.e):
    """
    NSB mutual information estimate from samples with known alphabet size m.

    Assumes X and Y both take values in {0,...,m-1}.
    The joint alphabet has size m^2.
    """
    counts = np.zeros((m, m), dtype=int)
    np.add.at(counts, (samples_x, samples_y), 1)

    return nsb_mi_from_counts_joint(counts, base=base)

def joint_counts_from_samples(x, y, m=2):
    """
    Convert paired samples into an m x m joint count table.
    """
    counts = np.zeros((m, m), dtype=int)
    np.add.at(counts, (np.asarray(x, dtype=int), np.asarray(y, dtype=int)), 1)
    return counts

def plugin_mi_from_counts(counts_xy, base=np.e):
    """
    Plug-in empirical mutual information from a joint count table.
    """
    counts_xy = np.asarray(counts_xy, dtype=float)
    N = counts_xy.sum()
    if N <= 0:
        return np.nan

    Pxy = counts_xy / N
    return mutual_information(Pxy, base=base)

def miller_madow_mi_from_counts(counts_xy, base=np.e):
    """
    Miller--Madow corrected mutual information from a joint count table.

    Uses:
        H_MM = H_plugin + (K_obs - 1) / (2N)

    so
        I_MM = I_plugin + (Kx_obs + Ky_obs - Kxy_obs - 1) / (2N)

    in nats. Converted to requested base if needed.
    """
    counts_xy = np.asarray(counts_xy, dtype=float)
    N = counts_xy.sum()
    if N <= 0:
        return np.nan

    I_plugin = plugin_mi_from_counts(counts_xy, base=base)

    counts_x = counts_xy.sum(axis=1)
    counts_y = counts_xy.sum(axis=0)

    Kx = np.sum(counts_x > 0)
    Ky = np.sum(counts_y > 0)
    Kxy = np.sum(counts_xy > 0)

    correction = (Kx + Ky - Kxy - 1) / (2.0 * N)

    if base != np.e:
        correction /= np.log(base)

    return float(I_plugin + correction)

