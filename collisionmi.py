
import numpy as np
from math import comb
from scipy.special import gammaln, digamma, polygamma, logsumexp
import pandas as pd

from mi_approx import normalize, entropy, mutual_information

EPS = 1e-300




def collision_probability(values, k, mode="population"):
    """
    Compute an order-k collision probability/statistic.

    Parameters
    ----------
    values : array-like
        Input vector.

        mode="population":
            values are probabilities p_i.
            Returns C_k(P) = sum_i p_i^k.

        mode="plugin":
            values are counts n_i.
            Converts to empirical probabilities n_i/N and returns
            C_k(P_hat) = sum_i (n_i/N)^k.
            This is the with-replacement plug-in empirical statistic.

        mode="u_stat":
            values are counts n_i.
            Returns the without-replacement U-statistic
            sum_i binom(n_i,k) / binom(N,k).
            This is unbiased for C_k(P) for fixed integer k.

    k : int or float
        Collision order. For mode="u_stat", k must be an integer >= 1.

    mode : {"population", "plugin", "u_stat"}
        Which collision quantity to compute.

    Returns
    -------
    float
        Collision probability/statistic.
    """
    values = np.asarray(values)

    if mode == "population":
        p = values.astype(float).ravel()
        p = p[p > 0]
        return float(np.sum(p ** k))

    if mode == "plugin":
        counts = values.astype(float).ravel()
        N = counts.sum()
        if N <= 0:
            return np.nan
        p_hat = counts / N
        p_hat = p_hat[p_hat > 0]
        return float(np.sum(p_hat ** k))

    if mode == "u_stat":
        k_int = int(k)
        if k != k_int or k_int < 1:
            raise ValueError('For mode="u_stat", k must be an integer >= 1.')

        counts = values.astype(int).ravel()
        N = counts.sum()
        if N < k_int:
            return np.nan

        numerator = 0.0
        for n in counts:
            if n >= k_int:
                prod = 1.0
                for a in range(k_int):
                    prod *= (n - a)
                numerator += prod

        denominator = 1.0
        for a in range(k_int):
            denominator *= (N - a)

        return float(numerator / denominator)

    raise ValueError('mode must be one of {"population", "plugin", "u_stat"}.')


def renyi_entropy(values, alpha, base=np.e, mode="population"):
    """
    Rényi entropy H_alpha.

    Parameters
    ----------
    values : array-like
        If mode="population", values are probabilities p_i.
        If mode in {"plugin", "u_stat"}, values are counts n_i.

    alpha : float or int
        Rényi/collision order. For mode="u_stat", alpha must be an integer.

    base : float, default np.e
        Logarithm base.

    mode : {"population", "plugin", "u_stat"}
        Collision probability mode passed to collision_probability().
    """
    if np.isclose(alpha, 1.0):
        if mode == "population":
            return entropy(values, base=base)

        counts = np.asarray(values, dtype=float).ravel()
        N = counts.sum()
        if N <= 0:
            return np.nan
        return entropy(counts / N, base=base)

    C = collision_probability(values, alpha, mode=mode)
    if not np.isfinite(C) or C <= 0:
        return np.nan

    val = np.log(max(C, EPS)) / (1.0 - alpha)
    if base != np.e:
        val /= np.log(base)
    return float(val)


def K_alpha(values_xy, alpha, base=np.e, mode="population"):
    """
    Sign-corrected collision-based Rényi contrast:

        K_alpha(X;Y) = H_alpha(X) + H_alpha(Y) - H_alpha(X,Y)

    Parameters
    ----------
    values_xy : array-like, shape (m, n)
        If mode="population", entries are joint probabilities.
        If mode in {"plugin", "u_stat"}, entries are joint counts.

    alpha : float or int
        Rényi/collision order. For mode="u_stat", alpha must be an integer.

    base : float, default np.e
        Logarithm base.

    mode : {"population", "plugin", "u_stat"}
        Collision probability mode passed to renyi_entropy().
    """
    values_xy = np.asarray(values_xy)

    if mode == "population":
        Pxy = normalize(values_xy)
        px = Pxy.sum(axis=1)
        py = Pxy.sum(axis=0)

        if np.isclose(alpha, 1.0):
            return mutual_information(Pxy, base=base)

        return (
            renyi_entropy(px, alpha, base=base, mode=mode)
            + renyi_entropy(py, alpha, base=base, mode=mode)
            - renyi_entropy(Pxy.ravel(), alpha, base=base, mode=mode)
        )

    if mode in {"plugin", "u_stat"}:
        counts_xy = np.asarray(values_xy, dtype=int)
        counts_x = counts_xy.sum(axis=1)
        counts_y = counts_xy.sum(axis=0)

        if np.isclose(alpha, 1.0):
            Pxy_hat = counts_xy / counts_xy.sum()
            return mutual_information(Pxy_hat, base=base)

        return (
            renyi_entropy(counts_x, alpha, base=base, mode=mode)
            + renyi_entropy(counts_y, alpha, base=base, mode=mode)
            - renyi_entropy(counts_xy.ravel(), alpha, base=base, mode=mode)
        )

    raise ValueError('mode must be one of {"population", "plugin", "u_stat"}.')


def finite_resolution_MI(values_xy, r, base=np.e, mode="population"):
    """
    Extrapolated finite-resolution approximation using orders 2,...,r+1:

        I_tilde_r = sum_{j=1}^r (-1)^{j-1} binom(r,j) K_{j+1}

    where K_k is the order-k Rényi entropy contrast.

    Parameters
    ----------
    values_xy : array-like, shape (m, n)
        If mode="population", entries are joint probabilities.
        If mode in {"plugin", "u_stat"}, entries are joint counts.

    r : int
        Approximation order. Uses collision orders 2,...,r+1.

    base : float, default np.e
        Logarithm base.

    mode : {"population", "plugin", "u_stat"}
        Collision probability mode.
    """
    if int(r) != r or r < 1:
        raise ValueError("r must be an integer >= 1.")

    r = int(r)
    val = 0.0

    for j in range(1, r + 1):
        K = K_alpha(values_xy, j + 1, base=base, mode=mode)
        if not np.isfinite(K):
            return np.nan
        val += ((-1) ** (j - 1)) * comb(r, j) * K

    return float(val)