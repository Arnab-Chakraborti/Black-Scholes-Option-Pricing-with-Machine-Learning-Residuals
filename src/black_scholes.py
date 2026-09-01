import numpy as np
from scipy.stats import norm


def _validate_inputs(S, K, T, sigma):
    if np.any(np.asarray(S) <= 0):
        raise ValueError("Spot price S must be positive.")

    if np.any(np.asarray(K) <= 0):
        raise ValueError("Strike K must be positive.")

    if np.any(np.asarray(T) <= 0):
        raise ValueError("Time to maturity T must be positive.")

    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("Volatility sigma must be positive.")


def d1_d2(S, K, T, r, sigma, q=0.0):
    _validate_inputs(S, K, T, sigma)

    sqrt_T = np.sqrt(T)

    d1 = (
        np.log(S / K)
        + (r - q + 0.5 * sigma**2) * T
    ) / (sigma * sqrt_T)

    d2 = d1 - sigma * sqrt_T

    return d1, d2


def call_price(S, K, T, r, sigma, q=0.0):
    d1, d2 = d1_d2(S, K, T, r, sigma, q)

    return (
        S * np.exp(-q * T) * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )


def put_price(S, K, T, r, sigma, q=0.0):
    d1, d2 = d1_d2(S, K, T, r, sigma, q)

    return (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * np.exp(-q * T) * norm.cdf(-d1)
    )


def option_price(
    S,
    K,
    T,
    r,
    sigma,
    option_type="call",
    q=0.0
):
    option_type = option_type.lower()

    if option_type == "call":
        return call_price(S, K, T, r, sigma, q)

    if option_type == "put":
        return put_price(S, K, T, r, sigma, q)

    raise ValueError("option_type must be 'call' or 'put'.")
