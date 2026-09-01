import numpy as np
from scipy.stats import norm

from .black_scholes import d1_d2


def delta(S, K, T, r, sigma, option_type="call", q=0.0):
    d1, _ = d1_d2(S, K, T, r, sigma, q)

    if option_type.lower() == "call":
        return np.exp(-q * T) * norm.cdf(d1)

    if option_type.lower() == "put":
        return np.exp(-q * T) * (norm.cdf(d1) - 1)

    raise ValueError("option_type must be 'call' or 'put'.")


def gamma(S, K, T, r, sigma, q=0.0):
    d1, _ = d1_d2(S, K, T, r, sigma, q)

    return (
        np.exp(-q * T)
        * norm.pdf(d1)
        / (S * sigma * np.sqrt(T))
    )


def vega(S, K, T, r, sigma, q=0.0):
    d1, _ = d1_d2(S, K, T, r, sigma, q)

    return (
        S
        * np.exp(-q * T)
        * norm.pdf(d1)
        * np.sqrt(T)
    )


def theta(S, K, T, r, sigma, option_type="call", q=0.0):
    d1, d2 = d1_d2(S, K, T, r, sigma, q)

    first_term = (
        -S
        * np.exp(-q * T)
        * norm.pdf(d1)
        * sigma
        / (2 * np.sqrt(T))
    )

    if option_type.lower() == "call":
        return (
            first_term
            + q * S * np.exp(-q * T) * norm.cdf(d1)
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )

    if option_type.lower() == "put":
        return (
            first_term
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )

    raise ValueError("option_type must be 'call' or 'put'.")


def rho(S, K, T, r, sigma, option_type="call", q=0.0):
    _, d2 = d1_d2(S, K, T, r, sigma, q)

    if option_type.lower() == "call":
        return (
            K
            * T
            * np.exp(-r * T)
            * norm.cdf(d2)
        )

    if option_type.lower() == "put":
        return (
            -K
            * T
            * np.exp(-r * T)
            * norm.cdf(-d2)
        )

    raise ValueError("option_type must be 'call' or 'put'.")


def all_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    return {
        "delta": delta(S, K, T, r, sigma, option_type, q),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, option_type, q),
        "rho": rho(S, K, T, r, sigma, option_type, q),
    }
