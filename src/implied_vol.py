import numpy as np
from scipy.optimize import brentq

from .black_scholes import option_price


def no_arbitrage_bounds(S, K, T, r, option_type="call", q=0.0):
    discount_spot = S * np.exp(-q * T)
    discount_strike = K * np.exp(-r * T)

    if option_type.lower() == "call":
        lower = max(0.0, discount_spot - discount_strike)
        upper = discount_spot

    elif option_type.lower() == "put":
        lower = max(0.0, discount_strike - discount_spot)
        upper = discount_strike

    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    return lower, upper


def implied_volatility(
    market_price,
    S,
    K,
    T,
    r,
    option_type="call",
    q=0.0,
    vol_lower=1e-6,
    vol_upper=5.0,
):
    lower, upper = no_arbitrage_bounds(
        S, K, T, r, option_type, q
    )

    if not (lower <= market_price <= upper):
        return np.nan

    def objective(sigma):
        return (
            option_price(
                S,
                K,
                T,
                r,
                sigma,
                option_type,
                q
            )
            - market_price
        )

    try:
        return brentq(
            objective,
            vol_lower,
            vol_upper,
            xtol=1e-10,
            rtol=1e-10,
            maxiter=100
        )

    except (ValueError, RuntimeError):
        return np.nan


def implied_volatility_series(df):
    df = df.copy()

    df["implied_vol"] = df.apply(
        lambda row: implied_volatility(
            market_price=row["mid"],
            S=row["spot"],
            K=row["strike"],
            T=row["T"],
            r=row["rate"],
            option_type=row["option_type"],
            q=row.get("dividend_yield", 0.0),
        ),
        axis=1
    )

    return df
