import numpy as np
import pandas as pd


def create_synthetic_option_chain(
    spot=100.0,
    rate=0.05,
    dividend_yield=0.0,
    seed=42,
):
    """
    Generate a reproducible synthetic option chain.

    The generated market prices include a controlled volatility
    smile/skew so that the downstream IV engine has realistic
    structure to recover.
    """

    rng = np.random.default_rng(seed)

    maturities = np.array([
        30, 60, 90, 180, 365
    ])

    strikes = np.arange(
        70,
        131,
        5
    )

    rows = []

    for days in maturities:

        T = days / 365.0

        for strike in strikes:

            log_moneyness = np.log(strike / spot)

            # Base volatility
            base_vol = 0.20

            # Volatility smile/skew
            skew = -0.08 * log_moneyness
            curvature = 0.20 * log_moneyness**2

            sigma = (
                base_vol
                + skew
                + curvature
                + 0.02 * np.sqrt(T)
            )

            sigma = max(sigma, 0.05)

            for option_type in ["call", "put"]:

                rows.append({
                    "spot": spot,
                    "strike": strike,
                    "expiry_days": days,
                    "T": T,
                    "rate": rate,
                    "dividend_yield": dividend_yield,
                    "option_type": option_type,
                    "true_vol": sigma,
                })

    df = pd.DataFrame(rows)

    return df
