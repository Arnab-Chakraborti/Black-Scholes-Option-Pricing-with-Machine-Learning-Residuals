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

from .black_scholes import option_price


def generate_market_prices(df, noise_std=0.02, seed=42):

    df = df.copy()

    rng = np.random.default_rng(seed)

    df["bs_price"] = df.apply(
        lambda row: option_price(
            S=row["spot"],
            K=row["strike"],
            T=row["T"],
            r=row["rate"],
            sigma=row["true_vol"],
            option_type=row["option_type"],
            q=row["dividend_yield"],
        ),
        axis=1
    )

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=len(df)
    )

    df["mid"] = np.maximum(
        df["bs_price"] + noise,
        0.001
    )

    # Simulated bid/ask spread
    spread = 0.02 + 0.005 * df["mid"]

    df["bid"] = np.maximum(
        df["mid"] - spread / 2,
        0.001
    )

    df["ask"] = df["mid"] + spread / 2

    # Simulated volume
    df["volume"] = rng.integers(
        10,
        1000,
        size=len(df)
    )

    # Simulated open interest
    df["open_interest"] = rng.integers(
        100,
        10000,
        size=len(df)
    )

    return df

def add_moneyness(df):

    df = df.copy()

    df["moneyness"] = (
        df["strike"] / df["spot"]
    )

    df["log_moneyness"] = np.log(
        df["strike"] / df["spot"]
    )

    return df

def build_synthetic_dataset():

    df = create_synthetic_option_chain()

    df = generate_market_prices(df)

    df = add_moneyness(df)

    return df
