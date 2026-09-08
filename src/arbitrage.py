import numpy as np
import pandas as pd

from .implied_vol import no_arbitrage_bounds


def check_price_bounds(df):
    """
    Check whether option prices lie within basic
    no-arbitrage bounds.
    """

    results = df.copy()

    bounds = results.apply(
        lambda row: no_arbitrage_bounds(
            S=row["spot"],
            K=row["strike"],
            T=row["T"],
            r=row["rate"],
            option_type=row["option_type"],
            q=row["dividend_yield"],
        ),
        axis=1,
        result_type="expand",
    )

    results["lower_bound"] = bounds[0]
    results["upper_bound"] = bounds[1]

    results["bounds_violation"] = (
        (results["mid"] < results["lower_bound"])
        | (results["mid"] > results["upper_bound"])
    )

    return results


def check_put_call_parity(df, tolerance=0.05):
    """
    Check put-call parity:

        C - P = S*exp(-qT) - K*exp(-rT)

    Returns one row per strike/maturity combination.
    """

    calls = (
        df[df["option_type"] == "call"][
            [
                "spot",
                "strike",
                "T",
                "rate",
                "dividend_yield",
                "mid",
            ]
        ]
        .rename(columns={"mid": "call_price"})
    )

    puts = (
        df[df["option_type"] == "put"][
            [
                "spot",
                "strike",
                "T",
                "rate",
                "dividend_yield",
                "mid",
            ]
        ]
        .rename(columns={"mid": "put_price"})
    )

    merged = calls.merge(
        puts,
        on=[
            "spot",
            "strike",
            "T",
            "rate",
            "dividend_yield",
        ],
    )

    merged["parity_lhs"] = (
        merged["call_price"]
        - merged["put_price"]
    )

    merged["parity_rhs"] = (
        merged["spot"]
        * np.exp(-merged["dividend_yield"] * merged["T"])
        - merged["strike"]
        * np.exp(-merged["rate"] * merged["T"])
    )

    merged["parity_error"] = (
        merged["parity_lhs"]
        - merged["parity_rhs"]
    )

    merged["parity_violation"] = (
        merged["parity_error"].abs() > tolerance
    )

    return merged[
        [
            "spot",
            "strike",
            "T",
            "call_price",
            "put_price",
            "parity_lhs",
            "parity_rhs",
            "parity_error",
            "parity_violation",
        ]
    ]

def check_call_monotonicity(df):
    """
    Call prices should be non-increasing with strike.
    """

    calls = df[
        df["option_type"] == "call"
    ].copy()

    violations = []

    for T, group in calls.groupby("T"):

        group = group.sort_values("strike")

        prices = group["mid"].values
        strikes = group["strike"].values

        for i in range(1, len(group)):

            if prices[i] > prices[i - 1]:
                violations.append({
                    "T": T,
                    "lower_strike": strikes[i - 1],
                    "higher_strike": strikes[i],
                    "lower_strike_price": prices[i - 1],
                    "higher_strike_price": prices[i],
                    "violation": True,
                })

    return pd.DataFrame(violations)


def check_put_monotonicity(df):
    """
    Put prices should be non-decreasing with strike.
    """

    puts = df[
        df["option_type"] == "put"
    ].copy()

    violations = []

    for T, group in puts.groupby("T"):

        group = group.sort_values("strike")

        prices = group["mid"].values
        strikes = group["strike"].values

        for i in range(1, len(group)):

            if prices[i] < prices[i - 1]:
                violations.append({
                    "T": T,
                    "lower_strike": strikes[i - 1],
                    "higher_strike": strikes[i],
                    "lower_strike_price": prices[i - 1],
                    "higher_strike_price": prices[i],
                    "violation": True,
                })

    return pd.DataFrame(violations)


def check_convexity(df):
    """
    Option prices should be convex in strike.

    For equally spaced strikes:

        C(K_{i-1}) - 2C(K_i) + C(K_{i+1}) >= 0
    """

    violations = []

    for option_type, type_group in df.groupby("option_type"):

        for T, group in type_group.groupby("T"):

            group = group.sort_values("strike")

            strikes = group["strike"].values
            prices = group["mid"].values

            for i in range(1, len(group) - 1):

                second_difference = (
                    prices[i - 1]
                    - 2 * prices[i]
                    + prices[i + 1]
                )

                if second_difference < -1e-6:
                    violations.append({
                        "option_type": option_type,
                        "T": T,
                        "strike": strikes[i],
                        "second_difference": second_difference,
                        "violation": True,
                    })

    return pd.DataFrame(violations)


def check_calendar_consistency(df):
    """
    Basic calendar consistency check.

    For a fixed strike, option prices should generally
    not decrease with maturity when using the standard
    European option assumptions and non-negative rates.

    This is a simplified diagnostic rather than a complete
    calendar-arbitrage test.
    """

    violations = []

    for option_type, type_group in df.groupby("option_type"):

        for strike, group in type_group.groupby("strike"):

            group = group.sort_values("T")

            prices = group["mid"].values
            maturities = group["T"].values

            for i in range(1, len(group)):

                if prices[i] < prices[i - 1] - 1e-6:

                    violations.append({
                        "option_type": option_type,
                        "strike": strike,
                        "shorter_T": maturities[i - 1],
                        "longer_T": maturities[i],
                        "shorter_price": prices[i - 1],
                        "longer_price": prices[i],
                        "violation": True,
                    })

    return pd.DataFrame(violations)


def run_all_arbitrage_checks(df):
    """
    Run all arbitrage diagnostics.
    """

    bounds = check_price_bounds(df)

    parity = check_put_call_parity(df)

    call_monotonicity = check_call_monotonicity(df)

    put_monotonicity = check_put_monotonicity(df)

    convexity = check_convexity(df)

    calendar = check_calendar_consistency(df)

    return {
        "bounds": bounds,
        "put_call_parity": parity,
        "call_monotonicity": call_monotonicity,
        "put_monotonicity": put_monotonicity,
        "convexity": convexity,
        "calendar": calendar,
    }
