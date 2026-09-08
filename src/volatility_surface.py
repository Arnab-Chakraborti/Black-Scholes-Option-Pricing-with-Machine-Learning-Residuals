import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def prepare_surface_data(df):
    """
    Prepare option data for volatility surface construction.
    Uses log-moneyness and implied volatility.
    """

    surface_df = df.copy()

    # Remove invalid IV observations
    surface_df = surface_df[
        surface_df["implied_vol"].notna()
    ].copy()

    surface_df = surface_df[
        surface_df["implied_vol"] > 0
    ].copy()

    return surface_df


def get_smile(df, maturity):
    """
    Return implied volatility smile for a given maturity.
    """

    data = prepare_surface_data(df)

    smile = data[
        np.isclose(data["T"], maturity)
    ].copy()

    smile = smile.sort_values("log_moneyness")

    return smile[
        [
            "strike",
            "moneyness",
            "log_moneyness",
            "implied_vol"
        ]
    ]


def get_term_structure(df, log_moneyness=0.0):
    """
    Return IV term structure near a specified log-moneyness.
    """

    data = prepare_surface_data(df)

    # Find observations closest to requested moneyness
    data["moneyness_distance"] = (
        data["log_moneyness"] - log_moneyness
    ).abs()

    closest = (
        data
        .sort_values("moneyness_distance")
        .groupby("T")
        .first()
        .reset_index()
    )

    return closest[
        [
            "T",
            "implied_vol",
            "log_moneyness"
        ]
    ].sort_values("T")


def build_surface_grid(
    df,
    n_moneyness=50,
    n_maturity=50
):
    """
    Interpolate implied volatility onto a regular
    moneyness-maturity grid.
    """

    data = prepare_surface_data(df)

    x = data["log_moneyness"].values
    y = data["T"].values
    z = data["implied_vol"].values

    grid_x = np.linspace(
        x.min(),
        x.max(),
        n_moneyness
    )

    grid_y = np.linspace(
        y.min(),
        y.max(),
        n_maturity
    )

    X, Y = np.meshgrid(
        grid_x,
        grid_y
    )

    Z = griddata(
        points=(x, y),
        values=z,
        xi=(X, Y),
        method="linear"
    )

    return X, Y, Z


def calculate_total_variance(df):
    """
    Calculate total implied variance:

        w = sigma^2 * T
    """

    data = prepare_surface_data(df)

    data["total_variance"] = (
        data["implied_vol"] ** 2
        * data["T"]
    )

    return data
