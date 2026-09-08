import matplotlib.pyplot as plt

from src.market_data import build_synthetic_dataset
from src.implied_vol import implied_volatility_series
from src.volatility_surface import (
    get_smile,
    get_term_structure,
    build_surface_grid
)


# --------------------------------
# Build dataset
# --------------------------------

df = build_synthetic_dataset()

df = implied_volatility_series(df)


# --------------------------------
# Volatility Smile
# --------------------------------

maturity = 30 / 365

smile = get_smile(
    df,
    maturity
)

plt.figure(figsize=(9, 5))

plt.plot(
    smile["strike"],
    smile["implied_vol"] * 100,
    marker="o"
)

plt.xlabel("Strike")
plt.ylabel("Implied Volatility (%)")
plt.title("30-Day Implied Volatility Smile")

plt.grid(True)
plt.show()


# --------------------------------
# Term Structure
# --------------------------------

term = get_term_structure(
    df,
    log_moneyness=0.0
)

plt.figure(figsize=(9, 5))

plt.plot(
    term["T"] * 365,
    term["implied_vol"] * 100,
    marker="o"
)

plt.xlabel("Days to Maturity")
plt.ylabel("Implied Volatility (%)")
plt.title("ATM Implied Volatility Term Structure")

plt.grid(True)
plt.show()


# --------------------------------
# Volatility Surface
# --------------------------------

X, Y, Z = build_surface_grid(df)

fig = plt.figure(figsize=(11, 8))

ax = fig.add_subplot(
    111,
    projection="3d"
)

surface = ax.plot_surface(
    X,
    Y * 365,
    Z * 100,
    cmap="viridis"
)

ax.set_xlabel("Log Moneyness")
ax.set_ylabel("Days to Maturity")
ax.set_zlabel("Implied Volatility (%)")

ax.set_title(
    "Implied Volatility Surface"
)

fig.colorbar(
    surface,
    shrink=0.6,
    aspect=10,
    label="IV (%)"
)

plt.show()
