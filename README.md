# Option Pricing & Volatility Surface Modeling

A quantitative options pricing framework built around **Black-Scholes pricing, analytical Greeks, implied-volatility estimation, volatility-surface construction, arbitrage diagnostics, stress testing, and machine-learning residual modeling**.

The project uses a modular Python architecture with an interactive Streamlit interface for exploring option prices, sensitivities, and implied-volatility dynamics.

---

## Overview

Black-Scholes provides a useful theoretical benchmark for European option pricing, but real option markets exhibit features such as volatility smiles, skew, and term-structure effects that cannot be captured by a single constant volatility assumption.

This project builds a complete workflow:

```text
Option Chain
     │
     ▼
Data Processing
     │
     ▼
Black-Scholes Pricing
     │
     ├── Greeks
     │
     └── Market Price
              │
              ▼
      Implied Volatility
              │
              ▼
     Volatility Surface
       ├── Smile / Skew
       ├── Term Structure
       └── Surface
              │
              ▼
      Arbitrage Diagnostics
              │
              ▼
        Stress Testing
              │
              ▼
      ML Residual Modeling
```

---

## Features

### 1. Black-Scholes Pricing

Implements European call and put pricing with support for:

* Spot price
* Strike price
* Time to maturity
* Risk-free interest rate
* Volatility
* Dividend yield

The implementation also exposes the intermediate \(d_1\) and \(d_2\) quantities used throughout the pricing framework.

### 2. Analytical Greeks

The framework calculates:

* Delta
* Gamma
* Vega
* Theta
* Rho

These measures are used for sensitivity analysis and stress testing.

### 3. Implied Volatility

Implied volatility is obtained by numerically solving:

$$
C_{BS}(S,K,T,r,\sigma) - C_{market}=0
$$

The implementation uses **Brent's root-finding algorithm**, providing a robust alternative to Newton-style iteration when volatility sensitivities become small.

Basic no-arbitrage price bounds are checked before attempting the inversion.

### 4. Volatility Smile and Term Structure

The implied-volatility framework allows option data to be analyzed across:

* Strike / moneyness
* Time to maturity
* Log-moneyness

This produces:

* Volatility smiles
* Volatility skew
* ATM volatility term structures
* Two-dimensional implied-volatility surfaces

The surface is represented approximately as:

$$
\sigma_{IV} = \sigma(k,T)
$$

where

$$
k = \log(K/S)
$$

is log-moneyness.

### 5. Volatility Surface

The project interpolates observed implied volatilities onto a regular:

```text
Log Moneyness × Time to Maturity
```

grid.

This allows the volatility structure of the option chain to be visualized as a three-dimensional surface.

### 6. No-Arbitrage Diagnostics

The framework is designed to evaluate basic static consistency conditions in option prices, including:

* Call/put price bounds
* Put-call parity
* Monotonicity with respect to strike
* Convexity across strikes
* Calendar consistency

These checks are useful for identifying inconsistent or potentially noisy option observations.

### 7. Stress Testing

Option portfolios can be evaluated under shocks to:

* Spot price
* Volatility
* Interest rates
* Time to maturity

The framework compares full repricing against Greek-based approximations such as:

$$
\Delta V
\approx
\Delta \Delta S
+
\frac{1}{2}\Gamma(\Delta S)^2
+
Vega\Delta\sigma
+
\rho\Delta r
+
\Theta\Delta t
$$

This provides a simple framework for understanding nonlinear option sensitivities.

### 8. Machine-Learning Residual Modeling

Rather than replacing Black-Scholes entirely with a machine-learning model, the project treats Black-Scholes as a theoretical baseline.

The residual is defined as:

$$
\epsilon =
C_{market}-C_{BS}
$$

A machine-learning model can then learn systematic deviations from the theoretical benchmark:

$$
\hat{C}
=
C_{BS}
+
\hat{\epsilon}
$$

Features can include:

* Spot price
* Strike
* Maturity
* Moneyness
* Implied volatility
* Greeks
* Interest rate
* Trading activity

This provides a hybrid **quantitative model + machine-learning correction** approach.

---

## Project Structure

```text
option-pricing-vol-surface/
│
├── app.py
├── visualize_surface.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── black_scholes.py
│   ├── greeks.py
│   ├── implied_vol.py
│   ├── market_data.py
│   ├── volatility_surface.py
│   ├── arbitrage.py
│   ├── stress_testing.py
│   └── ml_residual.py
│
└── tests/
    ├── test_black_scholes.py
    ├── test_greeks.py
    └── test_implied_vol.py
```

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd option-pricing-vol-surface

pip install -r requirements.txt
```

---

## Quick Start

### Generate the volatility surface

Run:

```bash
python visualize_surface.py
```

This generates:

1. Implied-volatility smile
2. ATM volatility term structure
3. Three-dimensional implied-volatility surface

### Run the Streamlit application

```bash
streamlit run app.py
```

The application provides an interactive interface for experimenting with pricing parameters, Greeks, implied volatility, and volatility-surface behavior.

---

## Example: Black-Scholes Pricing

```python
from src.black_scholes import option_price

price = option_price(
    S=100,
    K=100,
    T=1,
    r=0.05,
    sigma=0.20,
    option_type="call"
)

print(price)
```

Output:

```text
10.4506
```

---

## Example: Greeks

```python
from src.greeks import all_greeks

greeks = all_greeks(
    S=100,
    K=100,
    T=1,
    r=0.05,
    sigma=0.20,
    option_type="call"
)

print(greeks)
```

Example output:

```text
{
    'delta': 0.6368,
    'gamma': 0.01876,
    'vega': 37.5240,
    'theta': -6.4140,
    'rho': 53.2325
}
```

---

## Data

The current implementation includes a reproducible synthetic option-chain generator.

The synthetic dataset contains multiple:

* Strikes
* Expiries
* Call/put contracts

Market prices are generated from Black-Scholes using a controlled strike- and maturity-dependent volatility structure, followed by small perturbations to simulate market noise and bid/ask spreads.

This allows the entire pipeline to be tested without requiring a live options-data subscription.

The data-generation layer can later be replaced with real market data without changing the core pricing and volatility modules.

---

## Methodology

### Pricing

Black-Scholes is used as the initial theoretical pricing benchmark.

### Calibration

Market option prices are inverted into implied volatilities using numerical root finding.

### Surface Construction

Implied volatilities are parameterized by:

$$
(k,T)
$$

where \(k\) is log-moneyness and \(T\) is time to maturity.

### Validation

The framework checks:

* Pricing bounds
* Put-call parity
* Strike monotonicity
* Strike convexity
* Cross-maturity consistency

### Risk Analysis

Analytical Greeks are combined with full repricing to evaluate sensitivity to market shocks.

### Machine Learning

Machine learning is used as a residual correction layer rather than as a replacement for the underlying financial model.

---

## Testing

Run the test suite with:

```bash
pytest
```

The tests cover core functionality including:

* Black-Scholes pricing
* Put-call parity
* Greek calculations
* Implied-volatility recovery
* Input validation

---

## Limitations

This project is intended as a research and educational quantitative-finance framework rather than a production trading or risk-management system.

Current limitations include:

* Synthetic market data
* European option assumptions
* Simplified interest-rate and dividend inputs
* Basic surface interpolation
* No transaction-cost modeling
* No stochastic-volatility model
* No jump-diffusion model
* No advanced volatility-surface parameterization such as SVI
* No live execution or trading functionality

These components can be added as extensions.

---

## Potential Extensions

Possible future improvements include:

* SVI volatility-surface parameterization
* Heston stochastic-volatility calibration
* Local-volatility modeling
* Monte Carlo option pricing
* Real-time market-data integration
* Portfolio-level Greeks
* VaR / Expected Shortfall
* Dynamic volatility-surface tracking
* More advanced ML residual models
* Model-comparison framework across pricing methodologies

---

## Technologies

* Python
* NumPy
* Pandas
* SciPy
* Scikit-learn
* XGBoost
* Plotly
* Streamlit
* yfinance
* PyTest

---

## Disclaimer

This project is intended for **research, educational, and portfolio purposes**. It does not constitute financial advice or a production-ready pricing/risk-management system.

---

## Author

**Arnab Chakraborti**

Quantitative Finance / Machine Learning

