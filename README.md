# Black–Scholes Option Pricing with Machine Learning Residual Correction  

## Project Overview  
This project combines the **Black–Scholes option pricing model** with **Machine Learning** to improve accuracy in pricing European call and put options.  
- Implemented the Black–Scholes closed-form solution and Greeks.  
- Computed implied volatility (IV) from market option data using numerical methods.  
- Trained ML models to capture residual errors between Black–Scholes theoretical prices and actual market prices.  
- Achieved a hybrid approach that balances **theoretical interpretability** with **data-driven accuracy**.  

## Features  
- Black–Scholes implementation (call/put pricing + Greeks).  
- Implied volatility estimation via Newton–Raphson / Brent’s method.  
- Data preprocessing and feature engineering for options data.  
- ML regressors (Random Forest, XGBoost, Neural Nets) for residual correction.  
- Visualization of volatility smile/surface and pricing improvements.  

## Results  
- Baseline: Black–Scholes option prices vs market prices.  
- Improved hybrid predictions after residual correction.  
- Visualization of pricing error reduction.  

##  Tech Stack  
- **Python**: NumPy, Pandas, Scikit-learn, Matplotlib/Seaborn  
- **ML models**: Random Forest, XGBoost, Neural Networks  
- **Optimization**: SciPy (for root-finding in IV estimation)  

##  Project Structure  
```
├── data/                 # Option market data (or sample CSV)
├── notebooks/            # Jupyter notebooks for experiments
├── src/                  # Core implementation (BS model, ML training, utils)
├── results/              # Plots, evaluation metrics
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

##  How to Run  
1. Clone the repo  
```bash
git clone https://github.com/your-username/black-scholes-ml.git
cd black-scholes-ml
```
2. Install dependencies  
```bash
pip install -r requirements.txt
```
3. Run the notebook in `notebooks/` to reproduce results.  

## 📖 References  
- Paul Wilmott introduces Quantitative Finance
- Youtube: Volatility Surface and Volatility Smile Explained (https://youtu.be/G7gf-oXptxE?si=FdCG6jFAkFeOhtPV) 
- The Trillion Dollar Equation (https://youtu.be/A5w-dEgIU1M?si=WN5PmIXNTTJpqpyx)
