import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 1: Define tickers for the assets in your portfolio
tickers = ['JNJ', 'PG', 'DUK', 'MSFT', 'HRL', 'NEE', 'CVS', 'GLD', 'TLT', 'IEF', 'BND', 'VNQ']  # Add more as needed
start_date = '2000-01-01'  # Start date before the dot-com bubble
end_date = '2020-12-31'  # End date after the COVID-19 recession

# Step 2: Download the historical adjusted close prices
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Step 3: Normalize the data (all starting at 100)
normalized_data = data / data.iloc[0] * 100

# Step 4: Define portfolio allocation (sum must be 1)
weights = [0.10, 0.07, 0.05, 0.03, 0.05, 0.03, 0.02, 0.10, 0.10, 0.10, 0.10, 0.15]  # Corresponding to the tickers

# Step 5: Calculate daily returns
daily_returns = data.pct_change().dropna()

# Step 6: Calculate portfolio returns
portfolio_daily_returns = daily_returns.dot(weights)
cumulative_returns = (1 + portfolio_daily_returns).cumprod()

# Step 7: Plot the portfolio performance over time
plt.figure(figsize=(12, 8))
plt.plot(cumulative_returns, label="Portfolio Returns")
plt.title('Portfolio Performance from 2000 to 2020')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

# Step 8: Calculate quantitative metrics

# Sharpe Ratio
risk_free_rate = 0.01  # Assumed annual risk-free rate (e.g., 1%)
sharpe_ratio = (portfolio_daily_returns.mean() - risk_free_rate/252) / portfolio_daily_returns.std() * np.sqrt(252)
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Sortino Ratio (downside deviation)
negative_returns = portfolio_daily_returns[portfolio_daily_returns < 0]
sortino_ratio = (portfolio_daily_returns.mean() - risk_free_rate/252) / negative_returns.std() * np.sqrt(252)
print(f"Sortino Ratio: {sortino_ratio:.2f}")

# Value at Risk (VaR)
confidence_level = 0.95
portfolio_mean = portfolio_daily_returns.mean()
portfolio_std = portfolio_daily_returns.std()
VaR = norm.ppf(1 - confidence_level) * portfolio_std - portfolio_mean
VaR_annualized = VaR * np.sqrt(252)
print(f"Value at Risk (95%): {-VaR_annualized:.2%}")

# Step 9: Qualitative analysis through visualizing key recession periods

recession_periods = {
    'Dot-com Bubble': ('2000-03-01', '2002-10-01'),
    'Global Financial Crisis': ('2007-12-01', '2009-06-01'),
    'COVID-19 Recession': ('2020-02-01', '2020-04-01')
}

for recession, (start, end) in recession_periods.items():
    recession_returns = cumulative_returns.loc[start:end]
    plt.plot(recession_returns, label=f'{recession}')
    plt.title(f'Portfolio Performance During {recession}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.legend()
    plt.show()
