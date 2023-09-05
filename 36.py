import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('stock_data.csv')

print(data.head())

dates = data['Date']
closing_prices = data['Close']

daily_returns = closing_prices.pct_change().dropna()

mean_return = daily_returns.mean()
std_dev = daily_returns.std()
variance = std_dev**2

cumulative_returns = (1 + daily_returns).cumprod()

# Plot the daily returns
plt.figure(figsize=(12, 6))
plt.plot(dates[1:], daily_returns)
plt.title('Daily Returns')
plt.xlabel('Date')
plt.ylabel('Return')
plt.grid(True)
plt.show()

print(f"Mean Daily Return: {mean_return:.4f}")
print(f"Standard Deviation of Daily Return: {std_dev:.4f}")
print(f"Variance of Daily Return: {variance:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(dates[1:], cumulative_returns)
plt.title('Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()
