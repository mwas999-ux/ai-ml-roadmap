# Pulls ~1 year of daily EUR/USD price data and saves it locally as a clean CSV.
# This simulates "finding" a real dataset — exactly what you'll do
# constantly once you start building the trading bot (Week 10+).

import yfinance as yf

data = yf.download("EURUSD=X", period="1y", interval="1d")

# yfinance gives back multi-level column headers (Price/Ticker) - flatten them
data.columns = data.columns.get_level_values(0)
data = data.reset_index()  # turn the Date index into a normal column

data.to_csv("week1/project/eurusd.csv", index=False)

print("Saved", len(data), "rows to week1/project/eurusd.csv")
print(data.head())
