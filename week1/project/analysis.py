# Week 1 Mini-Project: Load, clean, and visualize real EUR/USD price data.
# Combines: pandas loading, null handling, filtering, and matplotlib/seaborn plotting.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- LOAD ----
df = pd.read_csv("week1/project/eurusd.csv")
print("Shape:", df.shape)
print(df.head())

# ---- CLEAN ----
print("\nMissing values per column:")
print(df.isnull().sum())

# (decide: fillna or dropna based on what's actually missing)

# ---- FEATURE: daily % return ----
df["daily_return_pct"] = df["Close"].pct_change() * 100

# ---- PLOT 1: Close price over time ----
plt.figure()
plt.plot(df["Date"], df["Close"])
plt.title("EUR/USD Close Price - Last Year")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("week1/project/plot1_close_price.png")

# ---- PLOT 2: Daily returns distribution ----
plt.figure()
sns.histplot(df["daily_return_pct"].dropna(), bins=30)
plt.title("Distribution of Daily Returns (%)")
plt.xlabel("Daily Return %")
plt.savefig("week1/project/plot2_returns_distribution.png")

# ---- PLOT 3: High-Low range (volatility proxy) over time ----
df["daily_range"] = df["High"] - df["Low"]
plt.figure()
plt.plot(df["Date"], df["daily_range"])
plt.title("Daily Price Range (High - Low)")
plt.xlabel("Date")
plt.ylabel("Range")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("week1/project/plot3_daily_range.png")

print("\nAll 3 plots saved to week1/project/")
