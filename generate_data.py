# -*- coding: utf-8 -*-
"""Welcome To Colab

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/notebooks/intro.ipynb
"""

import pandas as pd
import yfinance as yf
from fredapi import Fred
from getpass import getpass
import os

# === Securely input API key ===
fred_api_key = getpass("\U0001F511 Enter your FRED API key: ")
fred = Fred(api_key=fred_api_key)

# === Download crude oil price data ===
oil_df = yf.download("CL=F", start="2010-01-01", end="2024-12-31")

# Use 'Close' if 'Adj Close' is missing
if 'Adj Close' not in oil_df.columns:
    oil_df['Adj Close'] = oil_df['Close']

# Flatten MultiIndex if present
oil_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in oil_df.columns.values]

# Reset index and rename columns
oil_df = oil_df.reset_index()

# Rename column to avoid KeyError in downstream scripts
oil_df.rename(columns={'Adj Close_': 'Crude_Oil_Price'}, inplace=True)
oil_df['Date'] = pd.to_datetime(oil_df['Date'])

# === Fetch macroeconomic indicators ===
# FRED codes: GDP, CPI, UNRATE (unemployment rate), etc.
macro_series = {
    'GDP': 'GDP',
    'CPI': 'CPIAUCSL',
    'UNRATE': 'UNRATE'
}

macro_data = pd.DataFrame()

for name, code in macro_series.items():
    series = fred.get_series(code)
    df = pd.DataFrame(series)
    df.reset_index(inplace=True)
    df.columns = ['Date', name]
    df['Date'] = pd.to_datetime(df['Date'])
    if macro_data.empty:
        macro_data = df
    else:
        macro_data = pd.merge(macro_data, df, on='Date', how='outer')

# Reset index and ensure flat structure
macro_data = macro_data.reset_index(drop=True)

# === Merge datasets ===
merged_df = pd.merge(oil_df, macro_data, on='Date', how='inner')
merged_df = merged_df.sort_values('Date').reset_index(drop=True)

# === Save the data ===
os.makedirs("data", exist_ok=True)
merged_df.to_csv("crude_oil_macro_data.csv", index=False)

print("\u2705 Data successfully downloaded and saved to crude_oil_macro_data.csv'")