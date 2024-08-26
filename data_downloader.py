import yfinance as yf
import pandas as pd

# Define the ticker symbol for NAS100 futures
ticker_symbol = "NQ=F"

# Download the data
# Set the period and interval
data = yf.download(tickers=ticker_symbol, period="2y", interval="60m")

# Display the first few rows of the data
print(data.head())

# Optionally, save the data to a CSV file
data.to_csv('nas100_futures_1h.csv')