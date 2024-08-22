import yfinance as yf

# Define the ticker symbol for Crude Oil Futures
ticker = 'CL=F'

# Download the historical data
data = yf.download(ticker, interval='1h', start='2023-01-01', end='2023-12-31')

# Display the data
print(data)

# Save the data to a CSV file
data.to_csv('WTI_prices.csv')