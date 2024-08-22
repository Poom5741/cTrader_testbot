import pandas as pd

# Load the WTI data
data = pd.read_csv('WTI_prices.csv', parse_dates=['Datetime'], index_col='Datetime')

# Define the backtesting function
def backtest_strategy(data, n):
    cash = 10000  # Starting cash
    position = 0  # Current position (0 means no position, 1 means holding)
    entry_price = 0  # Price at which the position was entered

    for i in range(n, len(data)):
        # Calculate the lowest low and highest high over the last n periods
        lowest_low = data['Low'].iloc[i-n:i].min()
        highest_high = data['High'].iloc[i-n:i].max()

        # Buy signal: close is lower than the lowest low of the last n candles
        if data['Close'].iloc[i] < lowest_low and position == 0:
            position = 1
            entry_price = data['Close'].iloc[i]
            print(f"Buying at {entry_price} on {data.index[i]}")

        # Sell signal: close is higher than the highest high of the last n candles
        elif data['Close'].iloc[i] > highest_high and position == 1:
            position = 0
            exit_price = data['Close'].iloc[i]
            cash += (exit_price - entry_price) * 1  # Assume 1 unit is traded
            print(f"Selling at {exit_price} on {data.index[i]}, Cash: {cash}")

    return cash

# Test different n values
best_n = None
best_value = float('-inf')

for n in range(2, 100):
    final_cash = backtest_strategy(data, n)
    print(f"n: {n}, Final Cash: {final_cash}")
    if final_cash > best_value:
        best_value = final_cash
        best_n = n

print(f'Best n: {best_n}, Final Portfolio Value: {best_value}')