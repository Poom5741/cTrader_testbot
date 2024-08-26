import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load your data
df = pd.read_csv('WTI_prices.csv')  # Replace with your actual data file
df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
df.set_index('Datetime', inplace=True)

# Ensure the dataframe has the required columns
required_columns = ['Open', 'High', 'Low', 'Close']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Dataframe must have columns: {required_columns}")

# Remove any rows with NaN values
df.dropna(inplace=True)

# Ensure the dataframe is not empty
if df.empty:
    raise ValueError("Dataframe is empty after processing")

def is_bullish_fractal(data, i, window):
    if i < window or i >= len(data) - window:
        return False
    middle_low = data.iloc[i]['Low']
    for j in range(1, window + 1):
        if data.iloc[i-j]['Low'] <= middle_low or data.iloc[i+j]['Low'] <= middle_low:
            return False
    return True

def is_bearish_fractal(data, i, window):
    if i < window or i >= len(data) - window:
        return False
    middle_high = data.iloc[i]['High']
    for j in range(1, window + 1):
        if data.iloc[i-j]['High'] >= middle_high or data.iloc[i+j]['High'] >= middle_high:
            return False
    return True

def backtest_strategy(df, window_size=2, stop_loss_pct=4.84, take_profit_pct=4.45):
    df['Signal'] = 0
    df['Position'] = 0
    df['Entry_Price'] = 0.0
    df['Exit_Price'] = 0.0
    df['PnL'] = 0.0
    
    position = 0
    entry_price = 0
    
    for i in range(len(df)):
        if position == 0:
            if is_bullish_fractal(df, i, window_size):
                position = 1
                entry_price = df.iloc[i]['Close']
                df.iloc[i, df.columns.get_loc('Signal')] = 1
                df.iloc[i, df.columns.get_loc('Position')] = 1
                df.iloc[i, df.columns.get_loc('Entry_Price')] = entry_price
            elif is_bearish_fractal(df, i, window_size):
                position = -1
                entry_price = df.iloc[i]['Close']
                df.iloc[i, df.columns.get_loc('Signal')] = -1
                df.iloc[i, df.columns.get_loc('Position')] = -1
                df.iloc[i, df.columns.get_loc('Entry_Price')] = entry_price
        else:
            df.iloc[i, df.columns.get_loc('Position')] = position
            if position == 1:
                stop_loss = entry_price * (1 - stop_loss_pct / 100)
                take_profit = entry_price * (1 + take_profit_pct / 100)
                if df.iloc[i]['Low'] <= stop_loss or df.iloc[i]['High'] >= take_profit:
                    position = 0
                    exit_price = stop_loss if df.iloc[i]['Low'] <= stop_loss else take_profit
                    df.iloc[i, df.columns.get_loc('Exit_Price')] = exit_price
                    df.iloc[i, df.columns.get_loc('PnL')] = (exit_price - entry_price) / entry_price
            else:  # position == -1
                stop_loss = entry_price * (1 + stop_loss_pct / 100)
                take_profit = entry_price * (1 - take_profit_pct / 100)
                if df.iloc[i]['High'] >= stop_loss or df.iloc[i]['Low'] <= take_profit:
                    position = 0
                    exit_price = stop_loss if df.iloc[i]['High'] >= stop_loss else take_profit
                    df.iloc[i, df.columns.get_loc('Exit_Price')] = exit_price
                    df.iloc[i, df.columns.get_loc('PnL')] = (entry_price - exit_price) / entry_price
    
    return df

# Run backtest
df = backtest_strategy(df)

# Calculate cumulative returns
df['Cumulative_Returns'] = (1 + df['PnL']).cumprod() - 1

# Create the plot
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=('Price', 'Cumulative Returns'), row_width=[0.7, 0.3])

# Add candlestick chart
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)

# Add buy signals
buy_signals = df[df['Signal'] == 1]
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy Signal'), row=1, col=1)

# Add sell signals
sell_signals = df[df['Signal'] == -1]
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell Signal'), row=1, col=1)

# Add cumulative returns
fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Returns'], mode='lines', name='Cumulative Returns'), row=2, col=1)

# Update layout
fig.update_layout(height=800, title_text="Fractal Trading Strategy Backtest")
fig.update_xaxes(rangeslider_visible=False)

# Show the plot
fig.show()

# Print some statistics
total_trades = len(df[df['Signal'] != 0])
winning_trades = len(df[df['PnL'] > 0])
losing_trades = len(df[df['PnL'] < 0])
win_rate = winning_trades / total_trades if total_trades > 0 else 0
total_return = df['Cumulative_Returns'].iloc[-1]

print(f"Total Trades: {total_trades}")
print(f"Winning Trades: {winning_trades}")
print(f"Losing Trades: {losing_trades}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Total Return: {total_return:.2%}")