import pandas as pd
import optuna
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('WTI_prices.csv')

# Ensure the 'Datetime' column is of datetime type
df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)

# Function to calculate fractals
def calculate_fractals(df, window_size):
    df['Bearish_Fractal'] = 0
    df['Bullish_Fractal'] = 0

    for i in range(window_size, len(df) - window_size):
        if (df['High'].iloc[i] > df['High'].iloc[i-window_size] and 
            df['High'].iloc[i] > df['High'].iloc[i-window_size+1] and 
            df['High'].iloc[i] > df['High'].iloc[i+1] and 
            df['High'].iloc[i] > df['High'].iloc[i+window_size]):
            df['Bearish_Fractal'].iloc[i] = df['High'].iloc[i]

        if (df['Low'].iloc[i] < df['Low'].iloc[i-window_size] and 
            df['Low'].iloc[i] < df['Low'].iloc[i-window_size+1] and 
            df['Low'].iloc[i] < df['Low'].iloc[i+1] and 
            df['Low'].iloc[i] < df['Low'].iloc[i+window_size]):
            df['Bullish_Fractal'].iloc[i] = df['Low'].iloc[i]

    return df

# Define the backtest function
def backtest_strategy(df, params, initial_balance):
    window_size = params['window_size']
    stop_loss_multiplier = params['stop_loss_multiplier']
    take_profit_multiplier = params['take_profit_multiplier']

    df = calculate_fractals(df, window_size)

    df['Signal'] = 0
    df['Entry_Price'] = 0
    df['Exit_Price'] = 0
    df['Trade_PnL'] = 0

    for i in range(len(df)):
        if df['Bullish_Fractal'].iloc[i] > 0:
            # Buy signal
            df['Signal'].iloc[i] = 1
            df['Entry_Price'].iloc[i] = df['Close'].iloc[i]
        elif df['Bearish_Fractal'].iloc[i] > 0:
            # Sell signal
            df['Signal'].iloc[i] = -1
            df['Exit_Price'].iloc[i] = df['Close'].iloc[i]

    position = 0
    previous_price = 0
    cumulative_pnl = 0

    for i in range(len(df)):
        if df['Signal'].iloc[i] == 1:
            # Enter long position
            position = 1
            previous_price = df['Entry_Price'].iloc[i]
        elif df['Signal'].iloc[i] == -1:
            # Exit long position
            if position == 1:
                df['Trade_PnL'].iloc[i] = df['Exit_Price'].iloc[i] - previous_price
                cumulative_pnl += df['Trade_PnL'].iloc[i]
                position = 0

        # Apply stop loss and take profit
        if position == 1:
            stop_loss = previous_price - (previous_price * stop_loss_multiplier)
            take_profit = previous_price + (previous_price * take_profit_multiplier)
            if df['Low'].iloc[i] <= stop_loss:
                df['Trade_PnL'].iloc[i] = stop_loss - previous_price
                cumulative_pnl += df['Trade_PnL'].iloc[i]
                position = 0
            elif df['High'].iloc[i] >= take_profit:
                df['Trade_PnL'].iloc[i] = take_profit - previous_price
                cumulative_pnl += df['Trade_PnL'].iloc[i]
                position = 0

    return cumulative_pnl / initial_balance * 100

# Define the optimization function
def optimize_strategy(trial):
    params = {
        'window_size': trial.suggest_int('window_size', 2, 10),
        'stop_loss_multiplier': trial.suggest_float('stop_loss_multiplier', 0.01, 0.05),
        'take_profit_multiplier': trial.suggest_float('take_profit_multiplier', 0.01, 0.05)
    }
    return backtest_strategy(df, params, initial_balance=10000)

# Perform optimization
study = optuna.create_study(direction='maximize')
study.optimize(optimize_strategy, n_trials=100)

# Print the best parameters and the corresponding PnL
best_params = study.best_params
best_pnl = study.best_value
print(f"Best Parameters: {best_params}")
print(f"Best PnL (%): {best_pnl:.2f}%")

# Plot the results with the best parameters
best_df = calculate_fractals(df, best_params['window_size'])
best_df['Signal'] = 0
best_df['Entry_Price'] = 0
best_df['Exit_Price'] = 0
best_df['Trade_PnL'] = 0

for i in range(len(best_df)):
    if best_df['Bullish_Fractal'].iloc[i] > 0:
        # Buy signal
        best_df['Signal'].iloc[i] = 1
        best_df['Entry_Price'].iloc[i] = best_df['Close'].iloc[i]
    elif best_df['Bearish_Fractal'].iloc[i] > 0:
        # Sell signal
        best_df['Signal'].iloc[i] = -1
        best_df['Exit_Price'].iloc[i] = best_df['Close'].iloc[i]

position = 0
previous_price = 0
cumulative_pnl = 0

for i in range(len(best_df)):
    if best_df['Signal'].iloc[i] == 1:
        # Enter long position
        position = 1
        previous_price = best_df['Entry_Price'].iloc[i]
    elif best_df['Signal'].iloc[i] == -1:
        # Exit long position
        if position == 1:
            best_df['Trade_PnL'].iloc[i] = best_df['Exit_Price'].iloc[i] - previous_price
            cumulative_pnl += best_df['Trade_PnL'].iloc[i]
            position = 0

    # Apply stop loss and take profit
    if position == 1:
        stop_loss = previous_price - (previous_price * best_params['stop_loss_multiplier'])
        take_profit = previous_price + (previous_price * best_params['take_profit_multiplier'])
        if best_df['Low'].iloc[i] <= stop_loss:
            best_df['Trade_PnL'].iloc[i] = stop_loss - previous_price
            cumulative_pnl += best_df['Trade_PnL'].iloc[i]
            position = 0
        elif best_df['High'].iloc[i] >= take_profit:
            best_df['Trade_PnL'].iloc[i] = take_profit - previous_price
            cumulative_pnl += best_df['Trade_PnL'].iloc[i]
            position = 0

best_df['Cumulative_PnL'] = best_df['Trade_PnL'].cumsum() / 10000 * 100

plt.figure(figsize=(12, 6))
plt.plot(best_df['Datetime'], best_df['Close'], label='Close Price')
plt.plot(best_df['Datetime'], best_df['Cumulative_PnL'], label='Cumulative PnL (%)')
plt.scatter(best_df['Datetime'], best_df['Entry_Price'], color='green', label='Buy Signal')
plt.scatter(best_df['Datetime'], best_df['Exit_Price'], color='red', label='Sell Signal')
plt.xlabel('Datetime')
plt.ylabel('Price')
plt.title('Fractal Trading Strategy Backtest with Optimized Parameters')
plt.legend()
plt.show()