import pandas as pd
import numpy as np
import yfinance as yf
import optuna

def fetch_data(symbol, start_date, end_date, interval):
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    return data

def ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def backtest(data, short_period, medium_period, long_period, tp_percent, sl_percent):
    data['EMA_Short'] = ema(data['Close'], short_period)
    data['EMA_Medium'] = ema(data['Close'], medium_period)
    data['EMA_Long'] = ema(data['Close'], long_period)
    
    position = 0
    entry_price = 0
    trades = []
    
    for i in range(len(data)):
        if position == 0:
            if data['EMA_Short'][i] > data['EMA_Medium'][i] and data['EMA_Long'][i] > data['Close'][i-1]:
                position = 1
                entry_price = data['Close'][i]
            elif data['EMA_Short'][i] < data['EMA_Medium'][i] and data['EMA_Long'][i] < data['Close'][i-1]:
                position = -1
                entry_price = data['Close'][i]
        else:
            if position == 1:
                stop_loss = entry_price * (1 - sl_percent)
                take_profit = entry_price * (1 + tp_percent)
                if data['Low'][i] <= stop_loss or data['High'][i] >= take_profit:
                    exit_price = stop_loss if data['Low'][i] <= stop_loss else take_profit
                    trades.append((entry_price, exit_price))
                    position = 0
            else:  # position == -1
                stop_loss = entry_price * (1 + sl_percent)
                take_profit = entry_price * (1 - tp_percent)
                if data['High'][i] >= stop_loss or data['Low'][i] <= take_profit:
                    exit_price = stop_loss if data['High'][i] >= stop_loss else take_profit
                    trades.append((entry_price, exit_price))
                    position = 0
    
    if trades:
        trades_df = pd.DataFrame(trades, columns=['Entry', 'Exit'])
        trades_df['PnL'] = trades_df['Exit'] - trades_df['Entry']
        total_pnl = trades_df['PnL'].sum()
        win_rate = (trades_df['PnL'] > 0).mean()
        return total_pnl, win_rate
    else:
        return 0, 0

def optimize(trial):
    short_period = trial.suggest_int('short_period', 5, 20)
    medium_period = trial.suggest_int('medium_period', 20, 50)
    long_period = trial.suggest_int('long_period', 50, 200)
    tp_percent = trial.suggest_float('tp_percent', 0.01, 0.1)
    sl_percent = trial.suggest_float('sl_percent', 0.01, 0.1)
    
    total_pnl, win_rate = backtest(data, short_period, medium_period, long_period, tp_percent, sl_percent)
    
    return total_pnl

# Fetch historical data for US30
symbol = '^DJI'  # Yahoo Finance symbol for the Dow Jones Industrial Average
start_date = '2010-01-01'
end_date = '2023-06-08'
interval = '1h'

data = fetch_data(symbol, start_date, end_date, interval)

# Optimize
study = optuna.create_study(direction='maximize')
study.optimize(optimize, n_trials=100)

# Print the best parameters and results
print("Best parameters:")
print(f"Short Period: {study.best_params['short_period']}")
print(f"Medium Period: {study.best_params['medium_period']}")
print(f"Long Period: {study.best_params['long_period']}")
print(f"Take Profit (%): {study.best_params['tp_percent']:.2%}")
print(f"Stop Loss (%): {study.best_params['sl_percent']:.2%}")
print(f"Best PnL: {study.best_value:.2f}")