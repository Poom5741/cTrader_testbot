import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas_montecarlo

def ichimoku_cloud(data, conversion_period, base_period):
    high = data['High']
    low = data['Low']
    conversion_line = (high.rolling(window=conversion_period).max() + low.rolling(window=conversion_period).min()) / 2
    base_line = (high.rolling(window=base_period).max() + low.rolling(window=base_period).min()) / 2
    return pd.DataFrame({
        'Conversion Line': conversion_line,
        'Base Line': base_line,
    })

def calculate_signals(data, ichimoku, ema_period, atr_period, sl_multiplier, tp_multiplier):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0
    signals['SL'] = 0.0
    signals['TP'] = 0.0
    
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
    data['ATR'] = data['High'].sub(data['Low']).rolling(window=atr_period).mean()
    
    buy_condition = (ichimoku['Conversion Line'] > ichimoku['Base Line']) & (data['Close'] > data['EMA'])
    signals.loc[buy_condition, 'Signal'] = 1
    
    sell_condition = (ichimoku['Conversion Line'] < ichimoku['Base Line']) & (data['Close'] < data['EMA'])
    signals.loc[sell_condition, 'Signal'] = -1
    
    long_signals = signals['Signal'] == 1
    short_signals = signals['Signal'] == -1
    
    signals.loc[long_signals, 'SL'] = data.loc[long_signals, 'Close'] - (data.loc[long_signals, 'ATR'] * sl_multiplier)
    signals.loc[long_signals, 'TP'] = data.loc[long_signals, 'Close'] + (data.loc[long_signals, 'ATR'] * tp_multiplier)
    signals.loc[short_signals, 'SL'] = data.loc[short_signals, 'Close'] + (data.loc[short_signals, 'ATR'] * sl_multiplier)
    signals.loc[short_signals, 'TP'] = data.loc[short_signals, 'Close'] - (data.loc[short_signals, 'ATR'] * tp_multiplier)
    
    return signals

def calculate_returns(data, signals):
    returns = pd.Series(index=signals.index)
    position = 0
    entry_price = 0
    
    for i in range(1, len(signals)):
        if position == 0:
            if signals['Signal'].iloc[i] == 1:
                position = 1
                entry_price = data['Close'].iloc[i]
            elif signals['Signal'].iloc[i] == -1:
                position = -1
                entry_price = data['Close'].iloc[i]
        elif position == 1:
            if data['Low'].iloc[i] <= signals['SL'].iloc[i-1]:
                returns.iloc[i] = (signals['SL'].iloc[i-1] - entry_price) / entry_price
                position = 0
            elif data['High'].iloc[i] >= signals['TP'].iloc[i-1]:
                returns.iloc[i] = (signals['TP'].iloc[i-1] - entry_price) / entry_price
                position = 0
            elif signals['Signal'].iloc[i] == -1:
                returns.iloc[i] = (data['Close'].iloc[i] - entry_price) / entry_price
                position = -1
                entry_price = data['Close'].iloc[i]
        elif position == -1:
            if data['High'].iloc[i] >= signals['SL'].iloc[i-1]:
                returns.iloc[i] = (entry_price - signals['SL'].iloc[i-1]) / entry_price
                position = 0
            elif data['Low'].iloc[i] <= signals['TP'].iloc[i-1]:
                returns.iloc[i] = (entry_price - signals['TP'].iloc[i-1]) / entry_price
                position = 0
            elif signals['Signal'].iloc[i] == 1:
                returns.iloc[i] = (entry_price - data['Close'].iloc[i]) / entry_price
                position = 1
                entry_price = data['Close'].iloc[i]
    
    return returns

def objective_function(params, data):
    conversion_period, base_period, ema_period, atr_period, sl_multiplier, tp_multiplier = params
    conversion_period = max(1, int(conversion_period))
    base_period = max(1, int(base_period))
    ema_period = max(1, int(ema_period))
    atr_period = max(1, int(atr_period))
    
    ichimoku = ichimoku_cloud(data, conversion_period, base_period)
    signals = calculate_signals(data, ichimoku, ema_period, atr_period, sl_multiplier, tp_multiplier)
    returns = calculate_returns(data, signals)
    
    if returns.empty or returns.iloc[-1] != returns.iloc[-1]:  # Check for empty Series or NaN
        return 0
    
    return -returns.sum()  # Negative because we want to maximize returns

def optimize_parameters(data):
    initial_params = [9, 26, 50, 14, 2.0, 1.5]
    bounds = [(5, 30), (20, 60), (10, 200), (5, 30), (0.5, 5), (0.5, 5)]
    
    result = minimize(
        objective_function,
        initial_params,
        args=(data,),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    return result.x

if __name__ == "__main__":
    # Load data from local CSV file
    data = pd.read_csv("./gold_1h.csv")
    
    # Optimize parameters
    optimized_params = optimize_parameters(data)
    
    print("Optimized Parameters:")
    print(f"Conversion Period: {int(optimized_params[0])}")
    print(f"Base Period: {int(optimized_params[1])}")
    print(f"EMA Period: {int(optimized_params[2])}")
    print(f"ATR Period: {int(optimized_params[3])}")
    print(f"SL Multiplier: {optimized_params[4]:.2f}")
    print(f"TP Multiplier: {optimized_params[5]:.2f}")
    
    ichimoku = ichimoku_cloud(data, int(optimized_params[0]), int(optimized_params[1]))
    signals = calculate_signals(data, ichimoku, int(optimized_params[2]), int(optimized_params[3]), optimized_params[4], optimized_params[5])
    strategy_returns = calculate_returns(data, signals)
    
    # Ensure strategy_returns does not contain NaN values
    strategy_returns = strategy_returns.dropna()
    
    print(f"Number of trades: {len(strategy_returns[strategy_returns != 0])}")
    print(f"Total return: {strategy_returns.sum():.2%}")
    
    # Run simple Monte Carlo simulation using pandas-montecarlo
    if not strategy_returns.empty:
        mc = strategy_returns.montecarlo(sims=1000, bust=-0.1, goal=1.0)
        
        # Print Monte Carlo statistics
        print("\nMonte Carlo Simulation Results:")
        print(f"Mean Return: {mc.stats['mean']:.2%}")
        print(f"Median Return: {mc.stats['median']:.2%}")
        print(f"Standard Deviation: {mc.stats['std']:.2%}")
        print(f"Minimum Return: {mc.stats['min']:.2%}")
        print(f"Maximum Return: {mc.stats['max']:.2%}")
        
        # Plot the Monte Carlo simulations
        plt.figure(figsize=(10, 6))
        mc.plot(title="Strategy Returns Monte Carlo Simulations")
        plt.show()
    else:
        print("No valid returns to perform Monte Carlo simulation.")