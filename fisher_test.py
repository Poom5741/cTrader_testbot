import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

# Data loading function using yfinance
def load_data():
    try:
        # Download Gold Futures 1-hour data
        data = yf.download('GC=F', interval='1h', period='1y')
        if data.empty:
            raise ValueError("No data found for Gold Futures.")
        data = data[['Open', 'High', 'Low', 'Close']]
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

# Fisher Transform calculation
def calculate_fisher_transform(data, period):
    high = data['High'].rolling(window=period).max()
    low = data['Low'].rolling(window=period).min()
    value = 0.33 * 2 * ((data['Close'] - low) / (high - low) - 0.5)
    fisher = np.log((1 + value) / (1 - value))
    return fisher.cumsum()

# Strategy function
def strategy(data, fisher_period, ema_period, tp, sl):
    if data.empty:
        return 0  # Return zero if data is empty

    data['Fisher'] = calculate_fisher_transform(data, fisher_period)
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
    data['Signal'] = np.where((data['Fisher'] > 0) & (data['Close'] > data['EMA']), 1, 
                              np.where((data['Fisher'] < 0) & (data['Close'] < data['EMA']), -1, 0))
    data['Returns'] = data['Signal'].shift(1) * data['Close'].pct_change()
    data['Cumulative'] = (1 + data['Returns']).cumprod()

    # Apply TP/SL logic
    data['TP'] = np.where(data['Cumulative'] >= tp, tp, np.nan)
    data['SL'] = np.where(data['Cumulative'] <= sl, sl, np.nan)
    
    # Calculate final returns
    data['Final'] = data[['TP', 'SL']].min(axis=1)
    return data['Final'].iloc[-1] if not data['Final'].isnull().all() else data['Cumulative'].iloc[-1]

# Objective function for optimization
def objective(params, data, fisher_period, ema_period):
    tp, sl = params
    return -strategy(data, fisher_period, ema_period, tp, sl)  # Negative for maximization

# Optimization function
def optimize_parameters(data, fisher_period, ema_period):
    initial_guess = [1.1, 0.9]  # Initial TP and SL
    bounds = [(1.01, 2.0), (0.5, 0.99)]  # Bounds for TP and SL
    result = minimize(objective, initial_guess, args=(data, fisher_period, ema_period), bounds=bounds, method='L-BFGS-B')
    return result.x

# Main function
def main():
    data = load_data()
    if data.empty:
        print("Data is not available.")
        return

    fisher_period = 10
    ema_period = 17
    best_params = optimize_parameters(data, fisher_period, ema_period)
    print(f"Optimized TP: {best_params[0]}, SL: {best_params[1]}")

if __name__ == "__main__":
    main()