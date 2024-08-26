import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas_montecarlo

class IchimokuCloudStrategy:
    def __init__(self, trading_volume=0.05, ema_period=50, conversion_period=9,
                 base_period=26, atr_period=14, sl_atr_multiplier=2.0, tp_atr_multiplier=1.5,
                 atr_scaling_factor=0.1):
        self.trading_volume = trading_volume
        self.ema_period = ema_period
        self.conversion_period = conversion_period
        self.base_period = base_period
        self.atr_period = atr_period
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.atr_scaling_factor = atr_scaling_factor
        
        self.data = None
        self.signals = None

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data.dropna(inplace=True)  # Drop any rows with NaN values

    def calculate_indicators(self):
        self.data['EMA'] = self.data['Close'].ewm(span=self.ema_period, adjust=False).mean()
        self.data['ATR'] = self.calculate_atr(self.atr_period)

    def calculate_atr(self, period):
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        true_range = pd.DataFrame({
            'high_low': high_low,
            'high_close': high_close,
            'low_close': low_close
        }).max(axis=1)
        return true_range.rolling(window=period).mean()

    def calculate_signals(self):
        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['Signal'] = 0
        self.signals['SL'] = 0.0
        self.signals['TP'] = 0.0
        
        conversion_line = self.calculate_conversion_line()
        base_line = self.calculate_base_line()

        buy_condition = (conversion_line > base_line) & (self.data['Close'] > self.data['EMA'])
        sell_condition = (conversion_line < base_line) & (self.data['Close'] < self.data['EMA'])

        self.signals.loc[buy_condition, 'Signal'] = 1
        self.signals.loc[sell_condition, 'Signal'] = -1

        long_signals = self.signals['Signal'] == 1
        short_signals = self.signals['Signal'] == -1

        self.signals.loc[long_signals, 'SL'] = self.data.loc[long_signals, 'Close'] - (self.data.loc[long_signals, 'ATR'] * self.sl_atr_multiplier)
        self.signals.loc[long_signals, 'TP'] = self.data.loc[long_signals, 'Close'] + (self.data.loc[long_signals, 'ATR'] * self.tp_atr_multiplier)
        self.signals.loc[short_signals, 'SL'] = self.data.loc[short_signals, 'Close'] + (self.data.loc[short_signals, 'ATR'] * self.sl_atr_multiplier)
        self.signals.loc[short_signals, 'TP'] = self.data.loc[short_signals, 'Close'] - (self.data.loc[short_signals, 'ATR'] * self.tp_atr_multiplier)

    def calculate_conversion_line(self):
        high = self.data['High'].rolling(window=self.conversion_period).max()
        low = self.data['Low'].rolling(window=self.conversion_period).min()
        return (high + low) / 2

    def calculate_base_line(self):
        high = self.data['High'].rolling(window=self.base_period).max()
        low = self.data['Low'].rolling(window=self.base_period).min()
        return (high + low) / 2

    def calculate_returns(self):
        returns = pd.Series(index=self.signals.index)
        position = 0  # Track if we are in a position (1 for long, -1 for short)
        entry_price = 0
        
        for i in range(1, len(self.signals)):
            if position == 0:  # No open position
                if self.signals['Signal'].iloc[i] == 1:  # Buy signal
                    position = 1
                    entry_price = self.data['Close'].iloc[i]
                elif self.signals['Signal'].iloc[i] == -1:  # Sell signal
                    position = -1
                    entry_price = self.data['Close'].iloc[i]
            elif position == 1:  # Long position
                if self.data['Low'].iloc[i] <= self.signals['SL'].iloc[i-1]:  # Stop loss hit
                    returns.iloc[i] = (self.signals['SL'].iloc[i-1] - entry_price) / entry_price
                    position = 0  # Close position
                elif self.data['High'].iloc[i] >= self.signals['TP'].iloc[i-1]:  # Take profit hit
                    returns.iloc[i] = (self.signals['TP'].iloc[i-1] - entry_price) / entry_price
                    position = 0  # Close position
                elif self.signals['Signal'].iloc[i] == -1:  # New sell signal
                    returns.iloc[i] = (self.data['Close'].iloc[i] - entry_price) / entry_price
                    position = -1  # Switch to short position
                    entry_price = self.data['Close'].iloc[i]
            elif position == -1:  # Short position
                if self.data['High'].iloc[i] >= self.signals['SL'].iloc[i-1]:  # Stop loss hit
                    returns.iloc[i] = (entry_price - self.signals['SL'].iloc[i-1]) / entry_price
                    position = 0  # Close position
                elif self.data['Low'].iloc[i] <= self.signals['TP'].iloc[i-1]:  # Take profit hit
                    returns.iloc[i] = (entry_price - self.signals['TP'].iloc[i-1]) / entry_price
                    position = 0  # Close position
                elif self.signals['Signal'].iloc[i] == 1:  # New buy signal
                    returns.iloc[i] = (entry_price - self.data['Close'].iloc[i]) / entry_price
                    position = 1  # Switch to long position
                    entry_price = self.data['Close'].iloc[i]
        
        return returns

if __name__ == "__main__":
    # Load data from local CSV file
    data = pd.read_csv("./nas100_1h.csv")  # Update the path to your CSV file
    
    # Initialize strategy parameters
    strategy = IchimokuCloudStrategy()
    
    # Load data into the strategy
    strategy.load_data("./nas100_1h.csv")  # Update the path to your CSV file
    
    # Calculate indicators and signals
    strategy.calculate_indicators()
    strategy.calculate_signals()

    # Calculate returns
    strategy_returns = strategy.calculate_returns()

    # Ensure strategy_returns does not contain NaN values
    strategy_returns = strategy_returns.dropna()

    print(f"Number of trades: {len(strategy_returns[strategy_returns != 0])}")
    print(f"Total return: {strategy_returns.sum():.2%}")

    # Run simple Monte Carlo simulation using pandas-montecarlo
    if not strategy_returns.empty:
        mc_results = strategy_returns.montecarlo(sims=1000, bust=-0.1, goal=0.5)

        # Print Monte Carlo statistics
        print("\nMonte Carlo Simulation Results:")
        print(f"Mean Return: {mc_results.stats['mean']:.2%}")
        print(f"Median Return: {mc_results.stats['median']:.2%}")
        print(f"Standard Deviation: {mc_results.stats['std']:.2%}")
        print(f"Minimum Return: {mc_results.stats['min']:.2%}")
        print(f"Maximum Return: {mc_results.stats['max']:.2%}")

        # Plot the Monte Carlo simulations
        mc_results.plot(title="Strategy Returns Monte Carlo Simulations")
        plt.show()
    else:
        print("No valid returns to perform Monte Carlo simulation.")