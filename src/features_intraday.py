# src/features_intraday.py

import pandas as pd
import numpy as np
import yfinance as yf

class TechnicalIndicators:
    
    @staticmethod
    def rsi(series, length=14):
        """RSI - Relative Strength Index"""
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        
        gain_ema = pd.Series(gain, index=series.index).ewm(
            alpha=1/length, adjust=False
        ).mean()
        loss_ema = pd.Series(loss, index=series.index).ewm(
            alpha=1/length, adjust=False
        ).mean()
        
        rs = gain_ema / loss_ema
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(series, fast=12, slow=26, signal_period=9):
        """MACD - Moving Average Convergence Divergence"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(series, length=20, std_dev=2):
        """Bollinger Bands"""
        sma = series.rolling(length).mean()
        std = series.rolling(length).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return sma, upper, lower
    
    @staticmethod
    def atr(df, length=14):
        """ATR - Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(length).mean()
    
    @staticmethod
    def stochastic(df, length=14, smooth_k=3, smooth_d=3):
        """Stochastic Oscillator"""
        low_min = df['low'].rolling(length).min()
        high_max = df['high'].rolling(length).max()
        
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        k_smooth = k_percent.rolling(smooth_k).mean()
        d_smooth = k_smooth.rolling(smooth_d).mean()
        
        return k_smooth, d_smooth
    
    @staticmethod
    def add_all_features(df):
        """Add all technical indicators - NO LOOKAHEAD BIAS"""
        df = df.copy()
        
        # Momentum indicators
        df['rsi_14'] = TechnicalIndicators.rsi(df['close'], 14)
        df['rsi_7'] = TechnicalIndicators.rsi(df['close'], 7)
        
        macd_line, macd_signal, macd_hist = TechnicalIndicators.macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # Volatility
        df['atr_14'] = TechnicalIndicators.atr(df, 14)
        sma_20, bb_upper, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
        df['bb_middle'] = sma_20
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / sma_20
        
        # Stochastic
        k_smooth, d_smooth = TechnicalIndicators.stochastic(df, 14, 3, 3)
        df['stoch_k'] = k_smooth
        df['stoch_d'] = d_smooth
        
        # Moving averages
        df['sma_9'] = df['close'].rolling(9).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # Price action
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(10).std()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Drop NaN rows (from indicators)
        df = df.dropna()
        
        return df

if __name__ == "__main__":
    # Test - MultiIndex handling
    print("🧪 Testing features_intraday.py...")
    df = yf.download("RELIANCE.NS", period="5d", interval="15m", progress=False)
    
    # Handle MultiIndex columns properly
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    else:
        df.columns = df.columns.str.lower()
    
    # Select OHLCV columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    
    print(f"📊 Raw data shape: {df.shape}")
    print(f"Latest candle: {df.index[-1]} | Close: ₹{df['close'].iloc[-1]:.2f}")
    
    # Add features
    df_features = TechnicalIndicators.add_all_features(df)
    
    print(f"\n✨ Features added successfully!")
    print(f"Features shape: {df_features.shape}")
    print(f"New feature columns: {len([col for col in df_features.columns if col not in ['open','high','low','close','volume']])}")
    print("\n📈 Last 3 rows:")
    print(df_features[['open','high','low','close','rsi_14','macd','atr_14','bb_width','stoch_k']].tail(3))

