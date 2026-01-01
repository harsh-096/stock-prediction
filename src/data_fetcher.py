# src/data_fetcher.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class DataFetcher:
    def __init__(self, symbol, timeframe="15m", lookback_days=60):
        """
        Fetch intraday OHLCV data
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE.NS")
            timeframe: "5m", "15m", "30m", "1h"
            lookback_days: How many days of historical data
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_days = lookback_days
    
    def fetch_data(self):
        """Download intraday data from yfinance - FIXED"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            print(f"📥 Fetching {self.symbol} ({self.timeframe}) from {start_date.date()} to {end_date.date()}")
            
            df = yf.download(
                self.symbol,
                start=start_date,
                end=end_date,
                interval=self.timeframe,
                progress=False,
                auto_adjust=True,  
                prepost=False,
                threads=True
            )
            
            if df.empty:
                raise ValueError(f"No data fetched for {self.symbol}")
            
            # Handle MultiIndex columns properly
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            # Rename columns to lowercase
            df.columns = df.columns.str.lower()
            
            # Select only OHLCV columns
            available_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in available_cols if col in df.columns]]
            
            # Reset index to make datetime a column
            df = df.reset_index()
            df.rename(columns={'Datetime': 'date', 'Date': 'date'}, inplace=True)
            
            print(f"✓ Fetched {len(df)} candles")
            print(f"  Latest: {df['date'].iloc[-1]} | Close: ₹{df['close'].iloc[-1]:.2f}")
            
            return df
            
        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            return None
    
    def save_data(self, df, filepath):
        """Save data to CSV"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✓ Data saved to {filepath}")

if __name__ == "__main__":
    fetcher = DataFetcher("RELIANCE.NS", timeframe="15m", lookback_days=30)
    df = fetcher.fetch_data()
    if df is not None:
        fetcher.save_data(df, "data/RELIANCE_15m.csv")
        print("\nFirst 3 rows:")
        print(df.head(3))

