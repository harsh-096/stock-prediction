# src/signal_generator.py

import pandas as pd
from datetime import datetime
import numpy as np

class SignalGenerator:
    
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
    
    def generate_signal(self, df, min_probability=0.65):
        """
        Generate BUY/SELL signal based on latest candle
        
        Returns: {signal, probability, reason}
        """
        if len(df) < 2:
            return {"signal": "WAIT", "probability": 0, "reason": "Not enough data"}
        
        # Get latest candle
        latest = df.iloc[-1]
        
        # Prepare features
        X = df[self.feature_names].iloc[-1:].values
        X_scaled = self.scaler.transform(X)
        
        # Predict
        proba = self.model.predict_proba(X_scaled)[0, 1]
        
        # Generate reason (for alert)
        reason = self._get_signal_reason(latest)
        
        if proba >= min_probability:
            return {
                "signal": "BUY",
                "probability": proba,
                "reason": reason,
                "price": latest['close']
            }
        elif proba <= (1 - min_probability):
            return {
                "signal": "SELL",
                "probability": 1 - proba,
                "reason": reason,
                "price": latest['close']
            }
        else:
            return {
                "signal": "HOLD",
                "probability": max(proba, 1-proba),
                "reason": "Neutral market conditions",
                "price": latest['close']
            }
    
    def _get_signal_reason(self, candle):
        """Generate human-readable reason for signal"""
        reasons = []
        
        # RSI signals
        if 'rsi_14' in candle:
            rsi = candle['rsi_14']
            if rsi > 70:
                reasons.append("RSI overbought (>70)")
            elif rsi < 30:
                reasons.append("RSI oversold (<30)")
        
        # MACD signals
        if 'macd_hist' in candle:
            macd_hist = candle['macd_hist']
            if macd_hist > 0:
                reasons.append("MACD positive histogram")
            else:
                reasons.append("MACD negative histogram")
        
        # Bollinger Bands
        if 'bb_upper' in candle and 'bb_lower' in candle:
            close = candle['close']
            if close > candle['bb_upper']:
                reasons.append("Price above BB upper band")
            elif close < candle['bb_lower']:
                reasons.append("Price below BB lower band")
        
        # Stochastic
        if 'stoch_k' in candle:
            stoch = candle['stoch_k']
            if stoch > 80:
                reasons.append("Stochastic overbought (>80)")
            elif stoch < 20:
                reasons.append("Stochastic oversold (<20)")
        
        if not reasons:
            reasons.append("Model consensus signal")
        
        return " | ".join(reasons[:2])

if __name__ == "__main__":
    print("Signal generator module ready")
