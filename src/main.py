# src/main.py - MULTI-STOCK TRADING BOT WITH CENTRALIZED CONFIG

import schedule
import time
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import pickle
import pytz
import traceback
from collections import defaultdict
import joblib

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config
from config import (
    STOCKS, ENABLED_STOCKS, SYMBOL, TIMEFRAME, LOOKBACK_DAYS,
    MARKET_OPEN, MARKET_CLOSE, TRADING_DAYS,
    MODELS_DIR, DATA_DIR, LOGS_DIR,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    ALERT_ON_BUY, ALERT_ON_ERROR,
    FETCH_THROTTLE_SECONDS, CHECK_INTERVAL_SECONDS,
    DUPLICATE_SIGNAL_TIMEOUT, MAX_CONCURRENT_POSITIONS,
    PAPER_TRADING_MODE, DRY_RUN, VERBOSE_LOGGING
)

# Import modules
from src.data_fetcher import DataFetcher
from src.model_trainer import ModelTrainerV2
from src.alerts import TelegramAlerts, log_trade


class MultiStockTradingBot:
    """Production-ready trading bot for multiple stocks"""

    def __init__(self):
        print("\n" + "="*60)
        print("🤖 MULTI-STOCK TRADING BOT - INITIALIZATION")
        print("="*60)

        # Telegram alerts
        try:
            self.alerts = TelegramAlerts(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            print("✓ Telegram alerts enabled")
        except Exception as e:
            print(f"⚠️ Telegram disabled: {e}")
            self.alerts = None

        # Directories
        for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
            os.makedirs(d, exist_ok=True)

        # State
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.last_fetch_time = {}
        self.last_signal_time = {}
        self.active_positions = defaultdict(list)

        # Load/train models
        print(f"\n📦 Loading models for {len(ENABLED_STOCKS)} stocks...\n")
        for symbol in ENABLED_STOCKS:
            try:
                self._load_or_train_model(symbol)
                self.last_fetch_time[symbol] = 0
            except Exception as e:
                print(f"❌ Failed to initialize {symbol}: {e}")

        print("\n" + "="*60)
        print(f"✅ Bot ready! Monitoring: {', '.join(ENABLED_STOCKS)}")
        print(f"📊 Market hours: {MARKET_OPEN} - {MARKET_CLOSE} IST (Mon-Fri)")
        print(f"⚙️  Checking every {CHECK_INTERVAL_SECONDS} seconds")
        if PAPER_TRADING_MODE:
            print("⚠️ PAPER TRADING MODE - NO REAL TRADES")
        print("="*60 + "\n")

    # ---------- Model load/train ----------

    def _load_or_train_model(self, symbol):
        model_name = symbol.replace('.NS', '_NS').lower()
        model_file = os.path.join(MODELS_DIR, f"{model_name}_model.pkl")
        scaler_file = os.path.join(MODELS_DIR, f"{model_name}_scaler.pkl")
        features_file = os.path.join(MODELS_DIR, f"{model_name}_features.pkl")

        # Try load existing model
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            try:
                self.models[symbol] = pickle.load(open(model_file, "rb"))
                self.scalers[symbol] = pickle.load(open(scaler_file, "rb"))
                if os.path.exists(features_file):
                    self.feature_names[symbol] = pickle.load(open(features_file, "rb"))
                print(f" ✓ Loaded existing model for {symbol}")
                return
            except Exception as e:
                print(f" ⚠️ Failed to load model for {symbol}: {e}")

        # Train new
        print(f" 🚀 Training new model for {symbol}...")
        self._train_model_for_stock(symbol)

    def _train_model_for_stock(self, symbol):
        try:
            trainer = ModelTrainerV2(symbol, model_type="xgboost")

            # Fetch raw data
            df = trainer.fetch_intraday_data(symbol=symbol, period="60d", interval=TIMEFRAME)
            if df is None or len(df) < 100:
                print(f" ❌ Insufficient data for {symbol}")
                return False

            # Features (same as training code)
            df_features = trainer.add_features(df)

            # Triple barrier labels (uses LABELING_CONFIG / stock config)
            df_labeled = trainer.triple_barrier_label(df_features)

            # Train model
            trainer.train(df_labeled)

            model_name = symbol.replace('.NS', '_NS').lower()
            pickle.dump(trainer.model, open(os.path.join(MODELS_DIR, f"{model_name}_model.pkl"), "wb"))
            pickle.dump(trainer.scaler, open(os.path.join(MODELS_DIR, f"{model_name}_scaler.pkl"), "wb"))
            pickle.dump(trainer.feature_names, open(os.path.join(MODELS_DIR, f"{model_name}_features.pkl"), "wb"))

            self.models[symbol] = trainer.model
            self.scalers[symbol] = trainer.scaler
            self.feature_names[symbol] = trainer.feature_names

            print(f" ✓ Model trained and saved for {symbol}")
            return True
        except Exception as e:
            print(f" ❌ Training failed for {symbol}: {e}")
            if VERBOSE_LOGGING:
                traceback.print_exc()
            return False

    # ---------- Main loop ----------

    def check_all_signals(self):
        try:
            ist = pytz.timezone('Asia/Kolkata')
            now = datetime.now(ist)

            market_open = now.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute,
                                      second=0, microsecond=0)
            market_close = now.replace(hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute,
                                       second=0, microsecond=0)
            is_market_hours = (now.weekday() in TRADING_DAYS and
                               market_open <= now <= market_close)

            if not is_market_hours:
                return

            print(f"\n⏰ {now.strftime('%Y-%m-%d %H:%M:%S')} - Checking {len(ENABLED_STOCKS)} stocks...")

            signals_generated = 0
            for symbol in ENABLED_STOCKS:
                if symbol not in self.models:
                    print(f" ⏭️ {symbol}: Model not ready")
                    continue

                current_time = time.time()
                if current_time - self.last_fetch_time.get(symbol, 0) < FETCH_THROTTLE_SECONDS:
                    if VERBOSE_LOGGING:
                        print(f" ⏭️ {symbol}: Throttled (wait {FETCH_THROTTLE_SECONDS}s)")
                    continue

                signal = self._check_single_stock(symbol)
                if signal and signal['signal'] == 'BUY':
                    signals_generated += 1

            if signals_generated > 0:
                print(f"\n✅ Generated {signals_generated} BUY signals")

        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            if VERBOSE_LOGGING:
                traceback.print_exc()
            if self.alerts and ALERT_ON_ERROR:
                self.alerts.send_error_alert(f"Bot error: {e}")

    # ---------- Single stock flow ----------

        def _check_single_stock(self, symbol):
        # """Check signal for single stock - returns signal dict or None"""
            try:
                self.last_fetch_time[symbol] = time.time()

            # Fetch data
                fetcher = DataFetcher(symbol, timeframe=TIMEFRAME, lookback_days=5)
                df = fetcher.fetch_data()
                if df is None or len(df) < 30:
                    if VERBOSE_LOGGING:
                        print(f" ⚠️ {symbol}: Insufficient raw data")
                    return None

            # ✅ Use EXACT SAME feature generation as in training
                from src.model_trainer import ModelTrainerV2
                trainer = ModelTrainerV2(symbol)
                df = trainer.add_features(df)   # <-- 30 training features

                if len(df) < 10:
                    if VERBOSE_LOGGING:
                        print(f" ⚠️ {symbol}: Not enough feature data")
                    return None

            # Match feature order with training
                feature_cols = self.feature_names.get(symbol, [])
                available_features = [f for f in feature_cols if f in df.columns]

                if len(available_features) < len(feature_cols) * 0.9:
                    if VERBOSE_LOGGING:
                        missing = set(feature_cols) - set(available_features)
                        print(f" ⚠️ {symbol}: Missing features: {missing}")
                    return None

                latest = df.iloc[-1:].copy()
                X = latest[available_features].values

            # Handle NaN
                if np.isnan(X).any():
                    if VERBOSE_LOGGING:
                        print(f" ⚠️ {symbol}: NaN in features")
                    return None

            # Predict
                X_scaled = self.scalers[symbol].transform(X)
                proba = self.models[symbol].predict_proba(X_scaled)[:, 1][0]
                price = float(df['close'].iloc[-1])

                stock_config = STOCKS[symbol]
                min_prob = stock_config['min_probability']

            # Determine signal
                signal_type = "HOLD"
                reason = "Neutral market"
                if proba >= min_prob:
                    signal_type = "BUY"
                    reason = f"ML confidence: {proba:.1%}"
                elif proba <= (1 - min_prob):
                    signal_type = "SELL"
                    reason = f"Weak signal: {proba:.1%}"

                status_emoji = "🟢" if signal_type == "BUY" else ("🔴" if signal_type == "SELL" else "⚪")
                print(f" {status_emoji} {symbol}: {signal_type:4} | {proba:5.1%} | ₹{price:8.2f} "
                  f"| {len(available_features)}/{len(feature_cols)} features")

                signal_dict = {
                    'symbol': symbol,
                    'signal': signal_type,
                    'probability': proba,
                    'price': price,
                    'reason': reason,
                    'timestamp': datetime.now(),
                    'config': stock_config,
                }

                if signal_type == "BUY":
                    self._handle_buy_signal(signal_dict)

                return signal_dict

            except Exception as e:
                print(f" ❌ {symbol}: Error - {e}")
                if VERBOSE_LOGGING:
                    traceback.print_exc()
                return None


    # ---------- BUY handling ----------

    def _handle_buy_signal(self, signal):
        symbol = signal['symbol']
        signal_key = f"{symbol}_buy"
        current_time = time.time()

        if signal_key in self.last_signal_time:
            time_since_last = current_time - self.last_signal_time[signal_key]
            if time_since_last < DUPLICATE_SIGNAL_TIMEOUT:
                if VERBOSE_LOGGING:
                    print(f" ⏭️ Duplicate signal suppressed (sent {time_since_last:.0f}s ago)")
                return

        if len(self.active_positions[symbol]) >= MAX_CONCURRENT_POSITIONS:
            print(f" ⏭️ Max positions reached for {symbol}")
            return

        price = signal['price']
        proba = signal['probability']
        config = signal['config']

        tp = price * (1 + config['take_profit_pct'])
        sl = price * (1 - config['stop_loss_pct'])

        print(f" 📊 TP: ₹{tp:.2f} | SL: ₹{sl:.2f} | Size: ₹{config['position_size']}")

        if ALERT_ON_BUY and self.alerts and not DRY_RUN:
            try:
                self.alerts.send_buy_signal(
                    symbol=symbol,
                    price=price,
                    probability=proba,
                    reason=signal['reason']
                )
            except Exception as e:
                print(f" ⚠️ Failed to send alert: {e}")

        try:
            log_trade(os.path.join(LOGS_DIR, "trades.jsonl"), {
                'symbol': symbol,
                'signal': 'BUY',
                'price': float(price),
                'probability': float(proba),
                'tp': float(tp),
                'sl': float(sl),
                'position_size': config['position_size'],
                'timestamp': signal['timestamp'].isoformat(),
                'reason': signal['reason'],
            })
        except Exception as e:
            print(f" ⚠️ Failed to log trade: {e}")

        self.active_positions[symbol].append({
            'price': price,
            'tp': tp,
            'sl': sl,
            'time': current_time,
        })
        self.last_signal_time[signal_key] = current_time
        print(" ✓ Signal recorded and alert sent")


def main():
    bot = MultiStockTradingBot()
    schedule.every(CHECK_INTERVAL_SECONDS).seconds.do(bot.check_all_signals)

    print("🚀 Bot started successfully!")
    print("📋 Press Ctrl+C to stop\n")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Bot stopped by user")
        print("="*60)


if __name__ == "__main__":
    main()
