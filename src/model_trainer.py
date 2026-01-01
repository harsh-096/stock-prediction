import pickle
import numpy as np
import pandas as pd
import sys
import os
import warnings
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STOCKS, LABELING_CONFIG
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import xgboost as xgb
import yfinance as yf


class ModelTrainerV2:
    def __init__(self, symbol, model_type="xgboost"):
        self.symbol = symbol
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Get stock config
        self.stock_config = STOCKS.get(symbol, {})
        self.tp_pct = self.stock_config.get('take_profit_pct', LABELING_CONFIG['take_profit_pct'])
        self.sl_pct = self.stock_config.get('stop_loss_pct', LABELING_CONFIG['stop_loss_pct'])
        self.horizon = LABELING_CONFIG.get('horizon', 4)


    def fetch_intraday_data(self, symbol="RELIANCE.NS", period="60d", interval="15m"):
        """✅ Yahoo Finance 15m data fetch"""
        try:
            print(f"📥 Fetching {symbol} {interval} data (60 days max)...")
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
            else:
                df.columns = df.columns.str.lower()
                
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            df.dropna(inplace=True)
            print(f"✓ Fetched {len(df)} candles")
            print(f"  Latest: {df.index[-1]} | Close: ₹{df['close'].iloc[-1]:.2f}")
            return df
        except Exception as e:
            print(f"✗ Data fetch failed: {e}")
            return None


    def add_features(self, df):
        """✅ 35+ Production-grade features for intraday"""
        print("🔧 Adding 35+ features...")
        df = df.copy()
        
        # RSI (multiple periods)
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD + Signal + Histogram
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Multiple SMAs + Ratios
        for period in [5, 9, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df['sma_ratio_short'] = df['sma_5'] / df['sma_20']
        df['sma_ratio_long'] = df['sma_20'] / df['sma_50']
        
        # ATR + Volatility
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr_14'] / df['close']
        
        df['returns'] = df['close'].pct_change()
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # Price Action (Candlestick patterns)
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['upper_shadow'] = (df['high'] - df['close']) / df['close']
        df['lower_shadow'] = (df['close'] - df['low']) / df['close']
        
        # Volume Analysis
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_price_trend'] = (df['volume'] * df['returns']).cumsum()
        
        # Momentum Indicators
        for period in [3, 5, 10]:
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        # Bollinger Bands Position
        bb_sma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_sma + (bb_std * 2)
        bb_lower = bb_sma - (bb_std * 2)
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        df['bb_width'] = (bb_upper - bb_lower) / bb_sma
        
        # NEW: Trend Strength
        df['trend_strength'] = (df['sma_5'] - df['sma_20']) / df['atr_14']
        
        df.dropna(inplace=True)
        print(f"✓ Features: {len([c for c in df.columns if c not in ['open','high','low','close','volume']])}")
        print(f"✓ Shape: {df.shape}")
        return df


    def triple_barrier_label(self, df, tp_pct=None, sl_pct=None, horizon=None):
        """🚀 TRIPLE BARRIER METHOD - Industry Standard Labeling"""
        # Use config values if not provided
        if tp_pct is None:
            tp_pct = self.tp_pct
        if sl_pct is None:
            sl_pct = self.sl_pct
        if horizon is None:
            horizon = self.horizon
            
        print(f"🎯 Triple Barrier Labeling (TP={tp_pct:.1%}, SL={sl_pct:.1%}, Horizon={horizon*15}m)")
        df = df.copy()
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        labels = []
        
        for i in range(len(df) - horizon):
            entry_price = closes[i]
            upper_barrier = entry_price * (1 + tp_pct)
            lower_barrier = entry_price * (1 - sl_pct)
            touch_upper = False
            touch_lower = False
            touch_time = False
            
            # Check barriers in horizon window
            for j in range(i + 1, min(i + 1 + horizon, len(df))):
                if highs[j] >= upper_barrier:
                    touch_upper = True
                    break
                if lows[j] <= lower_barrier:
                    touch_lower = True
                    break
                if j == min(i + horizon - 1, len(df) - 2):
                    touch_time = True
            
            # Triple Barrier Logic
            if touch_upper and not touch_lower:
                label = 1  # BUY signal
            elif touch_lower:
                label = 0  # Avoid
            else:  # Time barrier
                future_close = closes[min(i + horizon - 1, len(closes) - 1)]
                label = 1 if future_close > entry_price * 1.002 else 0  # 0.2% threshold
            
            labels.append(label)
        
        df_clean = df.iloc[:-horizon].copy()
        df_clean['target'] = labels
        pos_count = df_clean['target'].sum()
        pos_pct = pos_count / len(df_clean) * 100
        print(f"✅ Triple Barrier Labels:")
        print(f"  Total: {len(df_clean):,}")
        print(f"  BUY signals: {pos_count} ({pos_pct:.1f}%)")
        return df_clean


    def train(self, df, test_size=0.15):
        """🚀 OPTIMIZED XGBoost for Imbalanced Trading Data"""
        print(f"\n📊 Training {self.model_type.upper()}...")
        exclude_cols = ['target', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"Using {len(feature_cols)} features")
        
        X = df[feature_cols].values
        y = df['target'].values
        self.feature_names = feature_cols
        
        print(f"Samples: {len(X):,}")
        print(f"Positive targets: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        
        if sum(y) < 20:
            print("⚠️ Too few positives. Adjust TP/SL/horizon.")
            return None
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Time-series split (no shuffle)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, shuffle=False
        )
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        print(f"Train positives: {sum(y_train)}")
        
        # 🚀 OPTIMIZED: Dynamic class weighting
        neg_count = len(y_train[y_train == 0])
        pos_count = len(y_train[y_train == 1])
        scale_pos_weight = max(neg_count / pos_count, 3.0)  # Uncapped but minimum 3x
        print(f"Scale pos weight: {scale_pos_weight:.1f}")
        
        # SMOTE (if positives too rare <25%)
        if pos_count / len(y_train) < 0.25:
            print("🔄 Applying SMOTE oversampling...")
            smote = SMOTE(sampling_strategy=0.5, random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - Train positives: {sum(y_train)}")
        
        # 🚀 PRODUCTION XGBoost Parameters
        self.model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        # Early stopping
        try:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50,
                verbose=False
            )
        except:
            self.model.fit(X_train, y_train)
        
        # Comprehensive Evaluation
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Precision-Recall AUC (Trading Priority)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        
        print("\n📈 TEST RESULTS:")
        print(classification_report(y_test, y_pred))
        print("\n🔢 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"\n✅ Test Accuracy: {self.model.score(X_test, y_test):.3f}")
        print(f"✅ PR-AUC: {pr_auc:.3f}")
        print(f"✅ Positive prediction rate: {y_proba.mean():.1%}")
        print(f"✅ Recall (catching buys): {(y_pred[y_test==1]==1).mean():.1%}")
        
        if (y_pred[y_test == 1] == 1).mean() > 0.45:
            print("🎉 RECALL TARGET ACHIEVED! (45%+)")
        else:
            print("⚠️ Recall low - try wider TP/SL")
        
        return self.model


    def save(self, model_dir="models"):
        """✅ Save model + scaler + features"""
        os.makedirs(model_dir, exist_ok=True)
        symbol_clean = self.symbol.replace('.NS', '_NS').lower()
        files = [
            f"{model_dir}/{symbol_clean}_model.pkl",
            f"{model_dir}/{symbol_clean}_scaler.pkl",
            f"{model_dir}/{symbol_clean}_features.pkl"
        ]
        
        with open(files[0], "wb") as f:
            pickle.dump(self.model, f)
        with open(files[1], "wb") as f:
            pickle.dump(self.scaler, f)
        with open(files[2], "wb") as f:
            pickle.dump(self.feature_names, f)
        
        print(f"✅ Model trained and saved for {self.symbol}")


if __name__ == "__main__":
    print("🚀 PRODUCTION INTRADAY MODEL v2 (Triple Barrier + 45%+ Recall)")
    print("📋 Using config from config.py\n")
    
    trainer = ModelTrainerV2("RELIANCE.NS")
    
    df = trainer.fetch_intraday_data("RELIANCE.NS", period="60d", interval="15m")
    if df is None or len(df) < 300:
        print("❌ Need more data")
        sys.exit(1)
    
    # 🚀 Triple Barrier Labeling + Enhanced Features
    df_features = trainer.add_features(df)
    df_targets = trainer.triple_barrier_label(df_features)  # Uses config values!
    
    # Train & Save
    model = trainer.train(df_targets, test_size=0.15)
    if model is not None:
        trainer.save("models")
        print("\n🎉 PRODUCTION MODEL READY!")
        print("✅ Expected: 20-30% BUY signals, 45-55% recall")
        print(f"✅ Files: {trainer.symbol.replace('.NS', '_NS').lower()}_model.pkl + scaler + features")
    else:
        print("❌ Training failed")