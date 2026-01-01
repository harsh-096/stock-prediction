# config.py - SINGLE SOURCE OF TRUTH FOR ALL CONFIGURATIONS

import os
from datetime import time as dtime

# ============================================
# 📌 TRADING CONFIGURATION
# ============================================

STOCKS = {
    'RELIANCE.NS': {
        'enabled': True,
        'min_probability': 0.68,
        'take_profit_pct': 0.008,  # 0.8%
        'stop_loss_pct': 0.004,    # 0.4%
        'position_size': 100000,   # ₹100,000 per signal
    },
    'JMFINANCIL.NS': {
        'enabled': True,
        'min_probability': 0.65,
        'take_profit_pct': 0.008,  # 0.8%
        'stop_loss_pct': 0.004,    # 0.4%
        'position_size': 100000,
    },
    'IEX.NS': {
        'enabled': True,
        'min_probability': 0.65,
        'take_profit_pct': 0.005,  # 0.5%
        'stop_loss_pct': 0.004,    # 0.4%
        'position_size': 100000,
    },
    'SWIGGY.NS': {
        'enabled': True,
        'min_probability': 0.65,
        'take_profit_pct': 0.006,  # 0.6%
        'stop_loss_pct': 0.004,    # 0.4%
        'position_size': 100000,
    },
    'HFCL.NS': {
        'enabled': True,           # Set to True to enable
        'min_probability': 0.65,
        'take_profit_pct': 0.005,
        'stop_loss_pct': 0.004,
        'position_size': 100000,
    },
}

LABELING_CONFIG = {
    'take_profit_pct': 0.008,   # 0.8% TP
    'stop_loss_pct': 0.0025,    # 0.25% SL
    'horizon': 4,               # 60 min (4 × 15m candles)
}

# Get list of enabled stocks
ENABLED_STOCKS = [s for s in STOCKS if STOCKS[s]['enabled']]
SYMBOL = ENABLED_STOCKS[0] if ENABLED_STOCKS else 'RELIANCE.NS'

print(f"✓ Config: Monitoring {len(ENABLED_STOCKS)} stocks: {', '.join(ENABLED_STOCKS)}")

# ============================================
# 📊 DATA SETTINGS
# ============================================

TIMEFRAME = "15m"
LOOKBACK_DAYS = 60
HORIZON_CANDLES = 4  # 4 * 15min = 60min forecast horizon

# ============================================
# 🕐 MARKET SETTINGS
# ============================================

MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
TRADING_DAYS = [0, 1, 2, 3, 4]

# ============================================
# 🤖 MODEL SETTINGS
# ============================================

MIN_PROBABILITY = 0.70
MODEL_TYPE = "xgboost"

# ============================================
# 📂 DIRECTORIES
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================
# 📱 TELEGRAM SETTINGS
# ============================================

TELEGRAM_BOT_TOKEN = "Bot_Token"
TELEGRAM_CHAT_ID = "Chat_ID"

ALERT_ON_BUY = True
ALERT_ON_SELL = True
ALERT_ON_ERROR = True
ALERT_ON_SUMMARY = True

# ============================================
# 💰 POSITION MANAGEMENT
# ============================================

MAX_CONCURRENT_POSITIONS = 3
MAX_DAILY_LOSS_LIMIT = -500
DUPLICATE_SIGNAL_TIMEOUT = 1800

# ============================================
# ⚙️ API & PERFORMANCE SETTINGS
# ============================================

FETCH_THROTTLE_SECONDS = 300
CHECK_INTERVAL_SECONDS = 60
RETRAIN_DAYS = 7
LOG_ALL_PREDICTIONS = True

# ============================================
# 🎯 MODEL TUNING PARAMETERS
# ============================================

TP_PERCENT = 0.005
SL_PERCENT = 0.004
HORIZON_MINUTES = 60
SCALE_POS_WEIGHT = 3.0

# ============================================
# 🔒 SAFETY SETTINGS
# ============================================

PAPER_TRADING_MODE = False
DRY_RUN = False
VERBOSE_LOGGING = True

# ============================================
# 📈 BACKTESTING SETTINGS
# ============================================

BACKTEST_DAYS = 30
STARTING_CAPITAL = 100000

print(f"✓ Market hours: {MARKET_OPEN} - {MARKET_CLOSE} IST")
print(f"✓ Telegram alerts: {'ON' if ALERT_ON_BUY else 'OFF'}")
print(f"✓ Paper trading: {'ON' if PAPER_TRADING_MODE else 'OFF'}\n")
