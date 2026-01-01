# src/alerts.py

import requests
import json
from datetime import datetime
import os

class TelegramAlerts:
    
    def __init__(self, bot_token, chat_id):
        """
        Setup Telegram bot
        
        Get bot token: @BotFather
        Get chat ID: @userinfobot
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_message(self, message):
        """Send text message"""
        try:
            url = f"{self.api_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                return True
            else:
                print(f"✗ Failed to send message: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Error sending Telegram message: {e}")
            return False
    
    def send_buy_signal(self, symbol, price, probability, reason):
        """Send BUY signal"""
        message = f"""
🟢 <b>BUY SIGNAL</b>

<b>Stock:</b> {symbol}
<b>Price:</b> ₹{price:.2f}
<b>Confidence:</b> {probability:.1%}

<b>Reason:</b>
{reason}

<b>Targets:</b>
• TP: ₹{price*1.02:.2f} (+2%)
• SL: ₹{price*0.99:.2f} (-1%)

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
        """
        return self.send_message(message)
    
    def send_sell_signal(self, symbol, price, reason):
        """Send SELL/EXIT signal"""
        message = f"""
🔴 <b>SELL SIGNAL / EXIT</b>

<b>Stock:</b> {symbol}
<b>Price:</b> ₹{price:.2f}

<b>Reason:</b>
{reason}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
        """
        return self.send_message(message)
    
    def send_error_alert(self, error_msg):
        """Send error notification"""
        message = f"""
🔺 <b>ERROR ALERT</b>

{error_msg}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
        """
        return self.send_message(message)
    
    def send_daily_summary(self, summary_data):
        """Send daily trading summary"""
        message = f"""
📊 <b>DAILY SUMMARY</b>

<b>Signals Generated:</b> {summary_data.get('signals', 0)}
<b>Wins:</b> {summary_data.get('wins', 0)}
<b>Losses:</b> {summary_data.get('losses', 0)}
<b>Win Rate:</b> {summary_data.get('win_rate', 0):.1%}

<b>P&L:</b> ₹{summary_data.get('pnl', 0):.2f}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
        """
        return self.send_message(message)

def log_trade(filepath, trade_data):
    """Log trade to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        **trade_data
    }
    
    with open(filepath, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

if __name__ == "__main__":
    # Test
    alerts = TelegramAlerts("bot_token", "chat_id")
    
    # Test buy signal
    alerts.send_buy_signal(
        "RELIANCE.NS",
        1575.50,
        0.72,
        "RSI oversold + MACD positive crossover"
    )
    
    # Test error
    alerts.send_error_alert("Connection to market data failed!")

