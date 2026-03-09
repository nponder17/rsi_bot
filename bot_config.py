import os
from dotenv import load_dotenv

load_dotenv()

# --- Alpaca ---
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_BASE_URL = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")

# --- Telegram ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- Strategy params ---
RSI_ENTRY_THRESHOLD = 5
RSI_PERIOD = 2

MIN_PRICE = 2.0
USE_MA200_FILTER = True

MAX_NEW_BUYS_PER_DAY = 50
NOTIONAL_PER_POSITION = 200.0

# Safety / practical controls
MAX_TOTAL_OPEN_POSITIONS = 150

# Storage
DB_PATH = os.getenv("DB_PATH", "rsi_bot_state.sqlite")
DATA_DIR = os.getenv("DATA_DIR", "data")

# Telegram label
BOT_NAME = os.getenv("BOT_NAME", "RSI Trend Bot")


def require_env():
    missing = []
    for k in ["ALPACA_KEY", "ALPACA_SECRET", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
        if not globals().get(k):
            missing.append(k)
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}")