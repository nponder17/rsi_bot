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
MA_EXIT = 20          # Exit when close > MA-20 (v6; was MA-5)

MAX_NEW_BUYS_PER_DAY = 50

# Safety / practical controls
MAX_TOTAL_OPEN_POSITIONS = 150

# --- ML v6 sizing ---
# Quintile notional sizes ($80 / $140 / $200 / $280 / $400)
QUINTILE_SIZE = {1: 80, 2: 140, 3: 200, 4: 280, 5: 400}
NOTIONAL_PER_POSITION = 200.0   # fallback if model not loaded

# Path to the serialized GBR model (produced by train_model_v6.py)
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.getenv("DATA_DIR", "data"), "model_v6.pkl"))

# --- VIX reporting ---
# Rolling 252-day trailing percentile — reported in Telegram but NOT used as a hard gate.
# Research showed the rv_5 cross-sectional filter handles regime selection better.
VIX_LOOKBACK_DAYS = 252
VIX_CSV_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

# --- rv_5 vol filter ---
# Only enter stocks whose 5-day realised vol is above the daily cross-sectional median.
# Backtested: improves avg return per trade and total compounded return significantly.
RV5_FILTER = True

# --- Stop-loss ---
# Sell at next open if a position closes <= entry_price * (1 + STOP_LOSS_PCT/100).
# -12% optimum from sweep: cuts max DD ~9% → ~3.7%, raises Sharpe 0.50 → 0.68,
# leaves CAGR essentially unchanged. Set STOP_LOSS_ENABLED=false in .env to disable.
STOP_LOSS_ENABLED = os.getenv("STOP_LOSS_ENABLED", "true").lower() == "true"
STOP_LOSS_PCT     = float(os.getenv("STOP_LOSS_PCT", "-12.0"))   # negative number

# --- LLM signal gate ---
# Filters buy candidates through Claude before execution.
# SKIP when Claude detects fundamental damage (earnings miss, FDA rejection, etc.)
# Set LLM_GATE_ENABLED=false in .env to disable without code changes.
LLM_GATE_ENABLED = os.getenv("LLM_GATE_ENABLED", "true").lower() == "true"
LLM_GATE_MAX_CANDIDATES = int(os.getenv("LLM_GATE_MAX_CANDIDATES", "20"))  # cap API calls/day

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