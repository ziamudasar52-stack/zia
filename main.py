import os
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import requests

# ========= CONFIG =========
WATCHLIST = [
    "SLV", "GLD", "NFLX", "META", "AAPL", "TSLA", "NVDA",
    "GOOG", "MSFT", "AMZN", "SPY", "SPX", "AMD", "PLTR",
    "QQQ", "ORCL", "IBM", "ABNB"
]

MBOUM_API_KEY = os.getenv("MBOUM_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SCAN_INTERVAL = 45
TZ_NY = ZoneInfo("America/New_York")

US_HOLIDAYS = {(1, 1), (7, 4), (12, 25)}
SEEN_TRADE_IDS = set()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ========= HELPERS =========
def clean_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = (
            value.replace("$", "")
                 .replace(",", "")
                 .replace("%", "")
                 .strip()
        )
        try:
            return float(cleaned)
        except:
            return None
    return None

def is_trading_time_now():
    now = datetime.now(TZ_NY)
    if now.weekday() > 4:
        return False
    if (now.month, now.day) in US_HOLIDAYS:
        return False
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=30, second=0, microsecond=0)
    return open_t <= now <= close_t

def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("Telegram credentials missing")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=10)
        logging.info("📤 Telegram sent")
    except Exception as e:
        logging.error(f"Telegram error: {e}")

def mboum_get(url, params=None):
    headers = {"Authorization": f"Bearer {MBOUM_API_KEY}"}
    try:
        resp = requests.get(url, headers=headers, params=params or {}, timeout=10)
        logging.info(f"📡 {url} - {resp.status_code}")
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception as e:
        logging.error(f"Mboum error: {e}")
        return None

# ========= UNUSUAL OPTIONS =========
def fetch_unusual_options_page(page=1):
    url = "https://api.mboum.com/v1/markets/options/unusual-options-activity"
    data = mboum_get(url, {"type": "STOCKS", "page": str(page)})
    if not data:
        return []
    return data.get("results", [])

def classify_flow_signal(trade):
    ticker = trade.get("ticker")
    if ticker not in WATCHLIST:
        return None

    opt_type_raw = trade.get("symbolType") or trade.get("optionType")
    side_raw = trade.get("side")
    premium_raw = trade.get("premium")
    strike_raw = trade.get("strikePrice") or trade.get("strike")
    expiration = trade.get("expiration") or trade.get("expirationDate")
    delta_raw = trade.get("delta")
    trade_id = trade.get("id") or f"{ticker}-{opt_type_raw}-{strike_raw}-{expiration}-{premium_raw}"

    if trade_id in SEEN_TRADE_IDS:
        return None
    SEEN_TRADE_IDS.add(trade_id)

    opt_type = (opt_type_raw or "").upper()
    side = (side_raw or "").upper()

    if opt_type not in ("CALL", "PUT") or side != "BUY":
        return None

    premium = clean_number(premium_raw)
    strike = clean_number(strike_raw)
    delta = clean_number(delta_raw)

    if premium is None or premium < 50000:
        return None
    if delta is None:
        return None

    direction = "BULLISH" if opt_type == "CALL" else "BEARISH"

    return {
        "ticker": ticker,
        "direction": direction,
        "type": opt_type,
        "premium": premium,
        "strike": strike,
        "expiration": expiration,
        "delta": delta,
    }

# ========= TREND / INDICATORS =========
def fetch_ema_trend(ticker):
    price_data = mboum_get("https://api.mboum.com/v3/markets/options", {"ticker": ticker})
    if not price_data:
        return None

    last_price = clean_number(price_data.get("underlyingPrice") or price_data.get("lastPrice"))
    if last_price is None:
        return None

    ema_data = mboum_get("https://api.mboum.com/v1/markets/indicators/ema", {
        "ticker": ticker,
        "interval": "5m",
        "series_type": "close",
        "time_period": "50",
        "limit": "1",
    })

    ema_val = None
    if ema_data:
        vals = ema_data.get("values") or ema_data.get("data")
        if vals:
            ema_val = clean_number(vals[-1].get("ema") or vals[-1].get("EMA"))

    if ema_val is None:
        return None

    if last_price > ema_val:
        return "UP"
    elif last_price < ema_val:
        return "DOWN"
    return "FLAT"

def fetch_indicators(ticker):
    out = {"rsi": None, "macd": None, "adx": None}

    # RSI
    rsi_data = mboum_get("https://api.mboum.com/v1/markets/indicators/rsi", {
        "ticker": ticker, "interval": "5m", "series_type": "close", "time_period": "14", "limit": "1"
    })
    if rsi_data:
        vals = rsi_data.get("values") or rsi_data.get("data")
        if vals:
            out["rsi"] = clean_number(vals[-1].get("rsi") or vals[-1].get("RSI"))

    # MACD
    macd_data = mboum_get("https://api.mboum.com/v1/markets/indicators/macd", {
        "ticker": ticker, "interval": "5m", "series_type": "close",
        "fastperiod": "12", "slowperiod": "26", "signalperiod": "9", "limit": "1"
    })
    if macd_data:
        vals = macd_data.get("values") or macd_data.get("data")
        if vals:
            out["macd"] = clean_number(vals[-1].get("macd") or vals[-1].get("MACD"))

    # ADX
    adx_data = mboum_get("https://api.mboum.com/v1/markets/indicators/adx", {
        "ticker": ticker, "interval": "5m", "time_period": "14", "limit": "1"
    })
    if adx_data:
        vals = adx_data.get("values") or adx_data.get("data")
        if vals:
            out["adx"] = clean_number(vals[-1].get("adx") or vals[-1].get("ADX"))

    return out

# ========= NEWS SENTIMENT =========
def fetch_news_sentiment(ticker):
    data = mboum_get("https://api.mboum.com/v2/markets/news", {"ticker": ticker, "type": "ALL"})
    if not data:
        return 0

    articles = data if isinstance(data, list) else data.get("results", [])
    if not isinstance(articles, list):
        return 0

    pos = ["beat", "upgrade", "strong", "record", "growth", "surge"]
    neg = ["miss", "downgrade", "lawsuit", "recall", "fraud", "weak"]

    score = 0
    for art in articles[:5]:
        title = (art.get("title") or art.get("headline") or "").lower()
        if any(w in title for w in pos):
            score += 1
        if any(w in title for w in neg):
            score -= 1

    return max(-2, min(2, score))

# ========= CONFIDENCE =========
def compute_confidence(flow, trend, ind, news):
    direction = flow["direction"]
    base = 60

    if flow["premium"] >= 150000:
        base += 10
    if abs(flow["delta"]) >= 0.6:
        base += 10

    if trend == "UP" and direction == "BULLISH":
        base += 10
    if trend == "DOWN" and direction == "BEARISH":
        base += 10

    rsi, macd, adx = ind["rsi"], ind["macd"], ind["adx"]

    if rsi is not None:
        if direction == "BULLISH" and rsi < 35:
            base += 5
        if direction == "BEARISH" and rsi > 65:
            base += 5

    if macd is not None:
        if direction == "BULLISH" and macd > 0:
            base += 5
        if direction == "BEARISH" and macd < 0:
            base += 5

    if adx is not None and adx >= 20:
        base += 5

    if news > 0 and direction == "BULLISH":
        base += 5
    if news < 0 and direction == "BEARISH":
        base += 5

    return max(10, min(99, base))

# ========= CATEGORY TAGGING =========
def classify_signal_type(flow, trend, confidence):
    direction = flow["direction"]

    # Default
    tag = "[ALL SIGNAL] 🔹"

    # High confidence
    if confidence >= 70:
        tag = "[HIGH CONFIDENCE] 🔥"

    # SuperTrend + Flow agreement
    if (direction == "BULLISH" and trend == "UP") or \
       (direction == "BEARISH" and trend == "DOWN"):
        tag = "[SUPER-TREND + FLOW AGREEMENT] 💎"

    return tag

# ========= MESSAGE FORMAT =========
def format_signal(flow, trend, ind, news, conf, tag):
    emoji = "🟢" if flow["direction"] == "BULLISH" else "🔴"

    news_text = "Neutral"
    if news > 0:
        news_text = "Positive"
    elif news < 0:
        news_text = "Negative"

    return (
        f"{emoji} *OPTIONS SIGNAL*: {flow['ticker']}\n"
        f"{tag}\n\n"
        f"*Direction:* {flow['direction']} ({conf}% confidence)\n"
        f"*Contract:* {flow['type']} {flow['strike']} exp {flow['expiration']}\n"
        f"*Premium:* ${flow['premium']:,.0f} | Δ {flow['delta']}\n"
        f"*Trend:* {trend}\n"
        f"*RSI:* {ind['rsi']} | *MACD:* {ind['macd']} | *ADX:* {ind['adx']}\n"
        f"*News sentiment:* {news_text}\n"
    )

# ========= MAIN LOOP =========
def main():
    logging.info("🚀 Combined options signal bot started")

    while True:
        if not is_trading_time_now():
            logging.info("⏸ Outside trading hours. Sleeping 60s...")
            time.sleep(60)
            continue

        for page in range(1, 3):
            trades = fetch_unusual_options_page(page)
            for trade in trades:
                flow = classify_flow_signal(trade)
                if not flow:
                    continue

                ticker = flow["ticker"]
                trend = fetch_ema_trend(ticker)
                ind = fetch_indicators(ticker)
                news = fetch_news_sentiment(ticker)
                conf = compute_confidence(flow, trend, ind, news)
                tag = classify_signal_type(flow, trend, conf)

                msg = format_signal(flow, trend, ind, news, conf, tag)
                send_telegram(msg)

        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()
