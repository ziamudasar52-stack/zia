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

SCAN_INTERVAL = 45  # seconds
TZ_NY = ZoneInfo("America/New_York")

US_HOLIDAYS = {
    (1, 1),    # New Year's Day
    (7, 4),    # Independence Day
    (12, 25),  # Christmas
}

SEEN_TRADE_IDS = set()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ========= TIME GUARDS =========
def is_trading_time_now() -> bool:
    now = datetime.now(TZ_NY)
    if now.weekday() > 4:
        return False
    if (now.month, now.day) in US_HOLIDAYS:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


# ========= TELEGRAM =========
def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("Telegram credentials missing")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            logging.error(f"Telegram error: {r.status_code} - {r.text}")
        else:
            logging.info("📤 Telegram sent")
    except Exception as e:
        logging.error(f"Telegram exception: {e}")


# ========= MBOUM HELPER =========
def mboum_get(url: str, params: dict | None = None):
    headers = {"Authorization": f"Bearer {MBOUM_API_KEY}"}
    try:
        start = time.time()
        resp = requests.get(url, headers=headers, params=params or {}, timeout=10)
        duration = round(time.time() - start, 2)
        logging.info(f"📡 {url} - {resp.status_code} - {duration}s")
        if resp.status_code != 200:
            logging.warning(f"Mboum status {resp.status_code}: {resp.text}")
            return None
        return resp.json()
    except Exception as e:
        logging.error(f"Mboum request error: {e}")
        return None


# ========= UNUSUAL OPTIONS =========
def fetch_unusual_options_page(page: int = 1):
    url = "https://api.mboum.com/v1/markets/options/unusual-options-activity"
    data = mboum_get(url, {"type": "STOCKS", "page": str(page)})
    if not data:
        return []
    # Adjust to actual schema; assuming {"results": [ {...}, ... ]}
    results = data.get("results", [])
    return results if isinstance(results, list) else []


def classify_flow_signal(trade: dict):
    """
    Map Mboum unusual options trade -> CALL/PUT + direction.
    You MUST adapt field names to actual Mboum response.
    Example assumed keys:
      ticker, optionType, side, premium, strike, expirationDate, delta, id
    """
    ticker = trade.get("ticker")
    if ticker not in WATCHLIST:
        return None

    opt_type = trade.get("optionType")  # "CALL" / "PUT"
    side = trade.get("side")           # "BUY" / "SELL"
    premium = trade.get("premium")
    strike = trade.get("strike")
    expiration = trade.get("expirationDate")
    delta = trade.get("delta")
    trade_id = trade.get("id") or f"{ticker}-{opt_type}-{strike}-{expiration}-{premium}"

    if trade_id in SEEN_TRADE_IDS:
        return None
    SEEN_TRADE_IDS.add(trade_id)

    if opt_type not in ("CALL", "PUT") or side != "BUY":
        return None
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
def fetch_ema_trend(ticker: str):
    """
    Simple trend proxy: price vs EMA(50, 5m).
    """
    # Price proxy from v3 options (you may swap to a pure quote endpoint if available)
    price_url = "https://api.mboum.com/v3/markets/options"
    price_data = mboum_get(price_url, {"ticker": ticker})
    if not price_data:
        return None

    # You MUST adapt this to actual schema.
    last_price = price_data.get("underlyingPrice") or price_data.get("lastPrice")
    if last_price is None:
        return None

    ema_url = "https://api.mboum.com/v1/markets/indicators/ema"
    ema_data = mboum_get(ema_url, {
        "ticker": ticker,
        "interval": "5m",
        "series_type": "close",
        "time_period": "50",
        "limit": "1",
    })
    ema_val = None
    if ema_data and isinstance(ema_data, dict):
        values = ema_data.get("values") or ema_data.get("data")
        if isinstance(values, list) and values:
            ema_val = values[-1].get("ema") or values[-1].get("EMA")

    if ema_val is None:
        return None

    # Trend proxy: price above EMA -> uptrend, below -> downtrend
    if last_price > ema_val:
        return "UP"
    elif last_price < ema_val:
        return "DOWN"
    else:
        return "FLAT"


def fetch_indicators(ticker: str):
    """
    Fetch a few key indicators: RSI, MACD, ADX.
    You MUST adapt to actual Mboum response schema.
    """
    indicators = {
        "rsi": None,
        "macd": None,
        "adx": None,
    }

    # RSI
    rsi_url = "https://api.mboum.com/v1/markets/indicators/rsi"
    rsi_data = mboum_get(rsi_url, {
        "ticker": ticker,
        "interval": "5m",
        "series_type": "close",
        "time_period": "14",
        "limit": "1",
    })
    if rsi_data and isinstance(rsi_data, dict):
        vals = rsi_data.get("values") or rsi_data.get("data")
        if isinstance(vals, list) and vals:
            indicators["rsi"] = vals[-1].get("rsi") or vals[-1].get("RSI")

    # MACD
    macd_url = "https://api.mboum.com/v1/markets/indicators/macd"
    macd_data = mboum_get(macd_url, {
        "ticker": ticker,
        "interval": "5m",
        "series_type": "close",
        "fastperiod": "12",
        "slowperiod": "26",
        "signalperiod": "9",
        "limit": "1",
    })
    if macd_data and isinstance(macd_data, dict):
        vals = macd_data.get("values") or macd_data.get("data")
        if isinstance(vals, list) and vals:
            indicators["macd"] = vals[-1].get("macd") or vals[-1].get("MACD")

    # ADX
    adx_url = "https://api.mboum.com/v1/markets/indicators/adx"
    adx_data = mboum_get(adx_url, {
        "ticker": ticker,
        "interval": "5m",
        "time_period": "14",
        "limit": "1",
    })
    if adx_data and isinstance(adx_data, dict):
        vals = adx_data.get("values") or adx_data.get("data")
        if isinstance(vals, list) and vals:
            indicators["adx"] = vals[-1].get("adx") or vals[-1].get("ADX")

    return indicators


# ========= NEWS SENTIMENT (SIMPLE) =========
def fetch_news_sentiment(ticker: str):
    """
    Very simple sentiment proxy: count positive/negative keywords in recent headlines.
    You MUST adapt to actual Mboum news schema.
    """
    url = "https://api.mboum.com/v2/markets/news"
    data = mboum_get(url, {"ticker": ticker, "type": "ALL"})
    if not data:
        return 0

    # Assume data is a list of articles with "title" or "headline"
    articles = data if isinstance(data, list) else data.get("results", [])
    if not isinstance(articles, list):
        return 0

    positive_words = ["beat", "upgrade", "strong", "record", "growth", "surge"]
    negative_words = ["miss", "downgrade", "lawsuit", "recall", "fraud", "weak"]

    score = 0
    for art in articles[:5]:
        title = (art.get("title") or art.get("headline") or "").lower()
        if not title:
            continue
        if any(w in title for w in positive_words):
            score += 1
        if any(w in title for w in negative_words):
            score -= 1

    # Clamp between -2 and +2
    if score > 2:
        score = 2
    if score < -2:
        score = -2
    return score


# ========= CONFIDENCE FUSION =========
def compute_confidence(flow_sig: dict, trend: str | None, indicators: dict, news_score: int):
    """
    Combine flow + trend + indicators + news into a confidence score.
    """
    direction = flow_sig["direction"]  # BULLISH / BEARISH
    base = 60  # base from flow

    # Flow strength: premium + |delta|
    premium = flow_sig["premium"]
    delta = flow_sig["delta"]
    if premium >= 150000:
        base += 10
    if abs(delta) >= 0.6:
        base += 10

    # Trend alignment
    if trend == "UP" and direction == "BULLISH":
        base += 10
    elif trend == "DOWN" and direction == "BEARISH":
        base += 10
    elif trend in ("UP", "DOWN"):
        base -= 5

    # Indicators
    rsi = indicators.get("rsi")
    macd = indicators.get("macd")
    adx = indicators.get("adx")

    if rsi is not None:
        # Overbought/oversold slight adjustments
        if direction == "BULLISH" and rsi < 35:
            base += 5
        if direction == "BEARISH" and rsi > 65:
            base += 5

    if macd is not None:
        # MACD sign alignment
        if direction == "BULLISH" and macd > 0:
            base += 5
        if direction == "BEARISH" and macd < 0:
            base += 5

    if adx is not None and adx >= 20:
        base += 5  # stronger trend

    # News
    if news_score > 0 and direction == "BULLISH":
        base += 5
    if news_score < 0 and direction == "BEARISH":
        base += 5

    # Clamp
    if base < 10:
        base = 10
    if base > 99:
        base = 99

    return base


# ========= MESSAGE FORMAT =========
def format_signal_message(flow_sig: dict, trend: str | None, indicators: dict, news_score: int, confidence: int) -> str:
    ticker = flow_sig["ticker"]
    direction = flow_sig["direction"]
    opt_type = flow_sig["type"]
    premium = flow_sig["premium"]
    strike = flow_sig["strike"]
    expiration = flow_sig["expiration"]
    delta = flow_sig["delta"]

    emoji = "🟢" if direction == "BULLISH" else "🔴"
    conf_tag = "🔥 HIGH CONFIDENCE" if confidence >= 70 else "⚠️ Low/Medium Confidence"

    trend_str = trend or "UNKNOWN"
    rsi = indicators.get("rsi")
    macd = indicators.get("macd")
    adx = indicators.get("adx")

    news_text = "Neutral"
    if news_score > 0:
        news_text = "Positive"
    elif news_score < 0:
        news_text = "Negative"

    msg = (
        f"{emoji} *OPTIONS SIGNAL*: {ticker}\n\n"
        f"*Direction:* {direction} ({confidence}% confidence) — {conf_tag}\n"
        f"*Contract:* {opt_type} {strike} exp {expiration}\n"
        f"*Premium:* ${premium:,.0f}  |  Δ: {delta}\n"
        f"*Trend:* {trend_str}\n"
        f"*RSI:* {rsi}  |  *MACD:* {macd}  |  *ADX:* {adx}\n"
        f"*News sentiment:* {news_text}\n"
    )
    return msg


# ========= MAIN LOOP =========
def main():
    if not MBOUM_API_KEY:
        logging.error("Missing MBOUM_API_KEY")
        return

    logging.info("🚀 Combined options signal bot started")

    while True:
        if not is_trading_time_now():
            logging.info("⏸ Outside trading hours or holiday/weekend. Sleeping 60s...")
            time.sleep(60)
            continue

        logging.info("🔍 Scanning unusual options activity...")
        for page in range(1, 3):  # pages 1–2 as example
            trades = fetch_unusual_options_page(page=page)
            if not trades:
                continue

            for trade in trades:
                flow_sig = classify_flow_signal(trade)
                if not flow_sig:
                    continue

                ticker = flow_sig["ticker"]

                # Trend + indicators + news
                trend = fetch_ema_trend(ticker)
                indicators = fetch_indicators(ticker)
                news_score = fetch_news_sentiment(ticker)

                confidence = compute_confidence(flow_sig, trend, indicators, news_score)
                msg = format_signal_message(flow_sig, trend, indicators, news_score, confidence)

                logging.info(
                    f"Signal: {ticker} {flow_sig['direction']} {flow_sig['type']} "
                    f"prem=${flow_sig['premium']:.0f} Δ={flow_sig['delta']} conf={confidence}"
                )
                send_telegram(msg)

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
