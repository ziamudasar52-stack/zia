import os
import time
import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from math import fabs
import requests

# ========= CONFIG =========
WATCHLIST = [
    "SLV", "GLD", "NFLX", "META", "AAPL", "TSLA", "NVDA",
    "GOOG", "MSFT", "AMZN", "SPY", "SPX", "AMD", "PLTR",
    "QQQ", "ORCL", "IBM", "ABNB"
]

# Timeframe options:
# A) 5m  -> "5m"
# B) 1m  -> "1m"
# C) 15m -> "15m"
# D) same as your TV chart -> e.g. "5m", "15m", "1h"
TIMEFRAME = "5m"

MBOUM_API_KEY = os.getenv("MBOUM_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TZ_NY = ZoneInfo("America/New_York")
US_HOLIDAYS = {(1, 1), (7, 4), (12, 25)}

SCAN_INTERVAL = 60  # seconds between full watchlist scans during market hours

# Startup/close schedule (NY time)
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 30)

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
        except Exception:
            return None
    return None


def now_ny():
    return datetime.now(TZ_NY)


def is_us_holiday(dt: datetime):
    return (dt.month, dt.day) in US_HOLIDAYS


def is_market_open(dt: datetime):
    if dt.weekday() > 4:
        return False
    if is_us_holiday(dt):
        return False
    t = dt.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("Telegram credentials missing")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
        if r.status_code != 200:
            logging.error(f"Telegram error: {r.status_code} - {r.text}")
        else:
            logging.info("📤 Telegram sent")
    except Exception as e:
        logging.error(f"Telegram error: {e}")


def mboum_get(url, params=None):
    headers = {"Authorization": f"Bearer {MBOUM_API_KEY}"}
    try:
        resp = requests.get(url, headers=headers, params=params or {}, timeout=15)
        logging.info(f"📡 {url} - {resp.status_code}")
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception as e:
        logging.error(f"Mboum error: {e}")
        return None


# ========= OHLC + SUPERTREND =========
def fetch_ohlc(ticker, interval):
    """
    Adapt this endpoint/fields to Mboum's actual candles API.
    Expecting list of dicts with open/high/low/close.
    """
    url = "https://api.mboum.com/v1/markets/candles"
    data = mboum_get(url, {"ticker": ticker, "interval": interval, "limit": "200"})
    if not data or not isinstance(data, list):
        return []
    ohlc = []
    for c in data:
        o = clean_number(c.get("open"))
        h = clean_number(c.get("high"))
        l = clean_number(c.get("low"))
        cl = clean_number(c.get("close"))
        if None in (o, h, l, cl):
            continue
        ohlc.append((o, h, l, cl))
    return ohlc


def compute_atr(ohlc, period=10):
    if len(ohlc) < period + 1:
        return []
    trs = []
    for i in range(1, len(ohlc)):
        _, h_prev, l_prev, c_prev = ohlc[i - 1]
        _, h_curr, l_curr, c_curr = ohlc[i]
        tr = max(
            h_curr - l_curr,
            fabs(h_curr - c_prev),
            fabs(l_curr - c_prev)
        )
        trs.append(tr)
    atr = []
    first = sum(trs[:period]) / period
    atr.append(first)
    for i in range(period, len(trs)):
        val = (atr[-1] * (period - 1) + trs[i]) / period
        atr.append(val)
    return [None] * (period + 1) + atr  # align with ohlc length


def compute_supertrend_signals(ohlc, period=10, multiplier=3.0):
    """
    Python port of your TradingView SuperTrend logic (simplified).
    Returns:
      trend_dir: "UP" or "DOWN"
      signal: "BUY", "SELL", or None (flip on last bar)
    """
    if len(ohlc) < period + 5:
        return None, None

    atr = compute_atr(ohlc, period)
    n = len(ohlc)
    up = [None] * n
    dn = [None] * n
    trend = [1] * n  # 1 = up, -1 = down

    for i in range(n):
        o, h, l, c = ohlc[i]
        if atr[i] is None:
            continue
        src = (h + l) / 2.0  # hl2
        basic_up = src - multiplier * atr[i]
        basic_dn = src + multiplier * atr[i]

        if i == 0:
            up[i] = basic_up
            dn[i] = basic_dn
            trend[i] = 1
            continue

        up_prev = up[i - 1] if up[i - 1] is not None else basic_up
        dn_prev = dn[i - 1] if dn[i - 1] is not None else basic_dn
        c_prev = ohlc[i - 1][3]

        # up band
        if c_prev > up_prev:
            up[i] = max(basic_up, up_prev)
        else:
            up[i] = basic_up

        # down band
        if c_prev < dn_prev:
            dn[i] = min(basic_dn, dn_prev)
        else:
            dn[i] = basic_dn

        tr_prev = trend[i - 1]
        if tr_prev == -1 and c > dn_prev:
            trend[i] = 1
        elif tr_prev == 1 and c < up_prev:
            trend[i] = -1
        else:
            trend[i] = tr_prev

    last_trend = trend[-1]
    prev_trend = trend[-2]
    signal = None
    if last_trend == 1 and prev_trend == -1:
        signal = "BUY"   # CALL
    elif last_trend == -1 and prev_trend == 1:
        signal = "SELL"  # PUT

    trend_dir = "UP" if last_trend == 1 else "DOWN"
    return trend_dir, signal


# ========= INDICATORS =========
def fetch_indicators(ticker):
    out = {"rsi": None, "macd": None, "adx": None}

    rsi_data = mboum_get("https://api.mboum.com/v1/markets/indicators/rsi", {
        "ticker": ticker, "interval": TIMEFRAME, "series_type": "close",
        "time_period": "14", "limit": "1"
    })
    if rsi_data:
        vals = rsi_data.get("values") or rsi_data.get("data")
        if vals:
            out["rsi"] = clean_number(vals[-1].get("rsi") or vals[-1].get("RSI"))

    macd_data = mboum_get("https://api.mboum.com/v1/markets/indicators/macd", {
        "ticker": ticker, "interval": TIMEFRAME, "series_type": "close",
        "fastperiod": "12", "slowperiod": "26", "signalperiod": "9", "limit": "1"
    })
    if macd_data:
        vals = macd_data.get("values") or macd_data.get("data")
        if vals:
            out["macd"] = clean_number(vals[-1].get("macd") or vals[-1].get("MACD"))

    adx_data = mboum_get("https://api.mboum.com/v1/markets/indicators/adx", {
        "ticker": ticker, "interval": TIMEFRAME, "time_period": "14", "limit": "1"
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


# ========= UNUSUAL OPTIONS (PER TICKER) =========
def fetch_unusual_for_ticker(ticker):
    """
    Mboum's unusual endpoint is global; we filter by ticker.
    """
    url = "https://api.mboum.com/v1/markets/options/unusual-options-activity"
    data = mboum_get(url, {"type": "STOCKS", "page": "1"})
    if not data:
        return []
    results = data.get("results", [])
    out = []
    for t in results:
        if t.get("ticker") == ticker:
            out.append(t)
    return out


def summarize_flow_strength(trades):
    total_premium = 0.0
    for t in trades:
        prem = clean_number(t.get("premium"))
        if prem:
            total_premium += prem
    return total_premium


# ========= CONFIDENCE + CATEGORY =========
def compute_confidence(direction, trend_dir, ind, news_score, flow_premium):
    base = 55

    if flow_premium >= 100000:
        base += 10
    if flow_premium >= 250000:
        base += 10

    if direction == "CALL" and trend_dir == "UP":
        base += 15
    if direction == "PUT" and trend_dir == "DOWN":
        base += 15

    rsi, macd, adx = ind["rsi"], ind["macd"], ind["adx"]

    if rsi is not None:
        if direction == "CALL" and rsi < 35:
            base += 5
        if direction == "PUT" and rsi > 65:
            base += 5

    if macd is not None:
        if direction == "CALL" and macd > 0:
            base += 5
        if direction == "PUT" and macd < 0:
            base += 5

    if adx is not None and adx >= 20:
        base += 5

    if news_score > 0 and direction == "CALL":
        base += 5
    if news_score < 0 and direction == "PUT":
        base += 5

    return max(10, min(99, base))


def classify_signal_type(direction, trend_dir, confidence):
    tag = "[ALL SIGNAL] 🔹"
    if confidence >= 70:
        tag = "[HIGH CONFIDENCE] 🔥"
    if (direction == "CALL" and trend_dir == "UP") or \
       (direction == "PUT" and trend_dir == "DOWN"):
        tag = "[SUPER-TREND + FLOW AGREEMENT] 💎"
    return tag


# ========= MESSAGE FORMATTING =========
def format_signal_message(ticker, direction, trend_dir, ind, news_score, confidence, tag, flow_premium):
    emoji = "🟢" if direction == "CALL" else "🔴"
    news_text = "Neutral"
    if news_score > 0:
        news_text = "Positive"
    elif news_score < 0:
        news_text = "Negative"

    return (
        f"{emoji} *OPTIONS SIGNAL*: {ticker}\n"
        f"{tag}\n\n"
        f"*Direction:* {'BULLISH (CALL)' if direction == 'CALL' else 'BEARISH (PUT)'} "
        f"({confidence}% confidence)\n"
        f"*SuperTrend:* {trend_dir}\n"
        f"*Flow premium (approx):* ${flow_premium:,.0f}\n"
        f"*RSI:* {ind['rsi']} | *MACD:* {ind['macd']} | *ADX:* {ind['adx']}\n"
        f"*News sentiment:* {news_text}\n"
        f"*Timeframe:* {TIMEFRAME}\n"
    )


def format_agreement_message(ticker, direction, trend_dir, ind, news_score, confidence, flow_premium):
    news_text = "Neutral"
    if news_score > 0:
        news_text = "Positive"
    elif news_score < 0:
        news_text = "Negative"

    return (
        "💎 *SUPER-TREND + FLOW AGREEMENT* 💎\n\n"
        f"*Ticker:* {ticker}\n"
        f"*Direction:* {'BULLISH (CALL)' if direction == 'CALL' else 'BEARISH (PUT)'}\n"
        f"*Confidence:* {confidence}%\n"
        f"*SuperTrend:* {trend_dir}\n"
        f"*Flow premium (approx):* ${flow_premium:,.0f}\n"
        f"*RSI:* {ind['rsi']} | *MACD:* {ind['macd']} | *ADX:* {ind['adx']}\n"
        f"*News sentiment:* {news_text}\n"
        f"*Timeframe:* {TIMEFRAME}\n\n"
        "This is a high‑quality alignment between trend and flow."
    )


def format_startup_message():
    wl = ", ".join(WATCHLIST)
    return (
        "🌅 *Good Morning, M!*\n\n"
        "The market is now *OPEN*.\n"
        "SuperTrend + Flow Bot is *LIVE* and scanning your watchlist:\n\n"
        f"{wl}\n\n"
        f"*Timeframe:* {TIMEFRAME}\n"
        "Let’s catch some moves today."
    )


def format_closing_message():
    return (
        "🌙 *Market Closed*\n\n"
        "SuperTrend + Flow Bot has stopped scanning for today.\n"
        "All signals, indicators, and flow analysis will resume tomorrow at 9:30 AM EST.\n\n"
        "Have a great evening, M."
    )


# ========= MAIN LOOP =========
def main():
    logging.info("🚀 SuperTrend + Flow combined bot started")

    if not MBOUM_API_KEY:
        logging.error("Missing MBOUM_API_KEY")
        return

    sent_startup_today = False
    sent_close_today = False
    last_date = now_ny().date()

    while True:
        dt = now_ny()
        today = dt.date()

        # Reset daily flags at midnight NY
        if today != last_date:
            sent_startup_today = False
            sent_close_today = False
            last_date = today

        # Scheduled startup (9:30) + fallback
        if is_market_open(dt) and not sent_startup_today:
            send_telegram(format_startup_message())
            sent_startup_today = True

        # Scheduled close (after 16:30) + fallback
        if dt.time() > MARKET_CLOSE and not sent_close_today:
            send_telegram(format_closing_message())
            sent_close_today = True

        # Hybrid mode: run 24/7, but only scan during market hours
        if not is_market_open(dt):
            logging.info("⏸ Outside trading hours. Sleeping 60s...")
            time.sleep(60)
            continue

        # ========== SCANNING LOOP ==========
        for ticker in WATCHLIST:
            logging.info(f"🔍 Scanning {ticker} on {TIMEFRAME}...")

            # 1) OHLC + SuperTrend
            ohlc = fetch_ohlc(ticker, TIMEFRAME)
            if len(ohlc) < 20:
                logging.info(f"Not enough OHLC data for {ticker}")
                continue

            trend_dir, st_signal = compute_supertrend_signals(ohlc)
            if st_signal is None:
                # No fresh flip; only alert on flips
                continue

            # BUY -> CALL, SELL -> PUT
            direction = "CALL" if st_signal == "BUY" else "PUT"

            # 2) Indicators
            ind = fetch_indicators(ticker)

            # 3) News
            news_score = fetch_news_sentiment(ticker)

            # 4) Flow (unusual options) for this ticker
            trades = fetch_unusual_for_ticker(ticker)
            flow_premium = summarize_flow_strength(trades)

            # 5) Confidence + category
            confidence = compute_confidence(direction, trend_dir, ind, news_score, flow_premium)
            tag = classify_signal_type(direction, trend_dir, confidence)

            # 6) Normal signal message
            signal_msg = format_signal_message(
                ticker, direction, trend_dir, ind, news_score, confidence, tag, flow_premium
            )
            logging.info(
                f"Signal {ticker} {direction} conf={confidence} flowPrem={flow_premium:,.0f} tag={tag}"
            )
            send_telegram(signal_msg)

            # 7) Separate SuperTrend + Flow Agreement message (in addition)
            if tag == "[SUPER-TREND + FLOW AGREEMENT] 💎":
                agreement_msg = format_agreement_message(
                    ticker, direction, trend_dir, ind, news_score, confidence, flow_premium
                )
                send_telegram(agreement_msg)

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
