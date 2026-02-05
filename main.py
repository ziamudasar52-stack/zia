import os
import time
import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import requests
import yfinance as yf

# ------------------ CONFIG ------------------
WATCHLIST = [
    "SLV", "GLD", "NFLX", "META", "AAPL", "TSLA", "NVDA", "GOOG", "MSFT",
    "AMZN", "SPY", "SPX", "AMD", "PLTR", "QQQ", "ORCL", "IBM", "ABNB"
]
TIMEFRAMES = ["1m", "5m", "15m", "1h"]

MBOUM_API_KEY = os.getenv("MBOUM_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TZ_NY = ZoneInfo("America/New_York")
US_HOLIDAYS = {(1, 1), (7, 4), (12, 25)}

# Scan every 90 seconds (reduced load on Yahoo)
SCAN_INTERVAL = 90

# Smart TTL per timeframe (seconds)
CACHE_TTL_MAP = {
    "1m": 30,
    "5m": 60,
    "15m": 120,
    "1h": 300,
}

# (ticker, timeframe) -> {"ts": datetime, "ohlc": list}
ohlc_cache = {}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------ HELPERS ------------------
def clean_number(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        x = v.replace("$", "").replace(",", "").replace("%", "").strip()
        try:
            return float(x)
        except ValueError:
            return None
    return None


def now_ny():
    return datetime.now(TZ_NY)


def is_us_holiday(dt: datetime) -> bool:
    return (dt.month, dt.day) in US_HOLIDAYS


def is_market_open(dt: datetime) -> bool:
    if dt.weekday() > 4:
        return False
    if is_us_holiday(dt):
        return False
    t = dt.time()
    return dtime(9, 30) <= t <= dtime(16, 30)


def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown"
            },
            timeout=10
        )
        if r.status_code != 200:
            logging.error(f"Telegram error: {r.text}")
    except Exception as e:
        logging.error(f"Telegram send error: {e}")


def mboum_get(url, params=None):
    headers = {"Authorization": f"Bearer {MBOUM_API_KEY}"} if MBOUM_API_KEY else {}
    try:
        r = requests.get(url, headers=headers, params=params or {}, timeout=15)
        if r.status_code != 200:
            logging.error(f"Mboum error {r.status_code}: {r.text}")
            return None
        return r.json()
    except Exception as e:
        logging.error(f"Mboum request error: {e}")
        return None

# ------------------ YAHOO WITH CACHING + RETRIES ------------------
def fetch_ohlc_yahoo(ticker, interval, retries=3):
    now = now_ny()
    key = (ticker, interval)
    ttl = CACHE_TTL_MAP.get(interval, 60)

    # Use cache if fresh
    cached = ohlc_cache.get(key)
    if cached:
        age = (now - cached["ts"]).total_seconds()
        if age < ttl:
            return cached["ohlc"]

    # Otherwise fetch fresh
    for attempt in range(retries):
        try:
            data = yf.download(
                ticker,
                period="5d",
                interval=interval,
                progress=False,
                threads=False
            )
            if data is not None and not data.empty:
                ohlc = []
                for _, row in data.iterrows():
                    o = clean_number(row.get("Open"))
                    h = clean_number(row.get("High"))
                    l = clean_number(row.get("Low"))
                    c = clean_number(row.get("Close"))
                    if None in (o, h, l, c):
                        continue
                    ohlc.append((o, h, l, c))

                # update cache
                ohlc_cache[key] = {"ts": now, "ohlc": ohlc}
                return ohlc
        except Exception as e:
            logging.error(f"Yahoo error {ticker} {interval}: {e}")
        time.sleep(1)

    logging.error(f"Yahoo failed for {ticker} {interval}")
    return []

# ------------------ SUPERTREND ------------------
def compute_atr(ohlc, p=10):
    if len(ohlc) < p + 1:
        return []
    trs = []
    for i in range(1, len(ohlc)):
        _, h0, l0, c0 = ohlc[i - 1]
        _, h1, l1, c1 = ohlc[i]
        tr = max(h1 - l1, abs(h1 - c0), abs(l1 - c0))
        trs.append(tr)
    atr = [sum(trs[:p]) / p]
    for i in range(p, len(trs)):
        atr.append((atr[-1] * (p - 1) + trs[i]) / p)
    return [None] * (p + 1) + atr


def compute_supertrend_signals(ohlc, p=10, m=3.0):
    if len(ohlc) < p + 5:
        return (None, None)
    atr = compute_atr(ohlc, p)
    n = len(ohlc)
    upper = [None] * n
    lower = [None] * n
    trend = [1] * n

    for i in range(n):
        o, h, l, c = ohlc[i]
        if atr[i] is None:
            continue
        hl2 = (h + l) / 2
        basic_upper = hl2 - m * atr[i]
        basic_lower = hl2 + m * atr[i]

        if i == 0:
            upper[i] = basic_upper
            lower[i] = basic_lower
            trend[i] = 1
            continue

        prev_upper = upper[i - 1] if upper[i - 1] is not None else basic_upper
        prev_lower = lower[i - 1] if lower[i - 1] is not None else basic_lower
        prev_close = ohlc[i - 1][3]

        upper[i] = max(basic_upper, prev_upper) if prev_close > prev_upper else basic_upper
        lower[i] = min(basic_lower, prev_lower) if prev_close < prev_lower else basic_lower

        if trend[i - 1] == -1 and c > prev_lower:
            trend[i] = 1
        elif trend[i - 1] == 1 and c < prev_upper:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    last = trend[-1]
    prev = trend[-2]
    flip = None
    if last == 1 and prev == -1:
        flip = "BUY"
    elif last == -1 and prev == 1:
        flip = "SELL"

    return ("UP" if last == 1 else "DOWN", flip)

# ------------------ INDICATORS ------------------
def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-diff)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def compute_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return None

    def ema(values, period):
        k = 2 / (period + 1)
        e = [sum(values[:period]) / period]
        for v in values[period:]:
            e.append(v * k + e[-1] * (1 - k))
        return e

    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    ln = min(len(fast_ema), len(slow_ema))
    macd_line = [fast_ema[-ln + i] - slow_ema[-ln + i] for i in range(ln)]
    if len(macd_line) < signal:
        return None
    signal_line = ema(macd_line, signal)
    hist = macd_line[-1] - signal_line[-1]
    return round(hist, 4)


def compute_adx(ohlc, period=14):
    if len(ohlc) < period + 2:
        return None
    highs = [x[1] for x in ohlc]
    lows = [x[2] for x in ohlc]
    closes = [x[3] for x in ohlc]

    trs = []
    plus_dm = []
    minus_dm = []
    for i in range(1, len(ohlc)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)

    def smoothed(values, period):
        s = [sum(values[:period])]
        for v in values[period:]:
            s.append(s[-1] - (s[-1] / period) + v)
        return s

    tr_s = smoothed(trs, period)
    plus_dm_s = smoothed(plus_dm, period)
    minus_dm_s = smoothed(minus_dm, period)

    plus_di = [
        100 * (plus_dm_s[i] / tr_s[i]) if tr_s[i] != 0 else 0
        for i in range(len(tr_s))
    ]
    minus_di = [
        100 * (minus_dm_s[i] / tr_s[i]) if tr_s[i] != 0 else 0
        for i in range(len(tr_s))
    ]

    dx = [
        100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        if (plus_di[i] + minus_di[i]) != 0 else 0
        for i in range(len(plus_di))
    ]
    if len(dx) < period:
        return None
    adx_vals = smoothed(dx, period)
    adx = adx_vals[-1] / period
    return round(adx, 2)


def compute_indicators(ohlc):
    closes = [x[3] for x in ohlc]
    return {
        "rsi": compute_rsi(closes),
        "macd": compute_macd(closes),
        "adx": compute_adx(ohlc),
    }

# ------------------ NEWS + FLOW ------------------
def fetch_news_sentiment(ticker):
    data = mboum_get(
        "https://api.mboum.com/v2/markets/news",
        {"ticker": ticker, "type": "ALL"}
    )
    if not data:
        return 0
    articles = data if isinstance(data, list) else data.get("results", [])
    if not isinstance(articles, list):
        return 0

    positive_words = ["beat", "upgrade", "strong", "record", "growth", "surge"]
    negative_words = ["miss", "downgrade", "lawsuit", "recall", "fraud", "weak"]

    score = 0
    for a in articles[:5]:
        title = (a.get("title") or a.get("headline") or "").lower()
        if any(w in title for w in positive_words):
            score += 1
        if any(w in title for w in negative_words):
            score -= 1
    return max(-2, min(2, score))


def fetch_unusual_for_ticker(ticker):
    data = mboum_get(
        "https://api.mboum.com/v1/markets/options/unusual-options-activity",
        {"type": "STOCKS", "page": "1"}
    )
    if not data:
        return []
    results = data.get("results", [])
    if not isinstance(results, list):
        return []
    return [t for t in results if t.get("ticker") == ticker]


def summarize_flow(trades):
    total = 0.0
    for t in trades:
        p = clean_number(t.get("premium"))
        if p:
            total += p
    return total

# ------------------ CONFIDENCE + TEXT ------------------
def compute_conf(direction, trend, indicators, news_score, flow_premium):
    base = 55
    if flow_premium >= 100000:
        base += 10
    if flow_premium >= 250000:
        base += 10
    if direction == "CALL" and trend == "UP":
        base += 15
    if direction == "PUT" and trend == "DOWN":
        base += 15

    rsi = indicators["rsi"]
    macd = indicators["macd"]
    adx = indicators["adx"]

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


def news_text(score):
    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


def format_multi(ticker, tf_results, news_score, flow_premium):
    lines = [f"📊 *MULTI‑TIMEFRAME SIGNAL*: {ticker}\n"]
    nt = news_text(news_score)
    for tf in ["1m", "5m", "15m", "1h"]:
        if tf not in tf_results:
            continue
        r = tf_results[tf]
        emoji = "🟢" if r["direction"] == "CALL" else "🔴"
        lines.append(
            f"{emoji} *({tf})* {r['direction']} — {r['confidence']}% confidence\n"
            f"SuperTrend: {r['trend']}\n"
            f"RSI: {r['ind']['rsi']} | MACD: {r['ind']['macd']} | ADX: {r['ind']['adx']}\n"
            f"Flow Premium: ${flow_premium:,.0f}\n"
            f"News Sentiment: {nt}\n"
        )
    return "\n".join(lines)


def format_agree(ticker, timeframe, r, news_score, flow_premium):
    nt = news_text(news_score)
    return (
        "💎 *SUPER-TREND + FLOW AGREEMENT* 💎\n\n"
        f"*Ticker:* {ticker}\n"
        f"*Timeframe:* {timeframe}\n"
        f"*Direction:* {'BULLISH (CALL)' if r['direction']=='CALL' else 'BEARISH (PUT)'}\n"
        f"*Confidence:* {r['confidence']}%\n"
        f"*SuperTrend:* {r['trend']}\n"
        f"*Flow premium:* ${flow_premium:,.0f}\n"
        f"*RSI:* {r['ind']['rsi']} | *MACD:* {r['ind']['macd']} | *ADX:* {r['ind']['adx']}\n"
        f"*News:* {nt}"
    )


def startup_msg():
    return (
        "🌅 *Good Morning, M!*\n\n"
        "Market open. Bot live.\n\n"
        "Watchlist:\n" + ", ".join(WATCHLIST)
    )


def close_msg():
    return "🌙 *Market Closed*\n\nBot paused until tomorrow."
# ------------------ MAIN LOOP ------------------
def main():
    logging.info("🚀 Bot started")

    if not MBOUM_API_KEY:
        logging.error("Missing Mboum API key")
        return

    sent_start = False
    sent_close = False
    last_date = now_ny().date()

    while True:
        dt = now_ny()
        today = dt.date()

        # Reset daily flags
        if today != last_date:
            sent_start = False
            sent_close = False
            last_date = today

        # Startup message
        if is_market_open(dt) and not sent_start:
            send_telegram(startup_msg())
            sent_start = True

        # Closing message
        if dt.time() > dtime(16, 30) and not sent_close:
            send_telegram(close_msg())
            sent_close = True

        # If market closed, sleep
        if not is_market_open(dt):
            time.sleep(60)
            continue

        # ------------------ SCAN LOOP ------------------
        for ticker in WATCHLIST:
            logging.info(f"🔍 Scanning {ticker} on {TIMEFRAMES}...")

            # Skip ticker if cooling down
            cd = ticker_cooldown.get(ticker)
            now = now_ny()
            if cd and now < cd:
                logging.warning(f"{ticker} cooling down until {cd}, skipping.")
                continue

            # Fetch flow + news once per ticker
            trades = fetch_unusual_for_ticker(ticker)
            flow_premium = summarize_flow(trades)
            news_score = fetch_news_sentiment(ticker)

            tf_results = {}
            any_flip = False
            agreement = None  # (timeframe, result)

            # -------- TIMEFRAME LOOP --------
            for tf in TIMEFRAMES:
                ohlc = fetch_ohlc_yahoo(ticker, tf)

                if len(ohlc) < 20:
                    logging.info(f"Not enough OHLC data for {ticker} {tf}")
                    continue

                trend, flip = compute_supertrend_signals(ohlc)
                if trend is None:
                    continue

                # Base direction from trend
                direction = "CALL" if trend == "UP" else "PUT"

                # Flip overrides direction
                if flip == "BUY":
                    any_flip = True
                    direction = "CALL"
                elif flip == "SELL":
                    any_flip = True
                    direction = "PUT"

                indicators = compute_indicators(ohlc)
                confidence = compute_conf(direction, trend, indicators, news_score, flow_premium)

                tf_results[tf] = {
                    "direction": direction,
                    "trend": trend,
                    "ind": indicators,
                    "confidence": confidence,
                    "flip": flip,
                }

                # Agreement logic on 5m
                if tf == "5m" and flip and (
                    (direction == "CALL" and trend == "UP") or
                    (direction == "PUT" and trend == "DOWN")
                ):
                    agreement = (tf, tf_results[tf])

            # -------- SEND SIGNALS --------
            if any_flip and tf_results:
                msg = format_multi(ticker, tf_results, news_score, flow_premium)
                send_telegram(msg)

                if agreement:
                    tf, res = agreement
                    agree_msg = format_agree(ticker, tf, res, news_score, flow_premium)
                    send_telegram(agree_msg)

            # Small delay between tickers to reduce Yahoo load
            time.sleep(0.5)

        # Sleep between full scans
        time.sleep(SCAN_INTERVAL)


# ------------------ ENTRYPOINT ------------------
if __name__ == "__main__":
    main()
