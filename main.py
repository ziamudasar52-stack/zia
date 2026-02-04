import os
import time
import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from math import fabs
import requests
import yfinance as yf

# ========= CONFIG =========
WATCHLIST = [
    "SLV", "GLD", "NFLX", "META", "AAPL", "TSLA", "NVDA",
    "GOOG", "MSFT", "AMZN", "SPY", "SPX", "AMD", "PLTR",
    "QQQ", "ORCL", "IBM", "ABNB"
]

# Multi-timeframe scanning (Yahoo Finance)
TIMEFRAMES = ["1m", "5m", "15m", "1h"]

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
    headers = {"Authorization": f"Bearer {MBOUM_API_KEY}"} if MBOUM_API_KEY else {}
    try:
        resp = requests.get(url, headers=headers, params=params or {}, timeout=15)
        logging.info(f"📡 {url} - {resp.status_code}")
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception as e:
        logging.error(f"Mboum error: {e}")
        return None


# ========= YAHOO FINANCE OHLC =========
def fetch_ohlc_yahoo(ticker, interval):
    """
    Fetch intraday OHLC from Yahoo Finance for the given interval.
    """
    try:
        data = yf.download(ticker, period="5d", interval=interval, progress=False)
    except Exception as e:
        logging.error(f"Yahoo Finance error for {ticker} {interval}: {e}")
        return []

    if data is None or data.empty:
        return []

    ohlc = []
    for _, row in data.iterrows():
        o = clean_number(row.get("Open"))
        h = clean_number(row.get("High"))
        l = clean_number(row.get("Low"))
        c = clean_number(row.get("Close"))
        if None in (o, h, l, c):
            continue
        ohlc.append((o, h, l, c))
    return ohlc


# ========= SUPERTREND =========
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


# ========= LOCAL INDICATORS (RSI, MACD, ADX) =========
def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rs = float("inf")
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)


def compute_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return None
    def ema(values, period):
        k = 2 / (period + 1)
        ema_vals = [sum(values[:period]) / period]
        for v in values[period:]:
            ema_vals.append(v * k + ema_vals[-1] * (1 - k))
        return ema_vals

    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    # align lengths
    diff_len = min(len(fast_ema), len(slow_ema))
    macd_line = [fast_ema[-diff_len + i] - slow_ema[-diff_len + i] for i in range(diff_len)]
    if len(macd_line) < signal:
        return None
    signal_line = ema(macd_line, signal)
    macd_val = macd_line[-1] - signal_line[-1]
    return round(macd_val, 4)


def compute_adx(ohlc, period=14):
    if len(ohlc) < period + 2:
        return None
    highs = [h for _, h, _, _ in ohlc]
    lows = [l for _, _, l, _ in ohlc]
    closes = [c for _, _, _, c in ohlc]

    trs = []
    plus_dm = []
    minus_dm = []
    for i in range(1, len(ohlc)):
        high = highs[i]
        low = lows[i]
        prev_high = highs[i - 1]
        prev_low = lows[i - 1]
        prev_close = closes[i - 1]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        trs.append(tr)

        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)

    def smoothed(values, period):
        first = sum(values[:period])
        out = [first]
        for v in values[period:]:
            out.append(out[-1] - (out[-1] / period) + v)
        return out

    tr_smooth = smoothed(trs, period)
    plus_dm_smooth = smoothed(plus_dm, period)
    minus_dm_smooth = smoothed(minus_dm, period)

    if len(tr_smooth) == 0:
        return None

    plus_di = []
    minus_di = []
    for i in range(len(tr_smooth)):
        if tr_smooth[i] == 0:
            plus_di.append(0.0)
            minus_di.append(0.0)
        else:
            plus_di.append(100 * (plus_dm_smooth[i] / tr_smooth[i]))
            minus_di.append(100 * (minus_dm_smooth[i] / tr_smooth[i]))

    dx = []
    for i in range(len(plus_di)):
        denom = plus_di[i] + minus_di[i]
        if denom == 0:
            dx.append(0.0)
        else:
            dx.append(100 * abs(plus_di[i] - minus_di[i]) / denom)

    if len(dx) < period:
        return None
    adx_vals = smoothed(dx, period)
    adx = adx_vals[-1] / period
    return round(adx, 2)


def compute_indicators_from_ohlc(ohlc):
    closes = [c for _, _, _, c in ohlc]
    rsi = compute_rsi(closes)
    macd = compute_macd(closes)
    adx = compute_adx(ohlc)
    return {"rsi": rsi, "macd": macd, "adx": adx}


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


# ========= CONFIDENCE + CATEGORIES =========
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


def news_text_from_score(news_score):
    if news_score > 0:
        return "Positive"
    if news_score < 0:
        return "Negative"
    return "Neutral"


# ========= MESSAGE FORMATTING =========
def format_multi_timeframe_message(ticker, tf_results, news_score, flow_premium):
    lines = []
    lines.append(f"📊 *MULTI‑TIMEFRAME SIGNAL*: {ticker}\n")

    news_text = news_text_from_score(news_score)

    # Order timeframes nicely
    order = ["1m", "5m", "15m", "1h"]
    for tf in order:
        if tf not in tf_results:
            continue
        r = tf_results[tf]
        direction = r["direction"]
        emoji = "🟢" if direction == "CALL" else "🔴"
        conf = r["confidence"]
        trend_dir = r["trend_dir"]
        rsi = r["ind"]["rsi"]
        macd = r["ind"]["macd"]
        adx = r["ind"]["adx"]

        lines.append(
            f"{emoji} *({tf})* {direction} — {conf}% confidence\n"
            f"SuperTrend: {trend_dir}\n"
            f"RSI: {rsi} | MACD: {macd} | ADX: {adx}\n"
            f"Flow Premium: ${flow_premium:,.0f}\n"
            f"News Sentiment: {news_text}\n"
        )

    return "\n".join(lines).strip()


def format_agreement_message(ticker, direction, trend_dir, ind, news_score, confidence, flow_premium, timeframe):
    news_text = news_text_from_score(news_score)

    return (
        "💎 *SUPER-TREND + FLOW AGREEMENT* 💎\n\n"
        f"*Ticker:* {ticker}\n"
        f"*Timeframe:* {timeframe}\n"
        f"*Direction:* {'BULLISH (CALL)' if direction == 'CALL' else 'BEARISH (PUT)'}\n"
        f"*Confidence:* {confidence}%\n"
        f"*SuperTrend:* {trend_dir}\n"
        f"*Flow premium (approx):* ${flow_premium:,.0f}\n"
        f"*RSI:* {ind['rsi']} | *MACD:* {ind['macd']} | *ADX:* {ind['adx']}\n"
        f"*News sentiment:* {news_text}\n\n"
        "This is a high‑quality alignment between trend and flow."
    )


def format_startup_message():
    wl = ", ".join(WATCHLIST)
    return (
        "🌅 *Good Morning, M!*\n\n"
        "The market is now *OPEN*.\n"
        "SuperTrend + Flow Bot is *LIVE* and scanning your watchlist:\n\n"
        f"{wl}\n\n"
        "Timeframes: 1m, 5m, 15m, 1h\n"
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
    logging.info("🚀 SuperTrend + Flow multi‑timeframe bot started")

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
            logging.info(f"🔍 Scanning {ticker} on {TIMEFRAMES}...")

            # 1) Flow + news (per ticker, shared across timeframes)
            trades = fetch_unusual_for_ticker(ticker)
            flow_premium = summarize_flow_strength(trades)
            news_score = fetch_news_sentiment(ticker)

            tf_results = {}
            any_flip = False
            agreement_candidate = None  # (tf, result)

            for tf in TIMEFRAMES:
                ohlc = fetch_ohlc_yahoo(ticker, tf)
                if len(ohlc) < 20:
                    logging.info(f"Not enough OHLC data for {ticker} {tf}")
                    continue

                trend_dir, st_signal = compute_supertrend_signals(ohlc)
                # If no flip, we still infer direction from trend_dir
                if st_signal is None:
                    direction = "CALL" if trend_dir == "UP" else "PUT"
                else:
                    any_flip = True
                    direction = "CALL" if st_signal == "BUY" else "PUT"

                ind = compute_indicators_from_ohlc(ohlc)
                confidence = compute_confidence(direction, trend_dir, ind, news_score, flow_premium)

                tf_results[tf] = {
                    "direction": direction,
                    "trend_dir": trend_dir,
                    "ind": ind,
                    "confidence": confidence,
                    "flipped": st_signal is not None,
                }

                # Candidate for agreement: 5m + flow + aligned direction
                if tf == "5m" and st_signal is not None:
                    if (direction == "CALL" and trend_dir == "UP") or (direction == "PUT" and trend_dir == "DOWN"):
                        agreement_candidate = (tf, tf_results[tf])

            # Only send combined message when ANY timeframe has a SuperTrend flip
            if any_flip and tf_results:
                msg = format_multi_timeframe_message(ticker, tf_results, news_score, flow_premium)
                logging.info(f"Sending multi‑timeframe signal for {ticker}")
                send_telegram(msg)

                # Separate agreement message if 5m + flow align
                if agreement_candidate is not None:
                    tf, res = agreement_candidate
                    agreement_msg = format_agreement_message(
                        ticker,
                        res["direction"],
                        res["trend_dir"],
                        res["ind"],
                        news_score,
                        res["confidence"],
                        flow_premium,
                        tf
                    )
                    send_telegram(agreement_msg)

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
