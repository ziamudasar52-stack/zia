import os, time, logging, requests, yfinance as yf
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from math import fabs

WATCHLIST=["SLV","GLD","NFLX","META","AAPL","TSLA","NVDA","GOOG","MSFT","AMZN","SPY","SPX","AMD","PLTR","QQQ","ORCL","IBM","ABNB"]
TIMEFRAMES=["1m","5m","15m","1h"]
MBOUM_API_KEY=os.getenv("MBOUM_API_KEY")
TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID")
TZ_NY=ZoneInfo("America/New_York")
US_HOLIDAYS={(1,1),(7,4),(12,25)}
SCAN_INTERVAL=60

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")

def clean_number(v):
    if v is None:return None
    if isinstance(v,(int,float)):return float(v)
    if isinstance(v,str):
        x=v.replace("$","").replace(",","").replace("%","").strip()
        try:return float(x)
        except:return None
    return None

def now_ny():return datetime.now(TZ_NY)
def is_us_holiday(dt):return(dt.month,dt.day)in US_HOLIDAYS
def is_market_open(dt):
    if dt.weekday()>4:return False
    if is_us_holiday(dt):return False
    t=dt.time()
    return dtime(9,30)<=t<=dtime(16,30)

def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:return
    try:
        r=requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                        data={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"Markdown"},timeout=10)
        if r.status_code!=200:logging.error(f"Telegram error: {r.text}")
    except Exception as e:logging.error(f"Telegram send error: {e}")

def mboum_get(url,params=None):
    headers={"Authorization":f"Bearer {MBOUM_API_KEY}"} if MBOUM_API_KEY else {}
    try:
        r=requests.get(url,headers=headers,params=params or {},timeout=15)
        if r.status_code!=200:return None
        return r.json()
    except:return None

# ------------------ YAHOO FINANCE WITH RETRIES ------------------
def fetch_ohlc_yahoo(ticker,interval,retries=3):
    for attempt in range(retries):
        try:
            data=yf.download(ticker,period="5d",interval=interval,progress=False,threads=False)
            if data is not None and not data.empty:
                ohlc=[]
                for _,row in data.iterrows():
                    o,h,l,c=[clean_number(row.get(x)) for x in["Open","High","Low","Close"]]
                    if None in(o,h,l,c):continue
                    ohlc.append((o,h,l,c))
                return ohlc
        except Exception as e:
            logging.error(f"Yahoo error {ticker} {interval}: {e}")
        time.sleep(1)
    logging.error(f"Yahoo failed for {ticker} {interval}")
    return []

# ------------------ SUPERTREND ------------------
def compute_atr(ohlc,p=10):
    if len(ohlc)<p+1:return[]
    trs=[]
    for i in range(1,len(ohlc)):
        _,h0,l0,c0=ohlc[i-1]
        _,h1,l1,c1=ohlc[i]
        trs.append(max(h1-l1,abs(h1-c0),abs(l1-c0)))
    atr=[sum(trs[:p])/p]
    for i in range(p,len(trs)):atr.append((atr[-1]*(p-1)+trs[i])/p)
    return[None]*(p+1)+atr

def compute_supertrend_signals(ohlc,p=10,m=3.0):
    if len(ohlc)<p+5:return(None,None)
    atr=compute_atr(ohlc,p)
    n=len(ohlc)
    up=[None]*n;dn=[None]*n;trend=[1]*n
    for i in range(n):
        o,h,l,c=ohlc[i]
        if atr[i]is None:continue
        hl2=(h+l)/2
        bu=hl2-m*atr[i];bd=hl2+m*atr[i]
        if i==0:up[i]=bu;dn[i]=bd;trend[i]=1;continue
        up_prev=up[i-1]if up[i-1]is not None else bu
        dn_prev=dn[i-1]if dn[i-1]is not None else bd
        c_prev=ohlc[i-1][3]
        up[i]=max(bu,up_prev) if c_prev>up_prev else bu
        dn[i]=min(bd,dn_prev) if c_prev<dn_prev else bd
        if trend[i-1]==-1 and c>dn_prev:trend[i]=1
        elif trend[i-1]==1 and c<up_prev:trend[i]=-1
        else:trend[i]=trend[i-1]
    last,prev=trend[-1],trend[-2]
    sig="BUY"if last==1 and prev==-1 else("SELL"if last==-1 and prev==1 else None)
    return("UP"if last==1 else"DOWN",sig)

# ------------------ INDICATORS ------------------
def compute_rsi(cl,p=14):
    if len(cl)<p+1:return None
    g=[];l=[]
    for i in range(1,len(cl)):
        d=cl[i]-cl[i-1]
        g.append(d if d>=0 else 0);l.append(-d if d<0 else 0)
    ag=sum(g[:p])/p;al=sum(l[:p])/p
    for i in range(p,len(g)):
        ag=(ag*(p-1)+g[i])/p;al=(al*(p-1)+l[i])/p
    if al==0:return 100
    rs=ag/al
    return round(100-(100/(1+rs)),2)

def compute_macd(cl,fa=12,sl=26,sg=9):
    if len(cl)<sl+sg:return None
    def ema(v,p):
        k=2/(p+1);e=[sum(v[:p])/p]
        for x in v[p:]:e.append(x*k+e[-1]*(1-k))
        return e
    fe=ema(cl,fa);se=ema(cl,sl)
    ln=min(len(fe),len(se))
    mac=[fe[-ln+i]-se[-ln+i]for i in range(ln)]
    if len(mac)<sg:return None
    sig=ema(mac,sg)
    return round(mac[-1]-sig[-1],4)

def compute_adx(ohlc,p=14):
    if len(ohlc)<p+2:return None
    h=[x[1]for x in ohlc];l=[x[2]for x in ohlc];c=[x[3]for x in ohlc]
    trs=[];pdm=[];mdm=[]
    for i in range(1,len(ohlc)):
        tr=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]));trs.append(tr)
        up=h[i]-h[i-1];dn=l[i-1]-l[i]
        pdm.append(up if up>dn and up>0 else 0);mdm.append(dn if dn>up and dn>0 else 0)
    def sm(v,p):
        s=[sum(v[:p])]
        for x in v[p:]:s.append(s[-1]-(s[-1]/p)+x)
        return s
    tr_s=sm(trs,p);pdm_s=sm(pdm,p);mdm_s=sm(mdm,p)
    di_p=[100*(pdm_s[i]/tr_s[i])if tr_s[i]!=0 else 0 for i in range(len(tr_s))]
    di_m=[100*(mdm_s[i]/tr_s[i])if tr_s[i]!=0 else 0 for i in range(len(tr_s))]
    dx=[100*abs(di_p[i]-di_m[i])/(di_p[i]+di_m[i]) if(di_p[i]+di_m[i])!=0 else 0 for i in range(len(di_p))]
    if len(dx)<p:return None
    ad=sm(dx,p)[-1]/p
    return round(ad,2)

def compute_indicators(ohlc):
    cl=[x[3]for x in ohlc]
    return{"rsi":compute_rsi(cl),"macd":compute_macd(cl),"adx":compute_adx(ohlc)}

# ------------------ NEWS + FLOW ------------------
def fetch_news_sentiment(t):
    d=mboum_get("https://api.mboum.com/v2/markets/news",{"ticker":t,"type":"ALL"})
    if not d:return 0
    arts=d if isinstance(d,list) else d.get("results",[])
    if not isinstance(arts,list):return 0
    pos=["beat","upgrade","strong","record","growth","surge"]
    neg=["miss","downgrade","lawsuit","recall","fraud","weak"]
    s=0
    for a in arts[:5]:
        t=(a.get("title")or a.get("headline")or"").lower()
        if any(w in t for w in pos):s+=1
        if any(w in t for w in neg):s-=1
    return max(-2,min(2,s))

def fetch_unusual_for_ticker(t):
    d=mboum_get("https://api.mboum.com/v1/markets/options/unusual-options-activity",
                {"type":"STOCKS","page":"1"})
    if not d:return[]
    return[x for x in d.get("results",[]) if x.get("ticker")==t]

def summarize_flow(tr):
    tot=0
    for t in tr:
        p=clean_number(t.get("premium"))
        if p:tot+=p
    return tot

# ------------------ CONFIDENCE ------------------
def compute_conf(direction,trend,ind,news,flow):
    b=55
    if flow>=100000:b+=10
    if flow>=250000:b+=10
    if direction=="CALL"and trend=="UP":b+=15
    if direction=="PUT"and trend=="DOWN":b+=15
    rsi,macd,adx=ind["rsi"],ind["macd"],ind["adx"]
    if rsi is not None:
        if direction=="CALL"and rsi<35:b+=5
        if direction=="PUT"and rsi>65:b+=5
    if macd is not None:
        if direction=="CALL"and macd>0:b+=5
        if direction=="PUT"and macd<0:b+=5
    if adx is not None and adx>=20:b+=5
    if news>0 and direction=="CALL":b+=5
    if news<0 and direction=="PUT":b+=5
    return max(10,min(99,b))

def news_text(n):return"Positive"if n>0 else("Negative"if n<0 else"Neutral")

# ------------------ MESSAGES ------------------
def format_multi(t,tf_res,news,flow):
    lines=[f"📊 *MULTI‑TIMEFRAME SIGNAL*: {t}\n"]
    nt=news_text(news)
    for tf in["1m","5m","15m","1h"]:
        if tf not in tf_res:continue
        r=tf_res[tf]
        e="🟢"if r["direction"]=="CALL"else"🔴"
        lines.append(f"{e} *({tf})* {r['direction']} — {r['confidence']}% confidence\n"
                     f"SuperTrend: {r['trend']}\n"
                     f"RSI: {r['ind']['rsi']} | MACD: {r['ind']['macd']} | ADX: {r['ind']['adx']}\n"
                     f"Flow Premium: ${flow:,.0f}\n"
                     f"News Sentiment: {nt}\n")
    return"\n".join(lines)

def format_agree(t,tf,r,news,flow):
    nt=news_text(news)
    return("💎 *SUPER-TREND + FLOW AGREEMENT* 💎\n\n"
           f"*Ticker:* {t}\n"
           f"*Timeframe:* {tf}\n"
           f"*Direction:* {'BULLISH (CALL)'if r['direction']=='CALL'else'BEARISH (PUT)'}\n"
           f"*Confidence:* {r['confidence']}%\n"
           f"*SuperTrend:* {r['trend']}\n"
           f"*Flow premium:* ${flow:,.0f}\n"
           f"*RSI:* {r['ind']['rsi']} | *MACD:* {r['ind']['macd']} | *ADX:* {r['ind']['adx']}\n"
           f"*News:* {nt}")

def startup_msg():
    return("🌅 *Good Morning, M!*\n\nMarket open. Bot live.\n\nWatchlist:\n"+
           ", ".join(WATCHLIST))

def close_msg():
    return("🌙 *Market Closed*\n\nBot paused until tomorrow.")

# ------------------ MAIN LOOP ------------------
def main():
    logging.info("🚀 Bot started")
    if not MBOUM_API_KEY:logging.error("Missing Mboum key");return
    sent_start=False;sent_close=False;last=now_ny().date()

    while True:
        dt=now_ny();today=dt.date()
        if today!=last:sent_start=False;sent_close=False;last=today
        if is_market_open(dt)and not sent_start:
            send_telegram(startup_msg());sent_start=True
        if dt.time()>dtime(16,30)and not sent_close:
            send_telegram(close_msg());sent_close=True
        if not is_market_open(dt):
            time.sleep(60);continue

        for t in WATCHLIST:
            logging.info(f"🔍 {t}")
            trades=fetch_unusual_for_ticker(t)
            flow=summarize_flow(trades)
            news=fetch_news_sentiment(t)
            tf_res={};any_flip=False;agree=None

            for tf in TIMEFRAMES:
                ohlc=fetch_ohlc_yahoo(t,tf)
                if len(ohlc)<20:continue
                trend,flip=compute_supertrend_signals(ohlc)
                direction="CALL"if trend=="UP"else"PUT"
                if flip:
                    any_flip=True
                    direction="CALL"if flip=="BUY"else"PUT"
                ind=compute_indicators(ohlc)
                conf=compute_conf(direction,trend,ind,news,flow)
                tf_res[tf]={"direction":direction,"trend":trend,"ind":ind,"confidence":conf,"flip":flip}
                if tf=="5m"and flip and((direction=="CALL"and trend=="UP")or(direction=="PUT"and trend=="DOWN")):
                    agree=(tf,tf_res[tf])

            if any_flip and tf_res:
                send_telegram(format_multi(t,tf_res,news,flow))
                if agree:
                    tf,r=agree
                    send_telegram(format_agree(t,tf,r,news,flow))

            time.sleep(0.5)

        time.sleep(SCAN_INTERVAL)

if __name__=="__main__":main()
