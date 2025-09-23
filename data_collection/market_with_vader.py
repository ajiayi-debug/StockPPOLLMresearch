# collect_market_data_google_news.py
import os, json, time, ssl, certifi, logging
from typing import Dict, List, Optional
from urllib.parse import urlencode, quote_plus

import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timezone
from tqdm import tqdm

# --- Enforce system certs (helps behind proxies/VPNs)
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

# --- Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Config
# -----------------------------
TICKERS: Dict[str, str] = {
    "Apple": "AAPL",
    "HSBC": "HSBC",
    "Pepsi": "PEP",
    "Tencent": "0700.HK",
    "Toyota": "7203.T",
}

# Query strings per ticker (tweak for recall/precision)
COMPANY_QUERIES: Dict[str, str] = {
    "AAPL": '"Apple Inc" OR AAPL OR "Apple stock"',
    "HSBC": '"HSBC Holdings" OR HSBC OR "HSBC stock"',
    "PEP": '"PepsiCo" OR Pepsi OR PEP OR "Pepsi stock"',
    "0700.HK": '"Tencent Holdings" OR Tencent OR "0700.HK" OR "Tencent stock"',
    "7203.T": '"Toyota Motor" OR Toyota OR "7203.T" OR "Toyota stock"',
}

START = "2015-01-01"
END   = "2024-12-31"
OUTPUT_DIR = "data_google_news"
REQUEST_PAUSE_SEC = 0.25   # be polite to Google News RSS

# -----------------------------
# Technical Indicators
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]
    out["SMA_20"] = c.rolling(20).mean()
    out["SMA_50"] = c.rolling(50).mean()
    out["EMA_12"] = ema(c, 12)
    out["EMA_26"] = ema(c, 26)
    out["RSI_14"] = rsi_wilder(c, 14)
    out["MACD"] = out["EMA_12"] - out["EMA_26"]
    out["MACD_signal"] = ema(out["MACD"], 9)
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    out["BB_up_20_2"] = bb_mid + 2 * bb_std
    out["BB_dn_20_2"] = bb_mid - 2 * bb_std
    out["BB_width_20_2"] = (out["BB_up_20_2"] - out["BB_dn_20_2"]) / bb_mid
    return out

# -----------------------------
# Prices (yfinance)
# -----------------------------
def fetch_ohlcv(yf_ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(yf_ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(-1)
    df = df.rename(columns=str.title)[["Open","High","Low","Close","Volume"]]
    df.index = pd.to_datetime(df.index, utc=True)
    df["session_ts"] = df.index.tz_convert("UTC")  # for merge_asof alignment
    return df

# -----------------------------
# Google News RSS — monthly windows
# -----------------------------
def _month_windows(start: str, end: str):
    p_start = pd.to_datetime(start).to_period("M")
    p_end = pd.to_datetime(end).to_period("M")
    for p in pd.period_range(p_start, p_end, freq="M"):
        b = pd.Timestamp(p.start_time)
        e = pd.Timestamp(p.end_time)
        b = max(b, pd.to_datetime(start))
        e = min(e, pd.to_datetime(end) + pd.Timedelta(hours=23, minutes=59, seconds=59))
        yield b.date().isoformat(), e.date().isoformat()

def fetch_google_news(query: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch headlines via Google News RSS, chunked by month.
    Uses date operators (after:, before:) to narrow per window.
    """
    rows = []
    base = "https://news.google.com/rss/search"
    month_windows = list(_month_windows(start, end))
    logger.debug(f"Fetching news for {len(month_windows)} month windows")
    
    for b, e in tqdm(month_windows, desc="Fetching news", leave=False):
        q = f"({query}) after:{b} before:{e}"
        params = {
            "q": q,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }
        url = f"{base}?{urlencode(params, quote_via=quote_plus)}"
        try:
            feed = feedparser.parse(url)
            entries_count = len(feed.entries) if feed.entries else 0
            logger.debug(f"Fetched {entries_count} entries for {b[:7]}")
        except Exception as e1:
            logger.warning(f"RSS fetch failed for {b[:7]}: {e1}")
            time.sleep(REQUEST_PAUSE_SEC)
            continue

        for e_item in feed.entries:
            # published timestamp
            dt = None
            if getattr(e_item, "published_parsed", None):
                dt = pd.Timestamp(*e_item.published_parsed[:6], tz="UTC")
            title = getattr(e_item, "title", "") or ""
            link = getattr(e_item, "link", None)
            # publisher/source if available
            pub = None
            if hasattr(e_item, "source") and getattr(e_item.source, "title", None):
                pub = e_item.source.title
            rows.append({
                "published_at": dt,
                "title": title,
                "publisher": pub,
                "link": link
            })
        time.sleep(REQUEST_PAUSE_SEC)

    if not rows:
        return pd.DataFrame(columns=["published_at", "title", "publisher", "link"])

    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"]).sort_values("published_at")
    return df.reset_index(drop=True)

# -----------------------------
# VADER Scoring + Alignment
# -----------------------------
def init_vader() -> Optional[SentimentIntensityAnalyzer]:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    try:
        return SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"[WARN] VADER init failed: {e}")
        return None

def score_vader(headlines: pd.DataFrame, sia: Optional[SentimentIntensityAnalyzer]) -> pd.DataFrame:
    if headlines.empty or sia is None:
        return headlines.assign(compound=pd.Series(dtype=float))
    out = headlines.copy()
    out["title"] = out["title"].fillna("")
    out["compound"] = out["title"].apply(lambda t: sia.polarity_scores(t)["compound"])
    return out

def align_to_trading_days(headlines_scored: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Map each headline timestamp to the most recent trading session at/behind it,
    then aggregate sentiment per session (trading day).
    """
    if headlines_scored.empty:
        return pd.DataFrame(columns=["session_ts","headline_count","sent_compound_mean"])

    h = headlines_scored.sort_values("published_at").copy()
    h_key = h[["published_at"]].rename(columns={"published_at":"ts"}).sort_values("ts")
    px_key = prices[["session_ts"]].reset_index(drop=True).sort_values("session_ts")

    aligned = pd.merge_asof(h_key, px_key, left_on="ts", right_on="session_ts", direction="backward")
    h["session_ts"] = aligned["session_ts"]  # keep timezone info
    h = h.dropna(subset=["session_ts"])

    daily = (h.groupby("session_ts")
               .agg(headline_count=("title","count"),
                    sent_compound_mean=("compound","mean"))
               .reset_index())
    return daily

# -----------------------------
# Build per ticker
# -----------------------------
def build_for_ticker(label: str, yf_ticker: str, start: str, end: str) -> None:
    out_dir = os.path.join(OUTPUT_DIR, yf_ticker.replace("^","").replace(".","_"))
    os.makedirs(out_dir, exist_ok=True)
    logger.debug(f"Created output directory: {out_dir}")

    # 1) Prices
    logger.info(f"Fetching price data for {yf_ticker}")
    prices = fetch_ohlcv(yf_ticker, start, end)
    if prices.empty:
        logger.warning(f"No price data found for {label} ({yf_ticker})")
        return
    logger.info(f"Retrieved {len(prices)} price records")

    # 2) Indicators
    logger.info("Computing technical indicators")
    indicators = compute_indicators(prices.drop(columns=["session_ts"]))
    # Ensure index has a stable name for later set_index("Date")
    indicators.index = pd.DatetimeIndex(indicators.index, name="Date")
    logger.info(f"Computed indicators for {len(indicators)} trading days")

    # 3) Headlines via Google News RSS
    query = COMPANY_QUERIES.get(yf_ticker, yf_ticker)
    logger.info(f"Fetching news headlines with query: '{query}'")
    headlines = fetch_google_news(query, start, end)
    logger.info(f"Retrieved {len(headlines)} headlines")

    # 4) VADER sentiment per headline
    logger.info("Scoring sentiment with VADER")
    sia = init_vader()
    scored = score_vader(headlines, sia)
    logger.info(f"Scored sentiment for {len(scored)} headlines")

    # 5) Align to trading days (actual sessions) and aggregate
    logger.info("Aligning sentiment to trading days")
    daily_sent = align_to_trading_days(scored, prices)
    logger.info(f"Aggregated sentiment for {len(daily_sent)} trading days")

    # 6) Merge with prices/indicators (convenience file)
    daily = indicators.copy()
    daily["session_ts"] = daily.index.tz_convert("UTC")
    merged = (daily.reset_index()
                    .merge(daily_sent, on="session_ts", how="left")
                    .set_index("Date"))
    merged["headline_count"] = merged["headline_count"].fillna(0)

    # ---- Save outputs ----
    logger.info("Saving output files")
    
    files_to_save = [
        ("prices.csv", prices.drop(columns=["session_ts"]), True),
        ("indicators.csv", indicators, True), 
        ("news_google_raw.csv", headlines, False),
        ("news_google_scored_vader.csv", scored, False),
        ("sentiment_by_trading_day.csv", daily_sent, False),
        ("merged.csv", merged.drop(columns=["session_ts"]), True)
    ]
    
    for filename, data, use_index_label in tqdm(files_to_save, desc="Saving files", leave=False):
        filepath = os.path.join(out_dir, filename)
        if use_index_label:
            data.to_csv(filepath, index_label="Date")
        else:
            data.to_csv(filepath, index=False)
        logger.debug(f"Saved {filename} with {len(data)} records")

    meta = {
        "label": label,
        "yahoo_ticker": yf_ticker,
        "google_query": query,
        "start": start,
        "end": end,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_count": {
            "prices": len(prices),
            "indicators": len(indicators), 
            "headlines": len(headlines),
            "sentiment_days": len(daily_sent)
        }
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.debug("Saved metadata")

    logger.info(f"Successfully completed {label} ({yf_ticker}) → {out_dir}")

# -----------------------------
# Main
# -----------------------------
def main():
    logger.info(f"Starting data collection for {len(TICKERS)} tickers")
    logger.info(f"Date range: {START} to {END}")
    
    # Use tqdm for progress tracking
    for label, tkr in tqdm(TICKERS.items(), desc="Processing tickers"):
        logger.info(f"Starting processing for {label} ({tkr})")
        try:
            build_for_ticker(label, tkr, START, END)
            logger.info(f"Successfully completed {label} ({tkr})")
        except Exception as e:
            logger.error(f"Failed to process {label} ({tkr}): {e}")
            raise
    
    logger.info("Data collection completed successfully")

if __name__ == "__main__":
    main()