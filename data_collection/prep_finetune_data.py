import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Splits (no leakage)
TRAIN_START = "2015-01-01"; TRAIN_END = "2021-12-31"
VAL_START   = "2022-01-01"; VAL_END   = "2022-12-31"
TEST_START  = "2023-01-01"; TEST_END  = "2024-12-31"

HEADLINE_CAP = 1500  # trim joined headlines in prompt

PAPER_PROMPT_HEADER = """You are a financial analyst with expertise in stock market forecasting.
Your task is to analyze market data and predict the next trading day stock price.
Use historical price trends, technical indicators, and sentiment analysis to provide an informed forecast.
Ensure that your predictions are well-justified, considering multiple financial factors.

• Predicted Stock Price: The forecasted close price for the next trading day.
• Price Movement Likelihood: The likelihood of the predicted stock price.
• Justification: Provide an explanation for the predicted stock price and the corresponding likelihood, considering the following:
  - Historical market data (e.g., recent closing prices).
  - Technical indicators (e.g., SMA, EMA, RSI, MACD, Bollinger Bands).
  - Sentiment analysis (e.g., news sentiment, market sentiment).

Please weigh these signals and justify the predicted stock price.
"""

def safe_join_headlines(s: pd.Series, cap_chars: int = HEADLINE_CAP) -> str:
    if s is None or len(s) == 0: return ""
    text = " | ".join(map(str, s.dropna().astype(str)))
    return text[:cap_chars]

def build_prompt(idx: pd.Timestamp, row: pd.Series, recent_closes: list[str]) -> str:
    return (
f"""{PAPER_PROMPT_HEADER}
TICKER: {row['ticker']}
DATE: {idx.strftime('%Y-%m-%d')}

RECENT CLOSING PRICES (most recent last): {", ".join(recent_closes)}

TECHNICAL INDICATORS:
SMA_20={row.get('SMA_20')}, SMA_50={row.get('SMA_50')},
EMA_12={row.get('EMA_12')}, EMA_26={row.get('EMA_26')},
RSI_14={row.get('RSI_14')}, MACD={row.get('MACD')}, MACD_signal={row.get('MACD_signal')}, MACD_hist={row.get('MACD_hist')},
BB_width_20_2={row.get('BB_width_20_2')}

SENTIMENT AGGREGATES:
headline_count={row.get('headline_count')}, sent_compound_mean={row.get('sent_compound_mean')}

HEADLINES (concise):
{str(row.get('titles_joined',''))}

Return STRICT JSON with keys:
- predicted_close (float, next-day close price),
- likelihood (float in [0,1]),
- justification (string, 1–2 sentences).
JSON:"""
    )

def build_response(next_close: float, likelihood: float) -> str:
    obj = {
        "predicted_close": float(next_close),
        "likelihood": float(likelihood),
        "justification": "n/a"
    }
    return json.dumps(obj, ensure_ascii=False)

def load_training_rows_or_merged(folder: Path) -> pd.DataFrame:
    """Use training_rows.csv if present; else rebuild minimal fields from merged.csv."""
    tr = folder / "training_rows.csv"
    if tr.exists():
        df = pd.read_csv(tr, parse_dates=["Date"]).set_index("Date").sort_index()
        for col in ["headline_count","sent_compound_mean","titles_joined"]:
            if col not in df.columns: df[col] = np.nan
        return df

    merged = pd.read_csv(folder/"merged.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    # Optional titles: derive from scored news if available
    scored_path = folder / "news_google_scored_vader.csv"
    if scored_path.exists():
        scored = pd.read_csv(scored_path, parse_dates=["published_at"])
        if not scored.empty:
            titles = (scored.assign(date=scored["published_at"].dt.floor("D"))
                             .groupby("date")["title"].apply(safe_join_headlines)
                             .rename("titles_joined"))
            merged = merged.join(titles, how="left")
    if "titles_joined" not in merged.columns:
        merged["titles_joined"] = ""
    # Ensure sentiment columns exist
    for c in ["headline_count","sent_compound_mean"]:
        if c not in merged.columns: merged[c] = np.nan
    return merged

def gather_all(input_dir: Path):
    """Yield per-ticker (features_df, prices_df, ticker_str)"""
    for p in input_dir.iterdir():
        if not p.is_dir(): continue
        if not (p/"prices.csv").exists(): 
            print(f"[WARN] skip {p.name}: missing prices.csv"); continue
        prices = pd.read_csv(p/"prices.csv", parse_dates=["Date"]).set_index("Date").sort_index()
        feats  = load_training_rows_or_merged(p)
        ticker = p.name.replace("_",".")
        yield feats, prices, ticker

def add_next_close_and_confidence(feats: pd.DataFrame, prices: pd.DataFrame):
    """Adds columns: next_close (label) and likelihood (vol-based)"""
    # Next-day close (label)
    next_close = prices["Close"].shift(-1)
    # Realized vol → confidence proxy
    ret = prices["Close"].pct_change()
    vol = ret.rolling(10).std()
    vol_rank = vol.rank(pct=True)  # 0..1
    # Map vol terciles to confidence (higher vol → lower confidence)
    def conf_from_rank(p):
        if pd.isna(p): return np.nan
        if p <= 1/3: return 0.9
        if p <= 2/3: return 0.7
        return 0.5

    conf = vol_rank.apply(conf_from_rank)

    out = feats.copy()
    out["next_close"] = next_close.reindex(out.index)
    out["confidence_proxy"] = conf.reindex(out.index)
    return out.dropna(subset=["next_close", "confidence_proxy"])

def to_jsonl(df: pd.DataFrame, all_prices: pd.DataFrame, out_path: Path, context_days: int = 5):
    """Write paper-style prompt/response jsonl."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        # build recent close context per (ticker, date)
        # We'll pass the *same ticker’s* recent closes only.
        for idx, row in df.iterrows():
            tkr = row["ticker"]
            # slice prices for this ticker from all_prices map
            px = all_prices[tkr]
            # collect last N closes up to idx (inclusive)
            hist = px.loc[px.index <= idx]["Close"].tail(context_days).round(4).tolist()
            recent = [f"{v:.4f}" for v in hist]
            prompt = build_prompt(idx, row, recent)
            response = build_response(row["next_close"], row["confidence_proxy"])
            w.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="data_google_news")
    ap.add_argument("--out_dir",   type=str, default="finetune_paper")
    ap.add_argument("--start",     type=str, default="2015-01-01")
    ap.add_argument("--end",       type=str, default="2024-12-31")
    ap.add_argument("--context_days", type=int, default=5)
    args = ap.parse_args()

    in_dir = Path(args.input_dir); out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Gather and enrich per ticker
    rows = []
    all_prices = {}
    for feats, prices, ticker in gather_all(in_dir):
        feats = feats.copy()
        feats["ticker"] = ticker
        enriched = add_next_close_and_confidence(feats, prices)
        # restrict date range
        enriched = enriched.loc[(enriched.index >= args.start) & (enriched.index <= args.end)]
        rows.append(enriched)
        all_prices[ticker] = prices  # keep to build recent-closes context

    all_df = pd.concat(rows).sort_index()
    # keep useful columns only
    keep = [
        # core features that appear in prompt
        "SMA_20","SMA_50","EMA_12","EMA_26","RSI_14","MACD","MACD_signal","MACD_hist","BB_width_20_2",
        "headline_count","sent_compound_mean","titles_joined",
        # labels we write into response
        "next_close","confidence_proxy",
        "ticker"
    ]
    present = [c for c in keep if c in all_df.columns]
    all_df = all_df[present]

    # Save a combined CSV for QA
    all_df.to_csv(out_dir/"all_supervised_price_labels.csv", index_label="Date")

    # Time splits
    train = all_df.loc[(all_df.index >= TRAIN_START) & (all_df.index <= TRAIN_END)].copy()
    val   = all_df.loc[(all_df.index >= VAL_START)   & (all_df.index <= VAL_END)].copy()
    test  = all_df.loc[(all_df.index >= TEST_START)  & (all_df.index <= TEST_END)].copy()

    # Write jsonl using paper prompt
    to_jsonl(train, all_prices, out_dir/"train.jsonl", context_days=args.context_days)
    to_jsonl(val,   all_prices, out_dir/"val.jsonl",   context_days=args.context_days)
    to_jsonl(test,  all_prices, out_dir/"test.jsonl",  context_days=args.context_days)

    print("DONE.")
    print("Rows total:", len(all_df), "| Train:", len(train), "Val:", len(val), "Test:", len(test))
    print(f"Wrote: {out_dir/'train.jsonl'}, {out_dir/'val.jsonl'}, {out_dir/'test.jsonl'}")
    print(f"Also wrote: {out_dir/'all_supervised_price_labels.csv'}")

if __name__ == "__main__":
    main()