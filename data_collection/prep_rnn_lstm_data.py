import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import ast

# Splits (no leakage)
TRAIN_START = "2015-01-01"; TRAIN_END = "2021-12-31"
VAL_START   = "2022-01-01"; VAL_END   = "2022-12-31"
TEST_START  = "2023-01-01"; TEST_END  = "2024-12-31"

def load_data_for_ticker(folder: Path) -> pd.DataFrame:
    """Loads and merges prices and features for a single ticker."""
    prices_path = folder / "prices.csv"
    merged_path = folder / "merged.csv"
    
    if not prices_path.exists() or not merged_path.exists():
        return None

    prices = pd.read_csv(prices_path, parse_dates=["Date"]).set_index("Date").sort_index()
    merged = pd.read_csv(merged_path, parse_dates=["Date"]).set_index("Date").sort_index()
    
    # Ensure sentiment column exists
    if "sent_compound_mean" not in merged.columns:
        merged["sent_compound_mean"] = 0.0
        
    data = prices.join(merged[["sent_compound_mean"]], how="left")
    data["sent_compound_mean"] = data["sent_compound_mean"].fillna(0.0)
    
    return data

def create_features(df: pd.DataFrame, context_days: int = 5):
    """Creates features (recent_prices) and labels (next_close)."""
    df['next_close'] = df['Close'].shift(-1)
    
    # Create a list of recent prices for each row
    price_history = []
    for i in range(len(df)):
        start_index = max(0, i - context_days + 1)
        price_window = df['Close'].iloc[start_index:i+1].tolist()
        if len(price_window) < context_days:
            # Pad with the earliest available price if not enough history
            price_window = [price_window[0]] * (context_days - len(price_window)) + price_window
        price_history.append(price_window)
        
    df['recent_prices'] = price_history
    
    # Drop rows where we can't have a target
    df = df.dropna(subset=['next_close'])
    
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="data_google_news")
    ap.add_argument("--out_dir",   type=str, default="rnn_lstm_data")
    ap.add_argument("--context_days", type=int, default=5)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ticker_data = []
    for p in in_dir.iterdir():
        if not p.is_dir():
            continue
        
        print(f"Processing {p.name}...")
        ticker_df = load_data_for_ticker(p)
        if ticker_df is not None:
            ticker_df = create_features(ticker_df, args.context_days)
            ticker_df['ticker'] = p.name.replace("_", ".")
            all_ticker_data.append(ticker_df)

    if not all_ticker_data:
        print("No data processed. Exiting.")
        return

    all_df = pd.concat(all_ticker_data).sort_index()
    
    # Select and rename columns
    final_df = all_df[['ticker', 'recent_prices', 'sent_compound_mean', 'next_close']].copy()
    final_df = final_df.rename(columns={'sent_compound_mean': 'sentiment', 'next_close': 'actual_price'})

    # Time splits
    train = final_df.loc[(final_df.index >= TRAIN_START) & (final_df.index <= TRAIN_END)].copy()
    val   = final_df.loc[(final_df.index >= VAL_START)   & (final_df.index <= VAL_END)].copy()
    test  = final_df.loc[(final_df.index >= TEST_START)  & (final_df.index <= TEST_END)].copy()

    # Save to CSV
    train.to_csv(out_dir / "train_rnn.csv")
    val.to_csv(out_dir / "val_rnn.csv")
    test.to_csv(out_dir / "test_rnn.csv")

    print("\\nDONE.")
    print("Rows total:", len(final_df), "| Train:", len(train), "Val:", len(val), "Test:", len(test))
    print(f"Wrote files to: {out_dir}")

if __name__ == "__main__":
    main()
