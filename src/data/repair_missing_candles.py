
from pathlib import Path
import shutil
import time

import ccxt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
]

TIMEFRAME = "1h"

MISSING_TIMESTAMP = pd.Timestamp(
    "2023-03-24 13:00:00",
    tz="UTC",
)

COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def recover_symbol(exchange, symbol):
    filename = symbol.replace("/", "_") + "_1h.parquet"
    file_path = RAW_DATA_DIR / filename

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    print(f"\n{'=' * 60}")
    print(f"Recovering {symbol}")
    print(f"{'=' * 60}")

    # Load existing data
    df = pd.read_parquet(file_path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="raise",
    )

    # Check whether the candle already exists
    existing = df[df["timestamp"] == MISSING_TIMESTAMP]

    if not existing.empty:
        print("Candle already exists. No change required.")
        return

    # Request the candle beginning at 13:00 UTC
    since = int(MISSING_TIMESTAMP.timestamp() * 1000)

    candles = exchange.fetch_ohlcv(
    symbol,
    timeframe=TIMEFRAME,
    since=since,
    limit=10,
    )

    print(f"Requested timestamp: {MISSING_TIMESTAMP}")
    print(f"Requested milliseconds: {since}")
    print(f"Returned candles: {len(candles)}")

    for candle in candles:
        timestamp = pd.to_datetime(
            candle[0],
            unit="ms",
            utc=True,
        )

        print(
            f"Timestamp: {timestamp} | "
            f"Open: {candle[1]} | "
            f"High: {candle[2]} | "
            f"Low: {candle[3]} | "
            f"Close: {candle[4]} | "
            f"Volume: {candle[5]}"
        )

    if not candles:
        print("ERROR: Binance returned no candles.")
        return

    recovered = pd.DataFrame(candles, columns=COLUMNS)

    recovered["timestamp"] = pd.to_datetime(
        recovered["timestamp"],
        unit="ms",
        utc=True,
    )

    # Keep only the exact missing timestamp
    recovered = recovered[
        recovered["timestamp"] == MISSING_TIMESTAMP
    ].copy()

    if recovered.empty:
        print(
            "ERROR: The requested candle was not returned. "
            "Original data remains unchanged."
        )
        return

    # Validate the recovered candle
    price_columns = ["open", "high", "low", "close"]

    if recovered[price_columns].isna().any().any():
        print("ERROR: Recovered candle contains missing prices.")
        return

    if (recovered[price_columns] <= 0).any().any():
        print("ERROR: Recovered candle contains invalid prices.")
        return

    candle = recovered.iloc[0]

    if (
        candle["high"]
        < max(candle["open"], candle["low"], candle["close"])
        or candle["low"]
        > min(candle["open"], candle["high"], candle["close"])
        or candle["volume"] < 0
    ):
        print("ERROR: Recovered candle failed OHLCV validation.")
        return

    # Create a backup before modifying the original file
    backup_path = file_path.with_suffix(".parquet.bak")

    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"Backup created: {backup_path.name}")

    # Merge and sort
    repaired = pd.concat(
        [df, recovered],
        ignore_index=True,
    )

    repaired = (
        repaired
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Ensure the candle was added
    if not (repaired["timestamp"] == MISSING_TIMESTAMP).any():
        print("ERROR: Recovered candle was not added.")
        return

    # Save the repaired dataset
    repaired.to_parquet(
        file_path,
        index=False,
        compression="snappy",
    )

    print(f"Recovered candle: {MISSING_TIMESTAMP}")
    print(f"OHLCV: {candle.to_dict()}")
    print(f"Rows before: {len(df):,}")
    print(f"Rows after:  {len(repaired):,}")
    print("Status: RECOVERED")


def main():
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "timeout": 30000,
    })

    try:
        exchange.load_markets()

        for index, symbol in enumerate(SYMBOLS):
            try:
                recover_symbol(exchange, symbol)

            except (ccxt.BaseError, ValueError, OSError) as error:
                print(
                    f"ERROR recovering {symbol}: "
                    f"{type(error).__name__}: {error}"
                )

            # Avoid sending requests too quickly
            if index < len(SYMBOLS) - 1:
                time.sleep(1)

    finally:
        exchange.close()


if __name__ == "__main__":
    main()
