
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_FILE = DATA_DIR / "download_metadata.json"

START_DATE = "2022-01-01T00:00:00Z"
TIMEFRAME = "1h"
BATCH_SIZE = 1000
MAX_RETRIES = 5

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
]

COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# CONNECT TO THE EXCHANGE
# --------------------------------------------------

def create_exchange():
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "timeout": 30000,
    })

    exchange.load_markets()

    if not exchange.has.get("fetchOHLCV"):
        raise RuntimeError("Exchange does not support OHLCV")

    logger.info("Connected to %s", exchange.id)

    return exchange


# --------------------------------------------------
# DOWNLOAD ONE BATCH WITH RETRIES
# --------------------------------------------------

def fetch_batch(exchange, symbol, since):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return exchange.fetch_ohlcv(
                symbol,
                timeframe=TIMEFRAME,
                since=since,
                limit=BATCH_SIZE,
            )

        except ccxt.BaseError as error:
            if attempt == MAX_RETRIES:
                raise

            delay = min(2 ** attempt, 30)

            logger.warning(
                "%s failed: %s. Retrying in %s seconds.",
                symbol,
                error,
                delay,
            )

            time.sleep(delay)


# --------------------------------------------------
# LOAD PREVIOUS DOWNLOAD
# --------------------------------------------------

def load_existing_file(path):
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)

    df = pd.read_parquet(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    return df[COLUMNS]


# --------------------------------------------------
# DOWNLOAD HISTORICAL DATA
# --------------------------------------------------

def download_symbol(exchange, symbol):
    filename = symbol.replace("/", "_") + "_1h.parquet"
    output_path = DATA_DIR / filename

    existing = load_existing_file(output_path)

    start_ms = exchange.parse8601(START_DATE)

    if not existing.empty:
        latest_ms = int(
            existing["timestamp"].max().timestamp() * 1000
        )

        # Re-fetch the latest saved candle in case
        # it needs to be refreshed.
        since = max(start_ms, latest_ms)
    else:
        since = start_ms

    collected = []

    logger.info("Downloading %s", symbol)

    while True:
        batch = fetch_batch(exchange, symbol, since)

        if not batch:
            break

        collected.extend(batch)

        last_timestamp = batch[-1][0]
        next_since = last_timestamp + 1

        if next_since <= since:
            raise RuntimeError(
                f"Pagination stopped advancing for {symbol}"
            )

        since = next_since

        logger.info(
            "%s: downloaded %s candles in this run",
            symbol,
            len(collected),
        )

        # Stop when the exchange returns a short batch.
        if len(batch) < BATCH_SIZE:
            break

    if collected:
        new_df = pd.DataFrame(
            collected,
            columns=COLUMNS,
        )

        new_df["timestamp"] = pd.to_datetime(
            new_df["timestamp"],
            unit="ms",
            utc=True,
        )

        df = pd.concat(
            [existing, new_df],
            ignore_index=True,
        )
    else:
        df = existing.copy()

    if df.empty:
        logger.warning("No candles returned for %s", symbol)
        return

    # Sort chronologically and remove duplicates.
    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Restrict data to the requested date range.
    df = df[
        df["timestamp"] >= pd.Timestamp(START_DATE)
    ].copy()

    # Exclude the currently incomplete hourly candle.
    current_hour = pd.Timestamp.now(
        tz="UTC"
    ).floor("h")

    df = df[df["timestamp"] < current_hour].copy()

    # Validate numeric fields.
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(subset=COLUMNS)

    # Check basic OHLCV consistency.
    valid = (
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["high"] >= df["low"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    )

    df = df[valid].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Save the cleaned historical candles.
    df.to_parquet(
        output_path,
        index=False,
        compression="snappy",
    )

    logger.info(
        "Saved %s | rows=%s | first=%s | last=%s",
        output_path,
        len(df),
        df["timestamp"].min(),
        df["timestamp"].max(),
    )


# --------------------------------------------------
# SAVE METADATA
# --------------------------------------------------

def save_metadata(exchange_id):
    metadata = {
        "exchange": exchange_id,
        "market_type": "spot",
        "timeframe": TIMEFRAME,
        "symbols": SYMBOLS,
        "requested_start_date": START_DATE,
        "downloaded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "format": "parquet",
        "timestamp_timezone": "UTC",
        "columns": COLUMNS,
        "api_key_required": False,
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    logger.info("Metadata saved to %s", METADATA_FILE)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    exchange = create_exchange()

    try:
        for symbol in SYMBOLS:
            if symbol not in exchange.markets:
                raise ValueError(
                    f"{symbol} is not available on this exchange"
                )

            download_symbol(exchange, symbol)

        save_metadata(exchange.id)

    finally:
        exchange.close()

    logger.info("Historical download finished.")


if __name__ == "__main__":
    main()
