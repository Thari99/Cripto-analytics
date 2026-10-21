from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
]

EXPECTED_INTERVAL = pd.Timedelta(hours=1)


def find_missing_candles(symbol: str) -> None:
    filename = symbol.replace("/", "_") + "_1h.parquet"
    file_path = RAW_DATA_DIR / filename

    df = pd.read_parquet(file_path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    timestamps = (
        df["timestamp"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    differences = timestamps.diff()

    gap_positions = differences[
        differences > EXPECTED_INTERVAL
    ].index

    print(f"\n{'=' * 60}")
    print(f"Symbol: {symbol}")
    print(f"{'=' * 60}")

    if len(gap_positions) == 0:
        print("No missing hourly candles found.")
        return

    for position in gap_positions:
        previous_timestamp = timestamps.iloc[position - 1]
        next_timestamp = timestamps.iloc[position]

        missing = pd.date_range(
            start=previous_timestamp + EXPECTED_INTERVAL,
            end=next_timestamp - EXPECTED_INTERVAL,
            freq=EXPECTED_INTERVAL,
        )

        print(f"Previous candle: {previous_timestamp}")
        print(f"Next candle:     {next_timestamp}")
        print(f"Missing count:   {len(missing)}")

        for timestamp in missing:
            print(f"Missing timestamp: {timestamp}")


def main():
    for symbol in SYMBOLS:
        find_missing_candles(symbol)


if __name__ == "__main__":
    main()
