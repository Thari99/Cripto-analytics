
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
]

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

TIMEFRAME = pd.Timedelta(hours=1)


# --------------------------------------------------
# Prepare one trading pair
# --------------------------------------------------

def prepare_symbol(symbol: str) -> dict:
    """
    Load and prepare one historical dataset.

    Original raw files are never modified.

    A new segment begins whenever consecutive rows
    are separated by more than one hour.
    """

    filename = symbol.replace("/", "_") + "_1h.parquet"

    input_path = RAW_DATA_DIR / filename

    output_path = (
        PROCESSED_DATA_DIR
        / filename
    )

    print("\n" + "=" * 65)
    print(f"Preparing: {symbol}")
    print("=" * 65)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Historical file not found: {input_path}"
        )

    # 1. Load raw historical data
    df = pd.read_parquet(input_path)

    # 2. Check required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{symbol} is missing columns: {missing_columns}"
        )

    # Keep only the expected columns
    df = df[REQUIRED_COLUMNS].copy()

    # 3. Normalize timestamps
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="raise",
    )

    # 4. Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 5. Refuse duplicate timestamps
    if df["timestamp"].duplicated().any():
        raise ValueError(
            f"{symbol} contains duplicate timestamps."
        )

    # 6. Check the spacing between consecutive rows
    time_difference = df["timestamp"].diff()

    # First row starts the first segment.
    # A new segment begins after a gap greater than one hour.
    new_segment = (
        time_difference.isna()
        | (time_difference != TIMEFRAME)
    )

    df["segment_id"] = (
        new_segment.cumsum().astype("int64") - 1
    )

    # 7. Calculate gap information
    gap_mask = time_difference > TIMEFRAME

    missing_candles = 0

    for difference in time_difference[gap_mask]:
        missing_candles += max(
            int(difference / TIMEFRAME) - 1,
            0,
        )

    gap_count = int(gap_mask.sum())

    # 8. Calculate segment lengths
    segment_lengths = df.groupby("segment_id").size()

    segment_count = int(df["segment_id"].nunique())

    shortest_segment = int(segment_lengths.min())
    longest_segment = int(segment_lengths.max())

    # 9. Save the prepared data
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    # 10. Print the summary
    print(f"Input file: {input_path.name}")
    print(f"Output file: {output_path}")
    print(f"Total rows: {len(df):,}")
    print(f"First timestamp: {df['timestamp'].min()}")
    print(f"Last timestamp: {df['timestamp'].max()}")
    print(f"Missing hourly candles: {missing_candles}")
    print(f"Time gaps: {gap_count}")
    print(f"Continuous segments: {segment_count}")
    print(f"Shortest segment: {shortest_segment:,} rows")
    print(f"Longest segment: {longest_segment:,} rows")

    return {
        "symbol": symbol,
        "rows": len(df),
        "first_timestamp": df["timestamp"].min(),
        "last_timestamp": df["timestamp"].max(),
        "missing_hourly_candles": missing_candles,
        "gap_count": gap_count,
        "segment_count": segment_count,
        "shortest_segment_rows": shortest_segment,
        "longest_segment_rows": longest_segment,
        "output_file": str(output_path.relative_to(PROJECT_ROOT)),
        "status": "PASS",
    }


# --------------------------------------------------
# Main execution
# --------------------------------------------------

def main():
    results = []

    for symbol in SYMBOLS:
        results.append(
            prepare_symbol(symbol)
        )

    report = pd.DataFrame(results)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORTS_DIR
        / "data_preparation_report.csv"
    )

    report.to_csv(
        report_path,
        index=False,
    )

    print("\n" + "=" * 65)
    print("DATA PREPARATION SUMMARY")
    print("=" * 65)

    print(
        report[
            [
                "symbol",
                "rows",
                "missing_hourly_candles",
                "gap_count",
                "segment_count",
                "status",
            ]
        ].to_string(index=False)
    )

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
