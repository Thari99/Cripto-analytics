
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Trading pairs
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
]

# Expected columns
REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

# Known historical gap confirmed during our API checks.
# We are documenting this gap, not fabricating a candle.
KNOWN_MISSING_TIMESTAMPS = {
    pd.Timestamp("2023-03-24 13:00:00", tz="UTC"),
}


def validate_symbol(symbol: str) -> dict:
    """Validate historical OHLCV data without modifying the source file."""

    filename = symbol.replace("/", "_") + "_1h.parquet"
    file_path = RAW_DATA_DIR / filename

    print("\n" + "=" * 65)
    print(f"Validating: {symbol}")
    print("=" * 65)

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return {
            "symbol": symbol,
            "status": "FILE_NOT_FOUND",
        }

    df = pd.read_parquet(file_path)

    # 1. Required columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print(f"ERROR: Missing columns: {missing_columns}")
        return {
            "symbol": symbol,
            "status": "MISSING_COLUMNS",
            "missing_columns": ",".join(missing_columns),
        }

    # 2. Normalize timestamps to UTC
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    invalid_timestamps = int(df["timestamp"].isna().sum())

    # Check ordering using original row order, before sorting
    valid_timestamp_df = df.dropna(subset=["timestamp"]).copy()

    duplicate_timestamps = int(
        valid_timestamp_df["timestamp"].duplicated().sum()
    )

    out_of_order = int(
        (
            valid_timestamp_df["timestamp"].diff()
            < pd.Timedelta(0)
        ).sum()
    )

    # 3. Numeric values
    numeric_columns = ["open", "high", "low", "close", "volume"]

    for column in numeric_columns:
        valid_timestamp_df[column] = pd.to_numeric(
            valid_timestamp_df[column],
            errors="coerce",
        )

    numeric_values = valid_timestamp_df[numeric_columns]

    non_numeric_or_missing = int(
        numeric_values.isna().any(axis=1).sum()
    )

    infinite_values = int(
        numeric_values.isin([float("inf"), float("-inf")])
        .any(axis=1)
        .sum()
    )

    # 4. Price and OHLC consistency
    price_columns = ["open", "high", "low", "close"]

    non_positive_prices = int(
        (valid_timestamp_df[price_columns] <= 0)
        .any(axis=1)
        .sum()
    )

    invalid_ohlc = int(
        (
            (
                valid_timestamp_df["high"]
                < valid_timestamp_df[
                    ["open", "low", "close"]
                ].max(axis=1)
            )
            |
            (
                valid_timestamp_df["low"]
                > valid_timestamp_df[
                    ["open", "high", "close"]
                ].min(axis=1)
            )
        ).sum()
    )

    negative_volume = int(
        (valid_timestamp_df["volume"] < 0).sum()
    )

    # 5. Detect missing hourly candles
    unique_timestamps = (
        valid_timestamp_df["timestamp"]
        .drop_duplicates()
        .sort_values()
    )

    missing_hourly_candles = 0
    missing_hour_gaps = 0
    largest_gap = pd.Timedelta(0)
    detected_missing_timestamps = set()

    if len(unique_timestamps) > 1:
        time_differences = unique_timestamps.diff()
        gap_mask = time_differences > pd.Timedelta(hours=1)

        missing_hour_gaps = int(gap_mask.sum())

        missing_hourly_candles = int(
            (
                time_differences[gap_mask]
                / pd.Timedelta(hours=1)
            ).round().astype(int).sub(1).sum()
        )

        largest_gap = time_differences.max()

        # Record each missing hourly timestamp when the gap is
        # an exact multiple of one hour.
        timestamps_list = unique_timestamps.tolist()

        for previous, current in zip(
            timestamps_list[:-1],
            timestamps_list[1:],
        ):
            difference = current - previous

            if difference > pd.Timedelta(hours=1):
                if difference % pd.Timedelta(hours=1) == pd.Timedelta(0):
                    missing_count = (
                        int(difference / pd.Timedelta(hours=1)) - 1
                    )

                    for step in range(1, missing_count + 1):
                        detected_missing_timestamps.add(
                            previous + pd.Timedelta(hours=step)
                        )

    known_gaps_found = (
        detected_missing_timestamps & KNOWN_MISSING_TIMESTAMPS
    )

    unexpected_gaps = (
        detected_missing_timestamps - KNOWN_MISSING_TIMESTAMPS
    )

    known_gap_count = len(known_gaps_found)
    unexpected_gap_count = len(unexpected_gaps)

    # 6. Check freshness of the latest completed candle
    now = pd.Timestamp.now(tz="UTC")
    expected_latest_completed = now.floor("h") - pd.Timedelta(hours=1)

    latest_timestamp = (
        unique_timestamps.max()
        if not unique_timestamps.empty
        else pd.NaT
    )

    latest_completed_is_missing = (
        pd.isna(latest_timestamp)
        or latest_timestamp < expected_latest_completed
    )

    # 7. Print results
    print(f"File: {file_path.name}")
    print(f"Total rows: {len(df):,}")
    print(f"First timestamp: {unique_timestamps.min() if not unique_timestamps.empty else pd.NaT}")
    print(f"Last timestamp: {latest_timestamp}")
    print(f"Invalid timestamps: {invalid_timestamps}")
    print(f"Duplicate timestamps: {duplicate_timestamps}")
    print(f"Out-of-order rows: {out_of_order}")
    print(f"Rows with missing/non-numeric values: {non_numeric_or_missing}")
    print(f"Rows with infinite values: {infinite_values}")
    print(f"Rows with non-positive prices: {non_positive_prices}")
    print(f"Rows with invalid OHLC relationships: {invalid_ohlc}")
    print(f"Rows with negative volume: {negative_volume}")
    print(f"Missing hourly candles: {missing_hourly_candles}")
    print(f"Known missing candles: {known_gap_count}")
    print(f"Unexpected missing candles: {unexpected_gap_count}")
    print(f"Number of gaps: {missing_hour_gaps}")
    print(f"Largest time gap: {largest_gap}")
    print(f"Expected latest completed candle: {expected_latest_completed}")
    print(f"Latest candle is stale: {latest_completed_is_missing}")

    # 8. Determine status
    basic_checks_passed = (
        invalid_timestamps == 0
        and duplicate_timestamps == 0
        and out_of_order == 0
        and non_numeric_or_missing == 0
        and infinite_values == 0
        and non_positive_prices == 0
        and invalid_ohlc == 0
        and negative_volume == 0
    )

    # The documented gap is allowed, but unexpected gaps and stale
    # data still require review.
    passed = (
        basic_checks_passed
        and unexpected_gap_count == 0
        and not latest_completed_is_missing
    )

    status = "PASS" if passed else "REVIEW"

    print(f"Validation status: {status}")

    return {
        "symbol": symbol,
        "rows": len(df),
        "first_timestamp": (
            unique_timestamps.min()
            if not unique_timestamps.empty
            else pd.NaT
        ),
        "last_timestamp": latest_timestamp,
        "invalid_timestamps": invalid_timestamps,
        "duplicate_timestamps": duplicate_timestamps,
        "out_of_order": out_of_order,
        "non_numeric_or_missing": non_numeric_or_missing,
        "infinite_values": infinite_values,
        "non_positive_prices": non_positive_prices,
        "invalid_ohlc": invalid_ohlc,
        "negative_volume": negative_volume,
        "missing_hourly_candles": missing_hourly_candles,
        "missing_hour_gaps": missing_hour_gaps,
        "known_gap_count": known_gap_count,
        "unexpected_gap_count": unexpected_gap_count,
        "known_gap_timestamps": ",".join(
            str(timestamp) for timestamp in sorted(known_gaps_found)
        ),
        "unexpected_gap_timestamps": ",".join(
            str(timestamp) for timestamp in sorted(unexpected_gaps)
        ),
        "largest_gap": str(largest_gap),
        "latest_completed_is_missing": latest_completed_is_missing,
        "status": status,
    }


def main():
    results = []

    for symbol in SYMBOLS:
        results.append(validate_symbol(symbol))

    report = pd.DataFrame(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / "data_validation_report.csv"
    report.to_csv(report_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 65)

    columns_to_show = [
        "symbol",
        "rows",
        "missing_hourly_candles",
        "known_gap_count",
        "unexpected_gap_count",
        "duplicate_timestamps",
        "status",
    ]

    available_columns = [
        column for column in columns_to_show
        if column in report.columns
    ]

    print(report[available_columns].to_string(index=False))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
