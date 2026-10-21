
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
]

# Initial experimental settings.
HORIZON = 4
STOP_ATR_MULTIPLIER = 1.0
TARGET_ATR_MULTIPLIER = 2.0

# Illustrative round-trip trading cost.
# Replace this with realistic, exchange-specific assumptions
# before using results to judge profitability.
ROUND_TRIP_COST_RATE = 0.001


# --------------------------------------------------
# Target calculation
# --------------------------------------------------

def evaluate_trade(
    future_candles: pd.DataFrame,
    entry_price: float,
    stop_distance: float,
    target_distance: float,
    direction: int,
) -> float:
    """
    Evaluate a hypothetical trade over future candles.

    Returns:
        Net return as a decimal.

    Conservative assumptions:
    - Stop-loss is assumed first if both barriers are touched
      in the same candle.
    - Stop-loss gaps are filled at the candle open if the open
      has already crossed the stop.
    - Take-profit gaps are conservatively filled at the target.
    - Trading cost is deducted from the gross return.

    This remains an approximation because OHLCV candles do not
    reveal the exact order of intrabar price movements.
    """

    if entry_price <= 0 or stop_distance <= 0:
        return np.nan

    if direction == 1:
        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance
    else:
        stop_price = entry_price + stop_distance
        target_price = entry_price - target_distance

    for candle in future_candles.itertuples(index=False):
        candle_open = candle.open
        candle_high = candle.high
        candle_low = candle.low

        if direction == 1:
            # A stop crossed at the open may incur a larger loss.
            if candle_open <= stop_price:
                exit_price = candle_open
                return (
                    (exit_price - entry_price) / entry_price
                    - ROUND_TRIP_COST_RATE
                )

            stop_hit = candle_low <= stop_price
            target_hit = candle_high >= target_price

            if stop_hit and target_hit:
                return (
                    (stop_price - entry_price) / entry_price
                    - ROUND_TRIP_COST_RATE
                )

            if stop_hit:
                return (
                    (stop_price - entry_price) / entry_price
                    - ROUND_TRIP_COST_RATE
                )

            if target_hit:
                return (
                    (target_price - entry_price) / entry_price
                    - ROUND_TRIP_COST_RATE
                )

        else:
            if candle_open >= stop_price:
                exit_price = candle_open
                return (
                    (entry_price - exit_price) / entry_price
                    - ROUND_TRIP_COST_RATE
                )

            stop_hit = candle_high >= stop_price
            target_hit = candle_low <= target_price

            if stop_hit and target_hit:
                return (
                    (entry_price - stop_price) / entry_price
                    - ROUND_TRIP_COST_RATE
                )

            if stop_hit:
                return (
                    (entry_price - stop_price) / entry_price
                    - ROUND_TRIP_COST_RATE
                )

            if target_hit:
                return (
                    (entry_price - target_price) / entry_price
                    - ROUND_TRIP_COST_RATE
                )

    # Neither barrier was hit. Mark the position at the final
    # candle close, then deduct the assumed trading cost.
    final_close = future_candles.iloc[-1]["close"]

    if direction == 1:
        gross_return = (final_close - entry_price) / entry_price
    else:
        gross_return = (entry_price - final_close) / entry_price

    return gross_return - ROUND_TRIP_COST_RATE


def build_targets(symbol: str) -> dict:
    """
    Generate LONG, SHORT, and NO TRADE labels.

    Features at candle t are used to create a hypothetical
    entry at candle t+1 open. The following HORIZON candles
    determine the hypothetical outcome.

    All future-candle inspection stays inside the same segment.
    """

    filename = symbol.replace("/", "_") + "_1h.parquet"
    input_path = INPUT_DIR / filename
    output_path = OUTPUT_DIR / filename

    print("\n" + "=" * 65)
    print(f"Building targets: {symbol}")
    print("=" * 65)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {input_path}"
        )

    df = pd.read_parquet(input_path)

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "atr_14",
        "segment_id",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{symbol} is missing columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="raise",
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    df["target_long_net_return"] = np.nan
    df["target_short_net_return"] = np.nan
    df["target"] = np.nan

    for _, segment in df.groupby("segment_id", sort=False):
        indices = segment.index.to_list()

        # Need one entry candle plus HORIZON future candles.
        for position in range(len(indices) - HORIZON):
            current_index = indices[position]
            entry_index = indices[position + 1]

            future_indices = indices[
                position + 1:position + HORIZON + 1
            ]

            current_atr = df.at[current_index, "atr_14"]

            if (
                pd.isna(current_atr)
                or not np.isfinite(current_atr)
                or current_atr <= 0
            ):
                continue

            entry_price = df.at[entry_index, "open"]

            if (
                pd.isna(entry_price)
                or not np.isfinite(entry_price)
                or entry_price <= 0
            ):
                continue

            stop_distance = (
                STOP_ATR_MULTIPLIER * current_atr
            )

            target_distance = (
                TARGET_ATR_MULTIPLIER * current_atr
            )

            future_candles = df.loc[
                future_indices,
                ["open", "high", "low", "close"],
            ]

            long_return = evaluate_trade(
                future_candles,
                entry_price,
                stop_distance,
                target_distance,
                direction=1,
            )

            short_return = evaluate_trade(
                future_candles,
                entry_price,
                stop_distance,
                target_distance,
                direction=-1,
            )

            df.at[
                current_index, "target_long_net_return"
            ] = long_return

            df.at[
                current_index, "target_short_net_return"
            ] = short_return

            # Select a direction only when its simulated
            # net return is positive and exceeds the other side.
            if long_return > 0 and short_return <= 0:
                df.at[current_index, "target"] = 1

            elif short_return > 0 and long_return <= 0:
                df.at[current_index, "target"] = -1

            else:
                df.at[current_index, "target"] = 0

    # Keep the target nullable so unavailable labels are not
    # accidentally treated as NO TRADE.
    df["target"] = df["target"].astype("Int8")

    # Save to a separate file to preserve the original features.
    target_filename = (
        symbol.replace("/", "_") + "_1h_targets.parquet"
    )

    target_path = OUTPUT_DIR / target_filename

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(target_path, index=False)

    labelled = df["target"].dropna().astype(int)
    counts = labelled.value_counts().to_dict()

    result = {
        "symbol": symbol,
        "rows": len(df),
        "labelled_rows": len(labelled),
        "unlabelled_rows": int(df["target"].isna().sum()),
        "long_count": int(counts.get(1, 0)),
        "short_count": int(counts.get(-1, 0)),
        "no_trade_count": int(counts.get(0, 0)),
        "output_file": str(
            target_path.relative_to(PROJECT_ROOT)
        ),
        "status": "PASS",
    }

    print(f"Rows: {result['rows']:,}")
    print(f"Labelled rows: {result['labelled_rows']:,}")
    print(f"Unlabelled rows: {result['unlabelled_rows']:,}")
    print(f"LONG labels: {result['long_count']:,}")
    print(f"SHORT labels: {result['short_count']:,}")
    print(f"NO TRADE labels: {result['no_trade_count']:,}")
    print(f"Output: {target_path}")

    return result


# --------------------------------------------------
# Main execution
# --------------------------------------------------

def main():
    results = []

    for symbol in SYMBOLS:
        results.append(build_targets(symbol))

    report = pd.DataFrame(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / "target_build_report.csv"

    report.to_csv(report_path, index=False)

    print("\n" + "=" * 65)
    print("TARGET BUILD SUMMARY")
    print("=" * 65)

    print(report.to_string(index=False))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
