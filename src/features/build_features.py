
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
]

PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


# --------------------------------------------------
# Feature calculation
# --------------------------------------------------

def build_features(symbol: str) -> dict:
    """
    Build basic price-return and candle-structure features.

    Calculations are performed independently within each
    continuous segment to prevent returns crossing time gaps.
    """

    filename = symbol.replace("/", "_") + "_1h.parquet"
    input_path = INPUT_DIR / filename

    output_path = OUTPUT_DIR / filename

    print("\n" + "=" * 65)
    print(f"Building features: {symbol}")
    print("=" * 65)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {input_path}"
        )

    df = pd.read_parquet(input_path)

    required_columns = [
        "timestamp",
        *PRICE_COLUMNS,
        "segment_id",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{symbol} is missing columns: {missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="raise",
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Make sure each continuous segment is processed independently.
    grouped = df.groupby("segment_id", sort=False)

    # --------------------------------------------------
    # 1. Price returns
    # --------------------------------------------------

    for period in [1, 4, 24]:
        df[f"return_{period}h"] = grouped["close"].transform(
            lambda series: series.pct_change(
                periods=period,
                fill_method=None,
            )
        )

    # --------------------------------------------------
    # 2. Candle structure
    # --------------------------------------------------

    candle_range = df["high"] - df["low"]

    # Candle body relative to the low price.
    df["candle_body_pct"] = (
        (df["close"] - df["open"]) / df["open"]
    )

    # Upper and lower wick lengths relative to the candle range.
    safe_range = candle_range.replace(0, np.nan)

    df["upper_wick_pct"] = (
        df["high"] - df[["open", "close"]].max(axis=1)
    ) / safe_range

    df["lower_wick_pct"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    ) / safe_range

    # Total candle range relative to the opening price.
    df["candle_range_pct"] = (
        candle_range / df["open"]
    )

    # Direction flags.
    df["is_bullish_candle"] = (
        df["close"] > df["open"]
    ).astype("int8")

    df["is_bearish_candle"] = (
        df["close"] < df["open"]
    ).astype("int8")

   
    # --------------------------------------------------
    # 3. Momentum and trend indicators
    # --------------------------------------------------

    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI using Wilder's smoothing method.

        Requires period valid price changes before producing
        the first RSI value.
        """

        delta = close.diff()

        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gain = gains.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False,
        ).mean()

        avg_loss = losses.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False,
        ).mean()

        relative_strength = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + relative_strength))

        # Handle periods with no losses.
        rsi = rsi.mask(
            (avg_loss == 0) & (avg_gain > 0),
            100.0,
        )

        # If neither gains nor losses occurred, momentum is neutral.
        rsi = rsi.mask(
            (avg_loss == 0) & (avg_gain == 0),
            50.0,
        )

        return rsi

    # Calculate every indicator separately for each segment.
    # This prevents rolling calculations from crossing time gaps.

    for period in [20, 50]:
        df[f"sma_{period}"] = grouped["close"].transform(
            lambda series: series.rolling(
                window=period,
                min_periods=period,
            ).mean()
        )

    for period in [12, 26]:
        df[f"ema_{period}"] = grouped["close"].transform(
            lambda series: series.ewm(
                span=period,
                min_periods=period,
                adjust=False,
            ).mean()
        )

    df["rsi_14"] = grouped["close"].transform(
        lambda series: calculate_rsi(series, period=14)
    )

    # MACD = short EMA - long EMA.
    df["macd"] = df["ema_12"] - df["ema_26"]

    # Calculate the signal line independently within each segment.
    df["macd_signal"] = (
        df.groupby("segment_id", sort=False)["macd"]
        .transform(
            lambda series: series.ewm(
                span=9,
                min_periods=9,
                adjust=False,
            ).mean()
        )
    )

    df["macd_histogram"] = (
        df["macd"] - df["macd_signal"]
    )

    # Normalize moving-average relationships by price.
    # This makes the features more comparable across different
    # price levels and different cryptocurrency pairs.

    df["price_vs_sma_20"] = (
        df["close"] / df["sma_20"] - 1
    )

    df["price_vs_sma_50"] = (
        df["close"] / df["sma_50"] - 1
    )

    df["ema_trend_difference"] = (
        df["ema_12"] / df["ema_26"] - 1
    )


    # --------------------------------------------------
    # 4. ATR and volatility features
    # --------------------------------------------------

    previous_close = grouped["close"].shift(1)

    range_high_low = df["high"] - df["low"]

    range_high_previous_close = (
        df["high"] - previous_close
    ).abs()

    range_low_previous_close = (
        df["low"] - previous_close
    ).abs()

    df["true_range"] = pd.concat(
        [
            range_high_low,
            range_high_previous_close,
            range_low_previous_close,
        ],
        axis=1,
    ).max(axis=1)

    # First candle of each segment: use high - low.
    df.loc[previous_close.isna(), "true_range"] = (
        range_high_low.loc[previous_close.isna()]
    )

    df["atr_14"] = (
        df.groupby("segment_id", sort=False)["true_range"]
        .transform(
            lambda s: s.rolling(
                window=14,
                min_periods=14,
            ).mean()
        )
    )

    df["atr_pct"] = df["atr_14"] / df["close"]

    for period in [24, 72, 168]:
        df[f"volatility_{period}h"] = (
            df.groupby("segment_id", sort=False)["return_1h"]
            .transform(
                lambda s: s.rolling(
                    window=period,
                    min_periods=period,
                ).std()
            )
        )

    # --------------------------------------------------
    # 5. Volume analysis features
    # --------------------------------------------------

    for period in [20, 50]:
        df[f"volume_sma_{period}"] = (
            df.groupby("segment_id", sort=False)["volume"]
            .transform(
                lambda s: s.rolling(
                    window=period,
                    min_periods=period,
                ).mean()
            )
        )

    df["relative_volume_20"] = (
        df["volume"]
        / df["volume_sma_20"].replace(0, float("nan"))
    )

    df["relative_volume_50"] = (
        df["volume"]
        / df["volume_sma_50"].replace(0, float("nan"))
    )

    previous_volume = grouped["volume"].shift(1)

    df["volume_change_1h"] = (
        (df["volume"] - previous_volume)
        / previous_volume.replace(0, float("nan"))
    )

    # 1 means volume exceeds 1.5 times its 20-candle average.
    # 0 means no spike is detected or insufficient history exists.
    df["volume_spike"] = (
        df["relative_volume_20"] > 1.5
    ).astype("int8")

    df["price_volume_direction"] = 0

    high_volume = df["relative_volume_20"] > 1.0

    df.loc[
        (df["return_1h"] > 0) & high_volume,
        "price_volume_direction",
    ] = 1

    df.loc[
        (df["return_1h"] < 0) & high_volume,
        "price_volume_direction",
    ] = -1
    df.loc[
        (df["return_1h"] < 0) & high_volume,
        "price_volume_direction",
    ] = -1

    # --------------------------------------------------
    # 6. Support and resistance features
    # --------------------------------------------------

    support_resistance_windows = {
        "24h": 24,
        "7d": 168,
        "30d": 720,
        "90d": 2160,
    }

    for label, window in support_resistance_windows.items():

        df[f"support_{label}"] = (
            df.groupby("segment_id", sort=False)["low"]
            .transform(
                lambda s: s.shift(1).rolling(
                    window=window,
                    min_periods=window,
                ).min()
            )
        )

        df[f"resistance_{label}"] = (
            df.groupby("segment_id", sort=False)["high"]
            .transform(
                lambda s: s.shift(1).rolling(
                    window=window,
                    min_periods=window,
                ).max()
            )
        )

        df[f"distance_to_support_{label}"] = (
            (df["close"] - df[f"support_{label}"])
            / df["close"]
        )

        df[f"distance_to_resistance_{label}"] = (
            (df[f"resistance_{label}"] - df["close"])
            / df["close"]
        )
    # --------------------------------------------------
    # 6. Basic output checks
    # --------------------------------------------------

    # --------------------------------------------------
    # 5. Basic output checks
    # --------------------------------------------------

    feature_columns = [
        # Price returns
        "return_1h",
        "return_4h",
        "return_24h",

        # Candle structure
        "candle_body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "candle_range_pct",
        "is_bullish_candle",
        "is_bearish_candle",

        # Momentum
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",

        # Moving averages and trend
        "sma_20",
        "sma_50",
        "ema_12",
        "ema_26",
        "price_vs_sma_20",
        "price_vs_sma_50",
        "ema_trend_difference",

        # ATR and volatility
        "true_range",
        "atr_14",
        "atr_pct",
        "volatility_24h",
        "volatility_72h",
        "volatility_168h",

        # Volume analysis
        "volume_sma_20",
        "volume_sma_50",
        "relative_volume_20",
        "relative_volume_50",
        "volume_change_1h",
        "volume_spike",
        "price_volume_direction",
    ]



    infinite_count = int(
        np.isinf(df[feature_columns].to_numpy(dtype=float)).sum()
    )

    # NaNs at the beginning of each segment are expected:
    # the model cannot calculate a 24-hour return without
    # 24 previous observations in that same segment.
    feature_nan_rows = int(
        df[feature_columns].isna().any(axis=1).sum()
    )

    # A zero-range candle has undefined wick percentages.
    zero_range_count = int(
        (candle_range == 0).sum()
    )

    # --------------------------------------------------
    # 4. Save feature dataset
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    # --------------------------------------------------
    # 5. Print summary
    # --------------------------------------------------

    print(f"Rows: {len(df):,}")
    print(f"Continuous segments: {df['segment_id'].nunique()}")
    print(f"Infinite feature values: {infinite_count}")
    print(f"Rows with at least one NaN feature: {feature_nan_rows}")
    print(f"Zero-range candles: {zero_range_count}")
    print(f"Output file: {output_path}")

    if infinite_count != 0:
        raise ValueError(
            f"Infinite features detected for {symbol}"
        )

    return {
        "symbol": symbol,
        "rows": len(df),
        "segment_count": int(df["segment_id"].nunique()),
        "infinite_feature_values": infinite_count,
        "rows_with_nan_features": feature_nan_rows,
        "zero_range_candles": zero_range_count,
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
            build_features(symbol)
        )

    report = pd.DataFrame(results)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORTS_DIR / "feature_build_report.csv"
    )

    report.to_csv(
        report_path,
        index=False,
    )

    print("\n" + "=" * 65)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 65)

    print(
        report[
            [
                "symbol",
                "rows",
                "segment_count",
                "infinite_feature_values",
                "rows_with_nan_features",
                "status",
            ]
        ].to_string(index=False)
    )

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
