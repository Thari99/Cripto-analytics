
import ccxt
import pandas as pd


SYMBOL = "BTC/USDT"
TARGET = pd.Timestamp("2023-03-24 13:00:00", tz="UTC")
TARGET_MS = int(TARGET.timestamp() * 1000)

exchange = ccxt.binance({
    "enableRateLimit": True,
    "timeout": 30000,
})


def show_candles(label, candles):
    print(f"\n--- {label} ---")

    if not candles:
        print("No candles returned.")
        return

    for candle in candles:
        timestamp = pd.to_datetime(
            candle[0],
            unit="ms",
            utc=True,
        )

        print(
            f"{timestamp} | "
            f"O={candle[1]} "
            f"H={candle[2]} "
            f"L={candle[3]} "
            f"C={candle[4]} "
            f"V={candle[5]}"
        )


try:
    exchange.load_markets()

    print("Exchange:", exchange.id)
    print("Symbol:", SYMBOL)
    print("Target:", TARGET)
    print("Target milliseconds:", TARGET_MS)

    # Test 1: Request from one hour before the target.
    before = exchange.fetch_ohlcv(
        SYMBOL,
        timeframe="1h",
        since=TARGET_MS - 60 * 60 * 1000,
        limit=5,
    )

    show_candles("Request starting one hour before", before)

    # Test 2: Request the target with a larger result limit.
    target_request = exchange.fetch_ohlcv(
        SYMBOL,
        timeframe="1h",
        since=TARGET_MS,
        limit=5,
    )

    show_candles("Request starting at target", target_request)

    # Test 3: Request a range starting two hours before.
    earlier = exchange.fetch_ohlcv(
        SYMBOL,
        timeframe="1h",
        since=TARGET_MS - 2 * 60 * 60 * 1000,
        limit=5,
    )

    show_candles("Request starting two hours before", earlier)

finally:
    exchange.close()
