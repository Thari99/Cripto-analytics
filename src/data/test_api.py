
import ccxt
from datetime import datetime, timezone

# Connect to the public Binance API.
exchange = ccxt.binance({
    "enableRateLimit": True,
})

try:
    # Load available trading markets.
    exchange.load_markets()

    symbol = "BTC/USDT"

    # Check that the market exists.
    if symbol not in exchange.markets:
        raise ValueError(f"{symbol} is not available")

    # Request the latest 5 hourly candles.
    candles = exchange.fetch_ohlcv(
        symbol,
        timeframe="1h",
        limit=5,
    )

    print(f"Exchange: {exchange.id}")
    print(f"Symbol: {symbol}")
    print(f"Candles received: {len(candles)}")

    for candle in candles:
        timestamp, open_price, high, low, close, volume = candle

        readable_time = datetime.fromtimestamp(
            timestamp / 1000,
            tz=timezone.utc,
        )

        print(
            f"Time: {readable_time} | "
            f"Open: {open_price} | "
            f"High: {high} | "
            f"Low: {low} | "
            f"Close: {close} | "
            f"Volume: {volume}"
        )

except ccxt.BaseError as error:
    print(f"API request failed: {type(error).__name__}: {error}")

finally:
    exchange.close()
