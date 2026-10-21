
import requests
import pandas as pd


BASE_URL = "https://api.binance.com/api/v3/klines"

SYMBOL = "BTCUSDT"
INTERVAL = "1h"

TARGET = pd.Timestamp(
    "2023-03-24 13:00:00",
    tz="UTC",
)

TARGET_MS = int(TARGET.timestamp() * 1000)

params = {
    "symbol": SYMBOL,
    "interval": INTERVAL,
    "startTime": TARGET_MS - 2 * 60 * 60 * 1000,
    "endTime": TARGET_MS + 3 * 60 * 60 * 1000,
    "limit": 10,
}

try:
    response = requests.get(
        BASE_URL,
        params=params,
        timeout=30,
    )

    print("HTTP status:", response.status_code)
    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict):
        print("API error:", data)
    else:
        print(f"Returned candles: {len(data)}")

        for candle in data:
            timestamp = pd.to_datetime(
                candle[0],
                unit="ms",
                utc=True,
            )

            print(
                f"Time: {timestamp} | "
                f"Open: {candle[1]} | "
                f"High: {candle[2]} | "
                f"Low: {candle[3]} | "
                f"Close: {candle[4]} | "
                f"Volume: {candle[5]}"
            )

        target_found = any(
            int(candle[0]) == TARGET_MS
            for candle in data
        )

        print(
            "\nTarget candle returned:",
            target_found,
        )

except requests.RequestException as error:
    print(f"Request failed: {error}")