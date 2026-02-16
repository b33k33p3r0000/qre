#!/usr/bin/env python3
"""
Data Fetching
=============
Stahování OHLCV dat z Binance API. Vždy fresh data, žádný disk cache.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

from qre.config import (
    BASE_TF,
    MAX_API_RETRIES,
    MIN_WARMUP_BARS,
    OHLCV_LIMIT_PER_CALL,
    SAFETY_MAX_ROWS,
    TF_MS,
)

logger = logging.getLogger("qre.data")


def utcnow_ms() -> int:
    """Vrátí aktuální čas jako timestamp v ms."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def fetch_ohlcv_paginated(
    exchange, symbol: str, tf: str, since_ms: int, until_ms: int,
) -> pd.DataFrame:
    """
    Stáhne OHLCV data z burzy po částech.

    Args:
        exchange: ccxt exchange
        symbol: Trading pár
        tf: Timeframe
        since_ms: Od kdy (ms)
        until_ms: Do kdy (ms)

    Returns:
        DataFrame s OHLCV daty
    """
    all_rows: List[list] = []
    tf_ms = TF_MS[tf]
    cursor = since_ms
    retry_count = 0

    logger.info(f"Fetching {symbol} {tf} from {datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)}")

    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=tf, since=cursor, limit=OHLCV_LIMIT_PER_CALL)
            retry_count = 0

        except Exception as e:
            retry_count += 1
            logger.warning(f"Fetch error {symbol} {tf} (attempt {retry_count}/{MAX_API_RETRIES}): {e}")

            if retry_count >= MAX_API_RETRIES:
                logger.error(f"Max retries reached for {symbol} {tf}")
                break

            time.sleep(exchange.rateLimit / 1000.0 + 1.0)
            continue

        if not batch:
            break

        all_rows.extend(batch)
        last_ts = batch[-1][0]

        if last_ts >= until_ms - tf_ms:
            break

        next_cursor = last_ts + tf_ms
        if next_cursor <= cursor:
            next_cursor = cursor + tf_ms
        cursor = next_cursor

        time.sleep(exchange.rateLimit / 1000.0)

        if len(all_rows) > SAFETY_MAX_ROWS:
            logger.warning(f"Safety limit reached for {symbol} {tf}")
            break

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    logger.info(f"Fetched {len(df)} rows for {symbol} {tf}")

    return df


def load_all_data(exchange, symbol: str, hours_1h: int) -> Dict[str, pd.DataFrame]:
    """
    Načte fresh data z Binance pro všechny timeframy.

    Args:
        exchange: ccxt exchange
        symbol: Trading pár
        hours_1h: Kolik hodin zpětně

    Returns:
        Dict {timeframe: DataFrame}
    """
    now_ms = utcnow_ms()
    since_1h = now_ms - hours_1h * TF_MS["1h"]

    data: Dict[str, pd.DataFrame] = {}

    # Base timeframe only (Chio Extreme uses single TF)
    data[BASE_TF] = fetch_ohlcv_paginated(exchange, symbol, BASE_TF, since_1h, now_ms)

    return data
