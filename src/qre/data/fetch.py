#!/usr/bin/env python3
"""
Data Fetching
=============
OHLCV data loading: local Parquet dataset first, Binance API fallback.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from qre.config import (
    BASE_TF,
    MAX_API_RETRIES,
    MIN_WARMUP_BARS,
    OHLCV_LIMIT_PER_CALL,
    SAFETY_MAX_ROWS,
    TF_MS,
    TREND_TFS,
)

logger = logging.getLogger("qre.data")

# Default dataset path (centralized OHLCV store)
DATASET_PATH = Path.home() / "projects" / "dataset" / "data"


def load_from_dataset(
    symbol: str,
    tf: str,
    dataset_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """
    Load OHLCV from local Parquet dataset.

    Args:
        symbol: Trading pair (e.g. "BTC/USDT")
        tf: Timeframe (e.g. "1h", "4h")
        dataset_path: Override dataset directory (for testing)

    Returns:
        DataFrame with OHLCV data, or None if file doesn't exist.
    """
    if dataset_path is None:
        dataset_path = DATASET_PATH

    # BTC/USDT -> BTCUSDT
    dir_name = symbol.replace("/", "")
    parquet_path = dataset_path / dir_name / f"{tf}.parquet"

    if not parquet_path.exists():
        return None

    logger.info("Loading %s %s from dataset: %s", symbol, tf, parquet_path)
    df = pd.read_parquet(parquet_path)

    # Ensure UTC DatetimeIndex named "timestamp"
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    return df


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
    all_rows: list[list] = []
    tf_ms = TF_MS[tf]
    cursor = since_ms
    retry_count = 0

    logger.info("Fetching %s %s from %s", symbol, tf, datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc))

    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=tf, since=cursor, limit=OHLCV_LIMIT_PER_CALL)
            retry_count = 0

        except Exception as e:
            retry_count += 1
            logger.warning("Fetch error %s %s (attempt %d/%d): %s", symbol, tf, retry_count, MAX_API_RETRIES, e)

            if retry_count >= MAX_API_RETRIES:
                logger.error("Max retries reached for %s %s", symbol, tf)
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
            logger.warning("Safety limit reached for %s %s", symbol, tf)
            break

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    logger.info("Fetched %d rows for %s %s", len(df), symbol, tf)

    return df


def load_all_data(exchange, symbol: str, hours_1h: int) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV data for base TF + trend TFs.

    Tries local Parquet dataset first, falls back to Binance API.

    Args:
        exchange: ccxt exchange (used for API fallback)
        symbol: Trading pair (e.g. "BTC/USDT")
        hours_1h: How many hours of 1h data to load

    Returns:
        Dict {timeframe: DataFrame} — keys: "1h", "4h", "8h", "1d"
    """
    now_ms = utcnow_ms()
    since_1h = now_ms - hours_1h * TF_MS["1h"]

    data: Dict[str, pd.DataFrame] = {}
    all_tfs = [BASE_TF] + list(TREND_TFS)

    for tf in all_tfs:
        # Try dataset first
        df = load_from_dataset(symbol, tf)
        if df is not None:
            # Trim to requested time range
            since_dt = pd.Timestamp(since_1h, unit="ms", tz="UTC")
            df = df[df.index >= since_dt]
            if len(df) > 0:
                logger.info(
                    "Using dataset for %s %s (%d rows)", symbol, tf, len(df),
                )
                data[tf] = df
                continue

        # Fallback to API
        logger.info("Falling back to API for %s %s", symbol, tf)
        data[tf] = fetch_ohlcv_paginated(exchange, symbol, tf, since_1h, now_ms)

    return data
