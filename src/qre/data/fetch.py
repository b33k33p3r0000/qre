#!/usr/bin/env python3
"""
Data Fetching & Caching
=======================
Stahování OHLCV dat s SQLite/Parquet cache.

v4.0 NEW:
- Parquet cache pro rychlé načítání
- Inkrementální update
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from qre.config import (
    BASE_TF,
    MAX_API_RETRIES,
    MIN_WARMUP_BARS,
    OHLCV_LIMIT_PER_CALL,
    SAFETY_MAX_ROWS,
    TF_LIST,
    TF_MS,
)

logger = logging.getLogger("qre.data")


class DataCache:
    """
    v4.0 NEW: Parquet-based cache pro OHLCV data.

    Šetří čas při opakovaných bězích - nemusí stahovat znovu.
    """

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, symbol: str, tf: str) -> Path:
        """Vrátí cestu k cache souboru."""
        symbol_safe = symbol.replace("/", "_")
        return self.cache_dir / f"{symbol_safe}_{tf}.parquet"

    def load(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        """Načte data z cache pokud existuje."""
        path = self.get_cache_path(symbol, tf)
        if path.exists():
            try:
                df = pd.read_parquet(path)
                logger.debug(f"Cache hit: {symbol} {tf} ({len(df)} rows)")
                return df
            except Exception as e:
                logger.warning(f"Cache read error for {symbol} {tf}: {e}")
                return None
        return None

    def save(self, symbol: str, tf: str, df: pd.DataFrame) -> None:
        """Uloží data do cache (atomic write pro race condition safety)."""
        if df.empty:
            return

        path = self.get_cache_path(symbol, tf)
        # Atomic write: zapsat do temp souboru, pak přejmenovat
        # Tím se vyhneme race condition při paralelním běhu
        import os
        import tempfile

        try:
            # Zapsat do temp souboru ve stejném adresáři
            fd, tmp_path = tempfile.mkstemp(
                suffix=".parquet", dir=self.cache_dir, prefix=f".tmp_{symbol.replace('/', '_')}_{tf}_"
            )
            os.close(fd)

            df.to_parquet(tmp_path, index=True)

            # Atomic rename (na stejném filesystem je atomic)
            os.replace(tmp_path, path)
            logger.debug(f"Cache saved: {symbol} {tf} ({len(df)} rows)")
        except Exception as e:
            logger.warning(f"Cache write error for {symbol} {tf}: {e}")
            # Cleanup temp souboru pokud existuje
            try:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def get_last_timestamp(self, symbol: str, tf: str) -> Optional[int]:
        """Vrátí poslední timestamp v cache (pro inkrementální update)."""
        df = self.load(symbol, tf)
        if df is not None and len(df) > 0:
            last_ts = df.index[-1]
            if isinstance(last_ts, pd.Timestamp):
                return int(last_ts.timestamp() * 1000)
        return None

    def update(self, symbol: str, tf: str, new_df: pd.DataFrame) -> pd.DataFrame:
        """Přidá nová data k existujícím v cache."""
        existing = self.load(symbol, tf)

        if existing is None or existing.empty:
            self.save(symbol, tf, new_df)
            return new_df

        # Sloučíme a odstraníme duplikáty
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        self.save(symbol, tf, combined)
        return combined

    def clear(self, symbol: Optional[str] = None):
        """Vymaže cache (pro symbol nebo celou)."""
        if symbol:
            for tf in [BASE_TF] + TF_LIST:
                path = self.get_cache_path(symbol, tf)
                if path.exists():
                    path.unlink()
                    logger.info(f"Cache cleared: {symbol} {tf}")
        else:
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()
            logger.info("All cache cleared")


def utcnow_ms() -> int:
    """Vrátí aktuální čas jako timestamp v ms."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def fetch_ohlcv_paginated(
    exchange, symbol: str, tf: str, since_ms: int, until_ms: int, cache: Optional[DataCache] = None
) -> pd.DataFrame:
    """
    Stáhne OHLCV data z burzy po částech.

    v4.0: Podpora pro cache (inkrementální stahování)

    Args:
        exchange: ccxt exchange
        symbol: Trading pár
        tf: Timeframe
        since_ms: Od kdy
        until_ms: Do kdy
        cache: Volitelná DataCache instance

    Returns:
        DataFrame s OHLCV daty
    """
    # Zkusíme cache
    if cache:
        cached_df = cache.load(symbol, tf)
        if cached_df is not None and len(cached_df) > 0:
            last_ts = cache.get_last_timestamp(symbol, tf)
            if last_ts and last_ts >= until_ms - TF_MS[tf]:
                # Cache je aktuální - ořezat na požadovaný rozsah
                since_dt = pd.Timestamp(since_ms, unit="ms", tz="UTC")
                filtered_df = cached_df[cached_df.index >= since_dt]
                logger.info(f"Using cached data for {symbol} {tf} ({len(filtered_df)} rows from {since_dt})")
                return filtered_df
            else:
                # Potřebujeme stáhnout jen nová data
                since_ms = max(since_ms, last_ts + TF_MS[tf])
                logger.info(f"Incremental update for {symbol} {tf} from {datetime.fromtimestamp(since_ms / 1000)}")

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

    # Uložíme do cache
    if cache:
        df = cache.update(symbol, tf, df)

    return df


def load_all_data(exchange, symbol: str, hours_1h: int, cache: Optional[DataCache] = None) -> Dict[str, pd.DataFrame]:
    """
    Načte data pro všechny timeframy.

    v4.0: Podpora pro cache

    Args:
        exchange: ccxt exchange
        symbol: Trading pár
        hours_1h: Kolik hodin zpětně
        cache: Volitelná DataCache

    Returns:
        Dict {timeframe: DataFrame}
    """
    now_ms = utcnow_ms()
    since_1h = now_ms - hours_1h * TF_MS["1h"]

    data: Dict[str, pd.DataFrame] = {}

    # Base timeframe
    data[BASE_TF] = fetch_ohlcv_paginated(exchange, symbol, BASE_TF, since_1h, now_ms, cache)

    # Higher timeframes
    span_hours = max(MIN_WARMUP_BARS, len(data[BASE_TF]))

    for tf in TF_LIST:
        since_tf = now_ms - (span_hours + MIN_WARMUP_BARS) * TF_MS["1h"]
        data[tf] = fetch_ohlcv_paginated(exchange, symbol, tf, since_tf, now_ms, cache)

    return data
