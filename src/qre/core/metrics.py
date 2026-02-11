#!/usr/bin/env python3
"""
Metrics Calculation
===================
Rozšířené metriky pro hodnocení strategie.

v4.0 NEW:
- Calmar Ratio
- Sortino Ratio
- Recovery Factor
- Max Loss Streak
- Profit per Bar
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Suppress pandas timezone warning for period conversion
warnings.filterwarnings(
    "ignore",
    message="Converting to PeriodArray/Index representation will drop timezone information",
)

from qre.config import MIN_HOLD_BARS, STARTING_EQUITY


@dataclass
class MonteCarloResult:
    """
    v9.0 NEW: Výsledky Monte Carlo validace.

    Shuffluje obchody pro odhad confidence intervals a robustnosti.
    """

    # Sharpe ratio statistics
    sharpe_mean: float
    sharpe_std: float
    sharpe_ci_low: float  # 2.5th percentile (95% CI lower)
    sharpe_ci_high: float  # 97.5th percentile (95% CI upper)

    # Max drawdown statistics
    max_dd_mean: float
    max_dd_std: float
    max_dd_ci_low: float  # 2.5th percentile (best case)
    max_dd_ci_high: float  # 97.5th percentile (worst case)

    # Win rate statistics
    win_rate_mean: float
    win_rate_ci_low: float
    win_rate_ci_high: float

    # Confidence assessment
    confidence_level: str  # "HIGH", "MEDIUM", "LOW"
    n_simulations: int

    # Robustness score (0-1, higher = more robust)
    robustness_score: float


@dataclass
class MetricsResult:
    """Výsledky metrické analýzy."""

    # Basic
    equity: float
    total_pnl: float
    total_pnl_pct: float
    trades: int
    trades_per_year: float

    # Win/Loss
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float

    # Risk
    max_drawdown: float
    avg_drawdown: float
    sharpe_ratio: float  # Trade-based (legacy, pro zpětnou kompatibilitu)
    sharpe_ratio_time_based: float  # v12.0 NEW: Time-based Sharpe (realistický)
    sortino_ratio: float  # v4.0 NEW
    calmar_ratio: float  # v4.0 NEW
    recovery_factor: float  # v4.0 NEW

    # Trade quality
    profit_factor: float
    expectancy: float
    avg_hold_bars: float
    short_hold_ratio: float
    profit_per_bar: float  # v4.0 NEW

    # Streaks
    max_win_streak: int  # v4.0 NEW
    max_loss_streak: int  # v4.0 NEW

    # Monthly
    profitable_months_ratio: float  # v4.0 NEW
    monthly_returns: List[float]  # v4.0 NEW


def calculate_annualized_trades(trades: List[Dict], backtest_days: int) -> float:
    """Spočítá průměrný počet obchodů za rok."""
    if not trades or backtest_days < 1:
        return 0.0
    years = backtest_days / 365.25
    return len(trades) / years if years > 0 else 0.0


def calculate_short_hold_ratio(trades: List[Dict], min_hold: int = MIN_HOLD_BARS) -> float:
    """Spočítá poměr obchodů s krátkým držením."""
    if not trades:
        return 0.0
    short_trades = sum(1 for t in trades if t.get("hold_bars", 0) <= min_hold)
    return short_trades / len(trades)


def calculate_streaks(trades: List[Dict]) -> tuple:
    """
    v4.0 NEW: Spočítá max win/loss streak.

    Returns:
        (max_win_streak, max_loss_streak)
    """
    if not trades:
        return 0, 0

    max_win = 0
    max_loss = 0
    current_win = 0
    current_loss = 0

    for trade in trades:
        if trade.get("pnl_abs", 0) > 0:
            current_win += 1
            current_loss = 0
            max_win = max(max_win, current_win)
        else:
            current_loss += 1
            current_win = 0
            max_loss = max(max_loss, current_loss)

    return max_win, max_loss


def calculate_monthly_returns(trades: List[Dict]) -> List[float]:
    """
    v4.0 NEW: Spočítá měsíční výnosy.
    """
    if not trades:
        return []

    df = pd.DataFrame(trades)
    if df.empty or "exit_ts" not in df.columns:
        return []

    df["exit_ts"] = pd.to_datetime(df["exit_ts"])
    df["month"] = df["exit_ts"].dt.to_period("M")

    monthly = df.groupby("month")["pnl_abs"].sum().tolist()
    return monthly


def calculate_time_based_sharpe(
    trades: List[Dict],
    price_data: pd.DataFrame,
    start_equity: float,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> float:
    """
    v12.0 NEW: Time-based Sharpe Ratio - počítá z hodinových výnosů včetně idle periods.

    Na rozdíl od trade-based Sharpe, tato metoda:
    - Zahrnuje období kdy strategie drží cash (return = 0%)
    - Dává realistické hodnoty srovnatelné s industry benchmarky (0.5-2.0)
    - Annualizuje správně pro hodinová data (sqrt(8760))

    Args:
        trades: Seznam obchodů z backtestu
        price_data: DataFrame s OHLCV daty (1h timeframe, musí mít 'close' sloupec)
        start_equity: Počáteční kapitál
        start_idx: Počáteční index v price_data
        end_idx: Koncový index v price_data (None = konec)

    Returns:
        Time-based Sharpe ratio (annualizovaný)
    """
    if not trades or price_data.empty:
        return 0.0

    end_idx = end_idx or len(price_data)
    n_hours = end_idx - start_idx

    if n_hours < 100:  # Minimum pro smysluplný výpočet
        return 0.0

    # Inicializace hourly returns array (default = 0 pro idle periods)
    hourly_returns = np.zeros(n_hours)

    # Převod trades na DataFrame pro rychlejší lookup
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return 0.0

    # Parse timestamps
    trades_df["entry_ts"] = pd.to_datetime(trades_df["entry_ts"])
    trades_df["exit_ts"] = pd.to_datetime(trades_df["exit_ts"])

    # Pro každý trade, spočítej hourly returns v době držení pozice
    price_index = price_data.index

    for _, trade in trades_df.iterrows():
        entry_ts = trade["entry_ts"]
        exit_ts = trade["exit_ts"]
        entry_price = trade["entry_price"]

        # Najdi indexy v price_data
        try:
            # Najdi první bar >= entry_ts
            entry_mask = price_index >= entry_ts
            if not entry_mask.any():
                continue
            trade_start_idx = entry_mask.argmax()

            # Najdi první bar >= exit_ts
            exit_mask = price_index >= exit_ts
            if not exit_mask.any():
                trade_end_idx = len(price_index) - 1
            else:
                trade_end_idx = exit_mask.argmax()

            # Relativní indexy vzhledem k start_idx
            rel_start = max(0, trade_start_idx - start_idx)
            rel_end = min(n_hours - 1, trade_end_idx - start_idx)

            if rel_start >= rel_end or rel_start < 0:
                continue

            # Spočítej hourly returns pro tento trade
            prev_price = entry_price
            for i in range(rel_start, rel_end + 1):
                abs_idx = start_idx + i
                if abs_idx >= len(price_data):
                    break

                current_price = float(price_data["close"].iloc[abs_idx])
                if prev_price > 0:
                    hourly_return = (current_price - prev_price) / prev_price
                    # Přičti k hourly_returns (může být více tradů v jednom čase, ale ne v této strategii)
                    hourly_returns[i] = hourly_return

                prev_price = current_price

        except (KeyError, IndexError, ValueError):
            continue

    # Sharpe výpočet
    if len(hourly_returns) < 100:
        return 0.0

    mean_return = np.mean(hourly_returns)
    std_return = np.std(hourly_returns)

    if std_return < 1e-12:
        return 0.0

    # Annualizace pro hodinová data: sqrt(8760) hodin v roce
    sharpe = (mean_return / std_return) * math.sqrt(8760)

    return float(sharpe)


def calculate_sortino_ratio(returns: pd.Series) -> float:
    """
    v4.0 NEW: Sortino Ratio - penalizuje pouze negativní volatilitu.
    """
    if returns.empty:
        return 0.0

    mean_return = returns.mean()
    negative_returns = returns[returns < 0]

    if len(negative_returns) == 0 or negative_returns.std() < 1e-12:
        return 0.0

    downside_std = negative_returns.std()
    return float((mean_return / downside_std) * math.sqrt(252))


def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """
    v4.0 NEW: Calmar Ratio = Annual Return / Max Drawdown
    """
    if abs(max_drawdown) < 1e-12:
        return 0.0
    return total_return / abs(max_drawdown)


def calculate_recovery_factor(total_pnl: float, max_drawdown: float, start_equity: float) -> float:
    """
    v4.0 NEW: Recovery Factor = Net Profit / Max Drawdown (v absolutních hodnotách)
    """
    max_dd_abs = abs(max_drawdown * start_equity / 100)
    if max_dd_abs < 1e-12:
        return 0.0
    return total_pnl / max_dd_abs


def calculate_metrics(
    trades: List[Dict[str, Any]],
    backtest_days: int,
    start_equity: float = STARTING_EQUITY,
    price_data: Optional[pd.DataFrame] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> MetricsResult:
    """
    Spočítá kompletní metriky z obchodů.

    v12.0: Přidán time-based Sharpe ratio (realistický výpočet)
    v4.0: Přidány Calmar, Sortino, Recovery Factor, Streaks, Monthly analysis

    Args:
        trades: Seznam obchodů
        backtest_days: Délka testovacího období v dnech
        start_equity: Počáteční kapitál
        price_data: (v12.0) DataFrame s OHLCV daty pro time-based Sharpe
        start_idx: (v12.0) Počáteční index v price_data
        end_idx: (v12.0) Koncový index v price_data

    Returns:
        MetricsResult s kompletními metrikami
    """
    if not trades:
        return MetricsResult(
            equity=start_equity,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            trades=0,
            trades_per_year=0.0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_drawdown=0.0,
            avg_drawdown=0.0,
            sharpe_ratio=0.0,
            sharpe_ratio_time_based=0.0,  # v12.0 NEW
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            recovery_factor=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            avg_hold_bars=0.0,
            short_hold_ratio=0.0,
            profit_per_bar=0.0,
            max_win_streak=0,
            max_loss_streak=0,
            profitable_months_ratio=0.0,
            monthly_returns=[],
        )

    df = pd.DataFrame(trades)

    # === EQUITY TIMELINE ===
    equity_timeline = [start_equity]
    for _, row in df.iterrows():
        pnl = float(row.get("pnl_abs", 0.0))
        equity_timeline.append(equity_timeline[-1] + pnl)

    final_equity = equity_timeline[-1]
    total_pnl = final_equity - start_equity
    total_pnl_pct = total_pnl / start_equity * 100

    # === DRAWDOWN ===
    equity_series = pd.Series(equity_timeline)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = float(drawdown.min())
    avg_drawdown = float(drawdown[drawdown < 0].mean()) if len(drawdown[drawdown < 0]) > 0 else 0.0

    # === RETURNS ===
    returns = equity_series.pct_change().dropna()

    # Sharpe Ratio (legacy - trade-based)
    sharpe = 0.0
    if not returns.empty and returns.std() > 1e-12:
        sharpe = float((returns.mean() / returns.std()) * math.sqrt(252))

    # v12.0 NEW: Time-based Sharpe Ratio (realistický)
    sharpe_time_based = 0.0
    if price_data is not None and not price_data.empty:
        sharpe_time_based = calculate_time_based_sharpe(trades, price_data, start_equity, start_idx, end_idx)

    # Sortino Ratio (v4.0 NEW)
    sortino = calculate_sortino_ratio(returns)

    # === WIN/LOSS ANALYSIS ===
    winners = [t for t in trades if t.get("pnl_abs", 0) > 0]
    losers = [t for t in trades if t.get("pnl_abs", 0) <= 0]

    win_rate = len(winners) / len(trades) * 100 if trades else 0.0

    avg_win = np.mean([t.get("pnl_abs", 0) for t in winners]) if winners else 0.0
    avg_loss = abs(np.mean([t.get("pnl_abs", 0) for t in losers])) if losers else 0.0

    # Profit Factor
    gross_profit = sum(t.get("pnl_abs", 0) for t in winners)
    gross_loss = abs(sum(t.get("pnl_abs", 0) for t in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy
    expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)

    # === TRADE QUALITY ===
    trades_per_year = calculate_annualized_trades(trades, backtest_days)
    avg_hold_bars = df["hold_bars"].mean() if "hold_bars" in df.columns else 0.0
    short_hold_ratio = calculate_short_hold_ratio(trades)

    # Profit per Bar (v4.0 NEW)
    total_hold_bars = df["hold_bars"].sum() if "hold_bars" in df.columns else 1
    profit_per_bar = total_pnl / total_hold_bars if total_hold_bars > 0 else 0.0

    # === v4.0 NEW METRICS ===

    # Calmar Ratio
    calmar = calculate_calmar_ratio(total_pnl_pct, max_drawdown)

    # Recovery Factor
    recovery = calculate_recovery_factor(total_pnl, max_drawdown, start_equity)

    # Streaks
    max_win_streak, max_loss_streak = calculate_streaks(trades)

    # Monthly Returns
    monthly_returns = calculate_monthly_returns(trades)
    profitable_months = sum(1 for r in monthly_returns if r > 0)
    profitable_months_ratio = profitable_months / len(monthly_returns) if monthly_returns else 0.0

    return MetricsResult(
        equity=float(final_equity),
        total_pnl=float(total_pnl),
        total_pnl_pct=float(total_pnl_pct),
        trades=len(trades),
        trades_per_year=float(trades_per_year),
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate=float(win_rate),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        max_drawdown=float(max_drawdown),
        avg_drawdown=float(avg_drawdown),
        sharpe_ratio=float(sharpe),
        sharpe_ratio_time_based=float(sharpe_time_based),  # v12.0 NEW
        sortino_ratio=float(sortino),
        calmar_ratio=float(calmar),
        recovery_factor=float(recovery),
        profit_factor=float(profit_factor) if profit_factor != float("inf") else 999.0,
        expectancy=float(expectancy),
        avg_hold_bars=float(avg_hold_bars),
        short_hold_ratio=float(short_hold_ratio),
        profit_per_bar=float(profit_per_bar),
        max_win_streak=int(max_win_streak),
        max_loss_streak=int(max_loss_streak),
        profitable_months_ratio=float(profitable_months_ratio),
        monthly_returns=monthly_returns,
    )


# =============================================================================
# MONTE CARLO VALIDATION (v9.0 NEW)
# =============================================================================


def monte_carlo_validation(
    trades: List[Dict[str, Any]],
    n_simulations: int = 1000,
    start_equity: float = STARTING_EQUITY,
    seed: int = 42,
    backtest_days: int = 365,
) -> MonteCarloResult:
    """
    v9.0 NEW: Monte Carlo validace pro odhad robustnosti strategie.

    Shuffluje pořadí obchodů a přepočítává metriky pro odhad
    confidence intervals. Strategie je robustní, pokud výsledky
    zůstávají podobné i při různém pořadí obchodů.

    Args:
        trades: Seznam obchodů z backtestu
        n_simulations: Počet Monte Carlo simulací (default: 1000)
        start_equity: Počáteční kapitál
        seed: Random seed pro reprodukovatelnost

    Returns:
        MonteCarloResult s confidence intervals a robustností
    """
    if not trades or len(trades) < 10:
        # Příliš málo obchodů pro smysluplnou MC simulaci
        return MonteCarloResult(
            sharpe_mean=0.0,
            sharpe_std=0.0,
            sharpe_ci_low=0.0,
            sharpe_ci_high=0.0,
            max_dd_mean=0.0,
            max_dd_std=0.0,
            max_dd_ci_low=0.0,
            max_dd_ci_high=0.0,
            win_rate_mean=0.0,
            win_rate_ci_low=0.0,
            win_rate_ci_high=0.0,
            confidence_level="LOW",
            n_simulations=0,
            robustness_score=0.0,
        )

    np.random.seed(seed)

    # Extract PnL values from trades
    pnl_values = np.array([t.get("pnl_abs", 0.0) for t in trades])
    pnl_pct_values = np.array([t.get("pnl_pct", 0.0) for t in trades])
    n_trades = len(pnl_values)

    # Storage for simulation results
    sharpes = []
    max_dds = []
    win_rates = []

    for _ in range(n_simulations):
        # Shuffle trade order
        indices = np.random.permutation(n_trades)
        shuffled_pnl = pnl_values[indices]

        # Build equity curve
        equity_curve = np.zeros(n_trades + 1)
        equity_curve[0] = start_equity
        equity_curve[1:] = start_equity + np.cumsum(shuffled_pnl)

        # Calculate max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak * 100
        max_dd = float(drawdown.min())
        max_dds.append(max_dd)

        # Calculate Sharpe from shuffled returns (annualize by trades/year)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        trades_per_year = n_trades * 365 / max(backtest_days, 1)
        if len(returns) > 1 and np.std(returns) > 1e-12:
            sharpe = float((np.mean(returns) / np.std(returns)) * np.sqrt(trades_per_year))
        else:
            sharpe = 0.0
        sharpes.append(sharpe)

        # Win rate (same for all shuffles, but calculate anyway)
        win_rate = float(np.sum(shuffled_pnl > 0) / n_trades * 100)
        win_rates.append(win_rate)

    # Convert to arrays for percentile calculation
    sharpes = np.array(sharpes)
    max_dds = np.array(max_dds)
    win_rates = np.array(win_rates)

    # Calculate statistics
    sharpe_mean = float(np.mean(sharpes))
    sharpe_std = float(np.std(sharpes))
    sharpe_ci_low = float(np.percentile(sharpes, 2.5))
    sharpe_ci_high = float(np.percentile(sharpes, 97.5))

    max_dd_mean = float(np.mean(max_dds))
    max_dd_std = float(np.std(max_dds))
    max_dd_ci_low = float(np.percentile(max_dds, 2.5))  # Best case (least negative)
    max_dd_ci_high = float(np.percentile(max_dds, 97.5))  # Worst case (most negative)

    win_rate_mean = float(np.mean(win_rates))
    win_rate_ci_low = float(np.percentile(win_rates, 2.5))
    win_rate_ci_high = float(np.percentile(win_rates, 97.5))

    # Calculate robustness score (0-1)
    # Based on: narrow CI, consistent positive Sharpe, acceptable DD variance
    robustness_factors = []

    # Factor 1: Sharpe CI width (narrower = better)
    sharpe_ci_width = sharpe_ci_high - sharpe_ci_low
    if sharpe_ci_width < 0.3:
        robustness_factors.append(1.0)
    elif sharpe_ci_width < 0.6:
        robustness_factors.append(0.7)
    elif sharpe_ci_width < 1.0:
        robustness_factors.append(0.4)
    else:
        robustness_factors.append(0.2)

    # Factor 2: Sharpe CI lower bound (positive = good)
    if sharpe_ci_low > 0.5:
        robustness_factors.append(1.0)
    elif sharpe_ci_low > 0.0:
        robustness_factors.append(0.7)
    elif sharpe_ci_low > -0.5:
        robustness_factors.append(0.4)
    else:
        robustness_factors.append(0.1)

    # Factor 3: Max DD consistency (less variance = better)
    dd_ci_width = abs(max_dd_ci_high - max_dd_ci_low)
    if dd_ci_width < 5:
        robustness_factors.append(1.0)
    elif dd_ci_width < 10:
        robustness_factors.append(0.7)
    elif dd_ci_width < 15:
        robustness_factors.append(0.4)
    else:
        robustness_factors.append(0.2)

    # Factor 4: Number of trades (more trades = more reliable MC)
    if n_trades >= 100:
        robustness_factors.append(1.0)
    elif n_trades >= 50:
        robustness_factors.append(0.8)
    elif n_trades >= 30:
        robustness_factors.append(0.6)
    else:
        robustness_factors.append(0.3)

    robustness_score = float(np.mean(robustness_factors))

    # Determine confidence level
    if robustness_score >= 0.7 and sharpe_ci_low > 0:
        confidence_level = "HIGH"
    elif robustness_score >= 0.5 and sharpe_ci_low > -0.3:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    return MonteCarloResult(
        sharpe_mean=round(sharpe_mean, 4),
        sharpe_std=round(sharpe_std, 4),
        sharpe_ci_low=round(sharpe_ci_low, 4),
        sharpe_ci_high=round(sharpe_ci_high, 4),
        max_dd_mean=round(max_dd_mean, 2),
        max_dd_std=round(max_dd_std, 2),
        max_dd_ci_low=round(max_dd_ci_low, 2),
        max_dd_ci_high=round(max_dd_ci_high, 2),
        win_rate_mean=round(win_rate_mean, 2),
        win_rate_ci_low=round(win_rate_ci_low, 2),
        win_rate_ci_high=round(win_rate_ci_high, 2),
        confidence_level=confidence_level,
        n_simulations=n_simulations,
        robustness_score=round(robustness_score, 4),
    )
