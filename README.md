# QRE — Quantitative Research Engine

Offline optimizer pro MACD+RSI strategii "Quant Whale Strategy". Hledá optimální parametry pro BTC/USDC a SOL/USDC pomocí Optuna (Anchored Walk-Forward), backtestuje s Numba a výsledky posílá na Discord.

Cíl: najít robustní parametry pro live trading přes [EE (Execution Engine)](https://github.com/b33k33p3r0000/ee) na BrightFunded účtu.

---

## Strategie — Quant Whale Strategy v4.2.0

> Quant Whale is a systematic long/short crypto strategy trading BTC and SOL on 1-hour bars. Entries require three-layer confirmation: a MACD crossover as the trigger, RSI within a lookback window confirming momentum exhaustion, and a higher-timeframe trend filter for directional alignment. The system operates in always-in mode — every exit is simultaneously an entry in the opposite direction — with a per-symbol catastrophic stop as an emergency circuit breaker. All 11 strategy parameters are optimized per-symbol using Optuna with a Log Calmar objective designed to resist overfitting.

Založena na studii Chio (2022) — MACD+RSI dosáhlo 78–86% win rate na US equities.

**Entry logika (3 vrstvy):**
- **Layer 1 — MACD crossover (trigger):** MACD protne signal line na 1H baru
- **Layer 2 — RSI lookback:** RSI bylo v extrémní zóně během posledních `rsi_lookback` barů (4–8h)
- **Layer 3 — Trend filtr:** MACD trend na vyšším TF (4h/8h/1d) souhlasí se směrem
- **LONG:** MACD bull cross AND RSI oversold (lookback) AND higher-TF bullish
- **SHORT:** MACD bear cross AND RSI overbought (lookback) AND higher-TF bearish

**Exit logika:**
- Opačný signál (symetrický flip) — long se zavře a otevře short na sell signálu a naopak
- Catastrophic stop: 5–15% emergency exit (optimalizováno Optunou per-symbol)

**11 Optuna parametrů:**

| Parametr | Rozsah | Popis |
|----------|--------|-------|
| `macd_fast` | 1.0–20.0 (float) | Rychlá EMA perioda |
| `macd_slow` | 10–45 | Pomalá EMA perioda |
| `macd_signal` | 2–15 | Signal line perioda |
| `rsi_period` | 3–30 | RSI výpočetní perioda |
| `rsi_lower` | 20–40 | Práh pro oversold zónu |
| `rsi_upper` | 60–80 | Práh pro overbought zónu |
| `rsi_lookback` | 4–8 | RSI lookback window (bary) |
| `trend_tf` | 4h/8h/1d | Vyšší TF pro trend filtr |
| `trend_strict` | 1 (fixní) | Trend filtr vždy zapnutý |
| `allow_flip` | 1 (fixní) | Position flip vždy zapnutý |
| `catastrophic_stop_pct` | 0.05–0.15 (step 0.01) | Emergency exit level |

Constraints: `macd_slow - macd_fast >= 5` (minimální MACD spread, jinak trial pruned).

**Vlastnosti:**
- Base TF: 1H + trend filtr z vyššího TF (4H/8H/1D)
- Long + Short s position flipping
- Min hold: 2 bary před exit signálem
- Position size: 25% kapitálu na trade

---

## Architektura

```
run.sh (presets)
  └→ python -m qre.optimize --symbol BTC/USDC --trials 20000 ...
       ├→ data/fetch.py       — stáhne OHLCV z Binance (1H + 4H/8H/1D)
       ├→ optimize.py         — Optuna AWF studie (TPE + SHA pruner)
       │    ├→ strategy.py    — generuje buy/sell signály (11 params)
       │    ├→ backtest.py    — Numba trading loop
       │    └→ metrics.py     — Sharpe, drawdown, win rate, ...
       ├→ report.py           — HTML report (Plotly grafy)
       ├→ notify.py           — Discord notifikace (start/complete)
       └→ analyze.py          — post-run health check + suggestions + Discord embed
```

---

## Moduly

| Modul | Popis |
|-------|-------|
| `optimize.py` | AWF orchestrátor — Optuna TPE + SHA pruner, RSI cache, hard constraints inline |
| `core/strategy.py` | Quant Whale Strategy v4.2.0 — MACD crossover + RSI lookback + trend filter |
| `core/backtest.py` | Numba JIT trading loop — Long+Short, position flipping, per-trial catastrophic stop |
| `core/indicators.py` | RSI (SMA-based) a MACD výpočty |
| `core/metrics.py` | Sharpe, Sortino, Calmar, drawdown, win rate, Monte Carlo |
| `monitor.py` | Live TUI dashboard — sledování běžících optimalizací v reálném čase (Rich) |
| `analyze.py` | Post-run diagnostika — health check, suggestions, Discord embed |
| `data/fetch.py` | Binance OHLCV fetch (1H + 4H/8H/1D, fresh data bez cache) |
| `report.py` | Self-contained HTML report s Plotly grafy |
| `notify.py` | Discord webhooky — start/complete notifikace |
| `io.py` | JSON/CSV zápis výsledků |
| `config.py` | Centrální konfigurace |

---

## Optimalizace (AWF)

Anchored Walk-Forward = trénink na rostoucím okně, test na dalším bloku.

```
Split 1:  [====== train 60% ======][= test =]
Split 2:  [========= train 70% =========][= test =]
Split 3:  [============ train 80% ============][= test =]
Split 4:  [=============== train 85% ==============][= test =]
Split 5:  [================== train 90% ================][= test =]
```

- Default: 5 splitů, test window 20%
- Krátká data (<1.5 roku): 2 splity (70%/85% train)
- Optuna TPE sampler s SuccessiveHalving prunerem
- RSI pre-computed cache: 28 period (3–30) místo počítání per-trial
- **Purge gap:** 50 barů mezi train/test splity (eliminace indicator leakage)
- Monte Carlo validace (1000 simulací) na OOS splitech

**Objective — Log Calmar + anti-gaming guards:**
```
score = log(1 + calmar) × trade_ramp × sharpe_penalty
```
- **Log Calmar:** `log(1 + annual_return / max(DD, 5%))` — komprese extrémních hodnot
- **Trade ramp:** `min(1.0, trades_per_year / 100)` — penalizuje <100 trades/rok
- **Sharpe decay:** `1/(1 + 0.3*(sharpe - 3.0))` když Sharpe > 3.0
- **Hard constraints (inline v optimize.py):** < 30 trades/rok NEBO < 5 test trades → score = 0

---

## Backtest Engine

Numba `@njit` compiled trading loop:

- **Pozice:** flat → long → short → flat (s flippingem)
- **Priority:** catastrophic stop > signal exit > new position
- **Catastrophic stop:** per-trial level (5–15%, Optuna optimized) → emergency exit
- **Force close:** otevřená pozice na konci dat se zavře
- **Směry:** Long (+1) a Short (-1) s korektním PnL modelem
- **Short PnL:** `pnl = size * entry * (1-fee) - size * exit * (1+fee)`

---

## Spuštění

```bash
cd ~/projects/qre
./run.sh 1               # Test: 5k trials, BTC+SOL, ~15 min
./run.sh 2               # BTC Main: 20k trials, 5 splits, ~60 min
./run.sh 3               # SOL Main: 40k trials, 5 splits, ~180 min
./run.sh 4               # Custom: interaktivní volba
```

| Preset | Trials | Splity | Symboly | Doba |
|--------|--------|--------|---------|------|
| 1 Test | 5,000 | 3 | BTC+SOL | ~15 min |
| 2 BTC Main | 20,000 | 5 | BTC | ~60 min |
| 3 SOL Main | 40,000 | 5 | SOL | ~180 min |
| 4 Custom | volba | volba | volba | — |

Výchozí: `--hours 18600` (~2 roky), `--skip-recent 1080` (skip posledních 45 dní).

**Process management:**
```bash
./run.sh attach      # Připojit se k běžícímu runu
./run.sh logs        # Výpis log souborů
./run.sh kill        # Zastavit optimalizaci
```

**Přímé spuštění (bez run.sh):**
```bash
python -m qre.optimize --symbol BTC/USDC --trials 5000 --hours 8760 --splits 3
```

CLI parametry: `--symbol`, `--hours`, `--trials`, `--splits`, `--seed`, `--timeout`, `--tag`, `--skip-recent`, `--results-dir`, `--test-size`.

---

## Live Monitor

Sledování běžících optimalizací v reálném čase. Čte Optuna SQLite checkpoint DB v read-only modu.

```bash
python -m qre.monitor                           # auto-detect aktivní runy
python -m qre.monitor calmar-btc                # filtr na konkrétní run
python -m qre.monitor --interval 5              # refresh každých 5s
```

**Zobrazuje per symbol:**
- Progress (completed / requested trials, trials/min, ETA)
- Best trial (Log Calmar value, trial číslo)
- Rozšířené metriky (Sharpe equity, max DD, P&L%, trades, trades/yr)
- Optimální parametry (MACD/RSI/trend v kompaktním formátu)

**Vlastnosti:**
- Auto-detect aktivních runů (DB modified < 5 minut)
- Multi-run podpora (BTC + SOL současně v separátních panelech)
- Detekce nového best trialu (`NEW` marker)
- Graceful degradation pro starší runy bez rozšířených metrik

---

## Výstupy

Každý run vytvoří složku `results/<timestamp>_<tag>/<SYMBOL>/`:

```
results/2026-02-22_08-30-00_btc-main/
  └── BTC/
      ├── best_params.json     # Optimální parametry + metriky
      ├── trades_BTC_USDC_1h_FULL.csv   # Všechny obchody
      ├── report_BTC.html      # Interaktivní HTML report
      └── analysis.json        # Post-run diagnostika
```

---

## Konfigurace

Klíčové konstanty v `config.py`:

| Konstanta | Hodnota | Popis |
|-----------|---------|-------|
| `SYMBOLS` | BTC/USDC, SOL/USDC | Obchodované páry |
| `BASE_TF` | 1h | Base timeframe (+ trend filtr z 4h/8h/1d) |
| `STARTING_EQUITY` | $50,000 | Per-pair alokace ($100k / 2) |
| `BACKTEST_POSITION_PCT` | 0.25 | 25% kapitálu na trade |
| `CATASTROPHIC_STOP_PCT` | 0.10 | Fallback default (Optuna override 5–15%) |
| `LONG_ONLY` | False | Long + Short povoleno |
| `MIN_HOLD_HOURS` | 2 | Min bary před exit signálem |
| `FEE` | 0.075% | Trading fee |
| `MIN_TRADES_YEAR_HARD` | 30 | Hard constraint |
| `MIN_TRADES_TEST_HARD` | 5 | Hard constraint per test split |
| `MIN_DRAWDOWN_FLOOR` | 0.05 | 5% DD floor pro Calmar (anti-gaming) |
| `TARGET_TRADES_YEAR` | 100 | Trade ramp target (plný score od 100/rok) |
| `SHARPE_SUSPECT_THRESHOLD` | 3.0 | Sharpe decay práh |
| `PURGE_GAP_BARS` | 50 | Purge gap mezi train/test splity |
| `N_SPLITS_DEFAULT` | 3 | Default pro krátká data (run.sh presety používají 5) |

Trading costs (slippage): BTC 0.08%, SOL 0.18%.

---

## Discord Notifikace

Kanál `#qre-runs` dostává:
1. **Start** — symbol, trials, history, splity
2. **Complete** — equity, PnL%, trades, win rate, Sharpe, MC confidence
3. **Run Analysis** — health check, verdict (PASS/REVIEW/FAIL), suggestions

Webhook URL v `.env` jako `DISCORD_WEBHOOK_RUNS`.

---

## Post-Run Analýza

Auto-diagnose po každém runu spustí `analyze_run()`:

**Health Check** (8 metrik):
- Sharpe, Max Drawdown, Trades/Year, Win Rate, Profit Factor, Expectancy
- Train/Test Sharpe divergence, Split consistency

**Verdikt:** PASS (vše zelené) / REVIEW (1 red nebo 3+ yellow) / FAIL (2+ red)

**Threshold Analysis:** MACD spread health, RSI zone width health

**Suggestions:** Konkrétní doporučení (widen RSI zones, tighten stops, broaden ranges, ...)

---

## Adresářová struktura

```
qre/
├── src/qre/
│   ├── config.py
│   ├── optimize.py
│   ├── analyze.py
│   ├── notify.py
│   ├── report.py
│   ├── monitor.py
│   ├── io.py
│   ├── core/
│   │   ├── strategy.py
│   │   ├── backtest.py
│   │   ├── indicators.py
│   │   └── metrics.py
│   └── data/
│       └── fetch.py
├── tests/
│   ├── unit/          # 209 testů
│   ├── integration/
│   └── conftest.py
├── scripts/           # Analytické skripty (compare_stops.py, ...)
├── results/           # Výstupy runů
├── logs/              # Log soubory (background runs)
├── run.sh             # Entry point s presety
├── NOTES.md           # Session notes (celá historie)
└── pyproject.toml
```

---

## Tech Stack

- **Python 3.11+** (testováno na 3.14)
- **Optuna 4.7** — TPE sampler, SuccessiveHalving pruner
- **Numba 0.63** — JIT kompilace trading loopu
- **NumPy, pandas** — data manipulace
- **ccxt** — Binance API
- **Plotly** — HTML reporty (equity curve, drawdown, trade distribuce)
- **requests** — Discord webhooky
- **Rich** — live monitor TUI
- **pytest** — 209 unit a integračních testů
