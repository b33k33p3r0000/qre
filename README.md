# QRE — Quantitative Research Engine

Offline optimizer pro MACD+RSI strategii "Quant Whale Strategy". Hledá optimální parametry pro BTC/USDC a SOL/USDC pomocí Optuna (Anchored Walk-Forward), backtestuje s Numba a výsledky posílá na Discord.

Cíl: najít robustní parametry pro live trading přes [EE (Execution Engine)](https://github.com/b33k33p3r0000/ee) na BrightFunded účtu.

---

## Strategie — Quant Whale Strategy v3.0

Založena na studii Chio (2022) — MACD+RSI dosáhlo 78–86% win rate na US equities.

**Entry logika:**
- **LONG:** MACD bullish crossover (MACD protne signal line zdola) AND RSI < `rsi_lower` (oversold)
- **SHORT:** MACD bearish crossover (MACD protne signal line shora) AND RSI > `rsi_upper` (overbought)

**Exit logika:**
- Opačný signál (symetrický flip) — long se zavře a otevře short na sell signálu a naopak
- Catastrophic stop: -15% emergency exit

**6 Optuna parametrů:**

| Parametr | Rozsah | Popis |
|----------|--------|-------|
| `macd_fast` | 3–20 | Rychlá EMA perioda |
| `macd_slow` | 15–45 | Pomalá EMA perioda |
| `macd_signal` | 2–15 | Signal line perioda |
| `rsi_period` | 3–30 | RSI výpočetní perioda |
| `rsi_lower` | 15–50 | Práh pro oversold zónu |
| `rsi_upper` | 50–85 | Práh pro overbought zónu |

Constraint: `macd_slow - macd_fast >= 5` (minimální MACD spread, jinak trial pruned).

**Vlastnosti:**
- Jeden timeframe: 1H
- Long + Short s position flipping (konfigurovatelné přes `LONG_ONLY`)
- Min hold: 2 bary před exit signálem
- Position size: 25% kapitálu na trade

---

## Architektura

```
run.sh (presets)
  └→ python -m qre.optimize --symbol BTC/USDC --trials 10000 ...
       ├→ data/fetch.py       — stáhne 1H OHLCV z Binance
       ├→ optimize.py         — Optuna AWF studie (TPE + SHA pruner)
       │    ├→ strategy.py    — generuje buy/sell signály
       │    ├→ backtest.py    — Numba trading loop
       │    ├→ metrics.py     — Sharpe, drawdown, win rate, ...
       │    └→ penalties.py   — hard constraint + overtrading
       ├→ report.py           — HTML report (Plotly grafy)
       ├→ notify.py           — Discord notifikace (start/complete)
       └→ hooks/              — post-run auto-diagnose
            └→ analyze.py     — health check + suggestions + Discord embed
```

---

## Moduly

| Modul | Popis |
|-------|-------|
| `optimize.py` | AWF orchestrátor — Optuna TPE + SHA pruner, RSI cache |
| `core/strategy.py` | Quant Whale Strategy v3.0 — MACD crossover + RSI extreme zones |
| `core/backtest.py` | Numba JIT trading loop — Long+Short, position flipping |
| `core/indicators.py` | RSI a MACD výpočty |
| `core/metrics.py` | Sharpe, Sortino, Calmar, drawdown, win rate, Monte Carlo |
| `penalties.py` | Hard constraint (min trades/year) + overtrading penalty |
| `analyze.py` | Post-run diagnostika — health check, suggestions, Discord embed |
| `data/fetch.py` | Binance OHLCV fetch (1H, fresh data bez cache) |
| `report.py` | Self-contained HTML report s Plotly grafy |
| `notify.py` | Discord webhooky — start/complete notifikace |
| `io.py` | JSON/CSV zápis výsledků |
| `hooks/` | Hook systém — auto_diagnose po každém runu |
| `config.py` | Centrální konfigurace |

---

## Optimalizace (AWF)

Anchored Walk-Forward = trénink na rostoucím okně, test na dalším bloku.

```
Split 1:  [====== train 60% ======][= test =]
Split 2:  [========= train 70% =========][= test =]
Split 3:  [============ train 80% ============][= test =]
```

- 2 splity pro krátká data (<1.5 roku), 3 splity jinak
- Optuna TPE sampler s SuccessiveHalving prunerem
- RSI pre-computed cache: 23 period (3–25) místo počítání per-trial
- Objective = penalizované equity z posledního test splitu
- Monte Carlo validace (1000 simulací) na finálním výsledku

**Penalties:**
1. **Hard constraint:** < 30 trades/rok NEBO < 3 test trades → equity = 0
2. **Overtrading:** > 500 trades/rok → až -15% penalizace

---

## Backtest Engine

Numba `@njit` compiled trading loop:

- **Pozice:** flat → long → short → flat (s flippingem)
- **Priority:** catastrophic stop > signal exit > new position
- **Catastrophic stop:** -15% od entry → emergency exit
- **Force close:** otevřená pozice na konci dat se zavře
- **Směry:** Long (+1) a Short (-1) s korektním PnL modelem
- **Short PnL:** `pnl = size * entry * (1-fee) - size * exit * (1+fee)`

---

## Spuštění

```bash
cd ~/projects/qre
./run.sh 1 --btc           # Test preset: 2k trials, BTC only
./run.sh 2 --btc --fg      # Prod: 10k trials, foreground
./run.sh 3 --both           # Deep: 15k trials, oba páry
./run.sh 4 --sol            # Über: 25k trials, SOL
```

| Preset | Trials | Splity | Doba |
|--------|--------|--------|------|
| 1 Test | 2,000 | 2 | ~5 min |
| 2 Prod | 10,000 | 3 | ~60 min |
| 3 Deep | 15,000 | 3 | ~120 min |
| 4 Über | 25,000 | 3 | ~300 min |

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

## Výstupy

Každý run vytvoří složku `results/<timestamp>_<tag>/<SYMBOL>/`:

```
results/2026-02-16_08-36-12_test-v/
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
| `BASE_TF` | 1h | Jediný timeframe |
| `STARTING_EQUITY` | $50,000 | Per-pair alokace ($100k / 2) |
| `BACKTEST_POSITION_PCT` | 0.25 | 25% kapitálu na trade |
| `CATASTROPHIC_STOP_PCT` | 0.10 | -10% emergency exit (Quant Whale Strategy spec) |
| `LONG_ONLY` | False | Long + Short povoleno |
| `MIN_HOLD_HOURS` | 2 | Min bary před exit signálem |
| `FEE` | 0.075% | Trading fee |
| `MIN_TRADES_YEAR_HARD` | 30 | Hard constraint |
| `MIN_TRADES_TEST_HARD` | 5 | Hard constraint per test split |
| `MIN_TRADES_TEST_SOFT` | 15 | Soft penalty threshold (-15%) |
| AWF `n_splits` | 5 | Default počet AWF splitů |
| AWF `test_size` | 0.20 | Test window = 20% dat |

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

Auto-diagnose hook po každém runu spustí `analyze_run()`:

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
│   ├── penalties.py
│   ├── analyze.py
│   ├── notify.py
│   ├── report.py
│   ├── io.py
│   ├── core/
│   │   ├── strategy.py
│   │   ├── backtest.py
│   │   ├── indicators.py
│   │   └── metrics.py
│   ├── data/
│   │   └── fetch.py
│   └── hooks/
│       ├── __init__.py
│       └── auto_diagnose.py
├── tests/
│   ├── unit/          # 168 testů
│   ├── integration/   # Pipeline, golden baseline, reproducibility
│   └── fixtures/
├── results/           # Výstupy runů
├── logs/              # Log soubory (nohup background runs)
├── run.sh             # Entry point s presety
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
- **pytest** — 168+ unit a integračních testů

---

## BTC Market Analysis — Training Data Strategy

### Historical Drawdowns (>15%), 2016-2026

```
                    BTC/USD — 10Y Drawdowns (log scale)

 $126k ┤                                                              ★ ATH
       │                                                            ╭╱╲
  $69k ┤                                    ★ ATH               ╭──╯   ╲ ⑥
       │                                   ╱╲                  ╱  -47.6% ╲
  $32k ┤                             ④   ╱  ╲  ⑤            ╱             ╰─ $70k
       │                           -50% ╱    ╲ -76%    ╭───╯
  $20k ┤              ★ ATH            ╱      ╲       ╱
       │             ╱╲               ╱        ╲     ╱
   $7k ┤            ╱  ╲  ②         ╱          ╰───╯
       │           ╱    ╲ -81.6%  ╱
   $3k ┤     ╭─╮ ╱      ╰──────╯
       │    ╱ ① ╲╱
  $430 ┤───╯ -18.7%
       ├──────┼──────┼──────┼──────┼──────┼──────┤
      2016  2017   2018   2019   2021   2023   2026
```

| # | Období | Peak | Trough | Propad | Trvání |
|---|--------|------|--------|--------|--------|
| 1 | Propad 2016 | $750 (Jun 2016) | $610 (Sep 2016) | **-18.7%** | 92 dní |
| 2 | Bear 2017-18 | $20,089 (Dec 2017) | $3,700 (Dec 2018) | **-81.6%** | 381 dní |
| 3 | COVID 2020 | ~$10,000 (Feb 2020) | $4,800 (Mar 2020) | **-52%** | ~35 dní |
| 4 | 2021 Q2 Propad | $64,000 (Apr 2021) | $32,000 (Jul 2021) | **-50.0%** | 78 dní |
| 5 | Bear 2021-22 | $68,789 (Nov 2021) | $16,500 (Nov 2022) | **-76.0%** | 370 dní |
| 6 | Propad 2025-26 | $126,000 (Oct 2025) | $66,000 (Feb 2026) | **-47.6%** | 123 dní |

Roční performance:

| 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|------|------|------|------|------|------|------|------|------|------|
| +120.9% | +1315.6% | -73.6% | +87.0% | +302.8% | +59.3% | -65.3% | +151.5% | +128.2% | -8.8% |

**Pattern:** BTC cykluje ~4 roky. Major bear markets (-65% až -82%) přicházejí po ATH cyklech. Mezi nimi korekce -18% až -50%.

*Zdroj: Binance/Yahoo Finance/CoinDesk, generováno 2026-02-15*
