# Quantitative Research Engine (QRE)

Systematic algo trading research engine — Anchored Walk-Forward · Bayesian optimisation · Monte Carlo validation

## Live Performance — 3-pair (QWS v4.3.0, 2026-03-16)

| Metric | BTC/USDT | SOL/USDT | BNB/USDT |
|--------|----------|----------|----------|
| Equity | $206,923 (+107%) | $264,595 (+165%) | $182,858 (+83%) |
| Sharpe (equity) | 2.89 | 2.35 | 2.17 |
| Max Drawdown | -4.51% | -6.63% | -11.28% |
| Trades / year | 153 | 115 | 125 |
| MC Confidence | 0.95 HIGH | 0.80 HIGH | 0.95 HIGH |

> 3 AWF splits, 50k Optuna trials per symbol. $100k starting equity per pair.

---

## Pipeline Overview

```
Data (Local Parquet / Binance OHLCV)
  │
  ├── 1H bary (primary)
  ├── 4H / 8H / 1D bary (trend filter)
  │
  ▼
Signály (strategy.py)
  │  Layer 1: MACD crossover
  │  Layer 2: RSI lookback window
  │  Layer 3: Multi-TF trend filter
  │  → buy_signal[], sell_signal[]
  │
  ▼
Backtest (backtest.py, Numba JIT)
  │  Simulace obchodů na 1H barech
  │  Position sizing 15%, fees, slippage, catastrophic stop, trailing stop
  │  → trades[], equity_curve[]
  │
  ▼
AWF Splity (optimize.py)
  │  3 splity s rostoucím train window
  │  Purge gap 50 barů mezi train/test
  │  Hard constraints (30 trades/yr train, 5 trades test)
  │
  ▼
Objective (optimize.py)
  │  Per-split: log(1 + Calmar) × trade_ramp × sharpe_decay
  │  Final score = průměr přes splity
  │
  ▼
Optuna (optimize.py)
  │  TPE sampler, 12 params, N trials
  │  → best_params (highest avg score)
  │
  ▼
Výsledky
  │  Full backtest s best params
  │  Per-split OOS Monte Carlo (1000 shuffles)
  │  → best_params.json, trades CSV, HTML report
```

---

## Entry Logic (3-Layer Confirmation)

> **Strategy version: v4.3.0** — Added trailing stop (4th exit layer) + ETH/USDT + XRP/USDT + 15% position sizing + Marathon preset.

Vstup vyžaduje **simultánní potvrzení všech 3 vrstev** na stejném 1H baru:

```
LONG  = MACD bullish cross AND RSI oversold (s lookback) AND HTF bullish
SHORT = MACD bearish cross AND RSI overbought (s lookback) AND HTF bearish
```

### Layer 1: MACD Crossover (Trigger)

```
ema_fast    = EMA(close, span=macd_fast)       # macd_fast: float 1.0-20.0
ema_slow    = EMA(close, span=macd_slow)       # macd_slow: int 10-45
macd_line   = ema_fast - ema_slow
signal_line = EMA(macd_line, span=macd_signal) # macd_signal: int 3-15

bullish_cross = (macd_prev <= signal_prev) AND (macd_curr > signal_curr)
bearish_cross = (macd_prev >= signal_prev) AND (macd_curr < signal_curr)
```

Hard constraint: `macd_slow - macd_fast >= 5` (porušení → TrialPruned).

### Layer 2: RSI Lookback Window (Filter)

RSI je SMA-based (ne Wilder's EMA).

```
delta    = close.diff()
avg_gain = SMA(max(delta, 0), rsi_period)      # rsi_period: int 3-30
avg_loss = SMA(abs(min(delta, 0)), rsi_period)
RSI      = 100 - (100 / (1 + avg_gain / avg_loss))

rsi_oversold   = rolling_max(RSI < rsi_lower, window=rsi_lookback + 1)
rsi_overbought = rolling_max(RSI > rsi_upper, window=rsi_lookback + 1)
```

`rsi_lookback` (int 1-4) = "paměť" — `lookback=1` vyžaduje RSI v zóně nyní/předchozí bar, `lookback=4` stačí v posledních 4h.

### Layer 3: Multi-TF Trend Filter (Guard)

Filtruje vstupy proti dominantnímu trendu. Používá **stejné MACD params** jako Layer 1.

```
# Výpočet na vyšším timeframe (trend_tf ∈ {4h, 8h, 1d})
htf_macd, htf_signal = MACD(htf_close, macd_fast, macd_slow, macd_signal)
htf_bullish = (htf_macd > htf_signal)
```

HTF signál se alignuje na 1H bary přes timestamp binary search. `trend_strict=1` (vždy zapnuto):
- LONG: vyžaduje `htf_bullish`
- SHORT: vyžaduje `htf_bearish`

---

## Exit Logic & Position Management

### Priorita exitů (per bar)

```
1. Catastrophic Stop    (check PRVNÍ, před signálem)
2. Trailing Stop        (ATR(14) based, aktivuje se po trail_activation_mult × ATR)
3. Signal Exit          (pokud bars_held >= 2)
4. Force Close          (konec dat)
```

### Catastrophic Stop (emergency)

Per-symbol fixní limity:

```
LONG:  if (low / entry_price - 1.0) <= -catastrophic_stop_pct
       → exit_price = entry_price × (1 - stop_pct) × (1 - slippage)

SHORT: if (high / entry_price - 1.0) >= catastrophic_stop_pct
       → exit_price = entry_price × (1 + stop_pct) × (1 + slippage)
```

| Symbol | Stop | ~Equity loss per trade |
|--------|------|----------------------|
| BTC | 8% | ~1.2% |
| SOL | 12% | ~1.8% |
| BNB | 10% | ~1.5% |
| ETH | 8% | ~1.2% |
| XRP | 10% | ~1.5% |

### Trailing Stop (harvest)

ATR(14) based trailing stop — aktivuje se po dosažení profit threshold, chrání realizovaný zisk.

```
trail_activation_mult (float 1.0-3.0):  aktivace po entry_price ± activation_mult × ATR(14)
trail_mult            (float 1.5-4.0):  trail vzdálenost = trail_mult × ATR(14)

LONG:  pokud high >= entry + activation_mult × ATR
       → trail_price = high - trail_mult × ATR  (pohybuje se jen nahoru)
       → exit pokud close < trail_price

SHORT: pokud low <= entry - activation_mult × ATR
       → trail_price = low + trail_mult × ATR  (pohybuje se jen dolů)
       → exit pokud close > trail_price
```

### Signal Exit (primární)

Vyžaduje **stejné 3 vrstvy jako entry, ale v opačném směru.** Min hold: 2 bary.

### Position Management: allow_flip

```
allow_flip=0 (Selective, default): close → FLAT → čekat na nový entry
allow_flip=1 (Always-In):         close → okamžitě otevřít opačný směr
```

---

## Backtest Engine

### Position Sizing

```
capital_at_entry = equity × 0.15     # 15% equity per trade (5 pairs × 15% = 75% max exposure)
position_size    = capital_at_entry / (entry_price × (1 + fee))
```

Start equity: $100,000 (celkový účet, bez per-pair dělení).

### Trading Costs

Fee: 6 bps per side. Slippage (asymetrický):

| Symbol | Slippage |
|--------|----------|
| BTC/USDT | 6 bps |
| SOL/USDT | 12 bps |
| BNB/USDT | 8 bps |
| ETH/USDT | 7 bps |
| XRP/USDT | 10 bps |
| Default fallback | 15 bps |

### Per-Bar Flow (Numba JIT)

```
Bar N:
  1. MÁ POZICI?
     a. Check catastrophic stop (high/low vs entry_price)
        → if triggered: EXIT, reason="catastrophic_stop"
     b. Check trailing stop (if activated: close vs trail_price)
        → if triggered: EXIT, reason="trailing_stop"
     c. Check signal exit (if bars_held >= 2):
        → if opačný 3-layer signál fires:
           - EXIT, reason="signal"
           - if allow_flip=1: OKAMŽITĚ otevři opačný směr

  2. NEMÁ POZICI (flat)?
     a. if buy_signal: OPEN LONG
     b. elif sell_signal: OPEN SHORT

Konec dat:
  → CLOSE jakákoli otevřená pozice, reason="force_close"
```

---

## AWF (Anchored Walk-Forward)

Anchored = train window ROSTE (kotvený ke startu). Test windows se nepřekrývají.

```
Split 1: Train [0% ──── 60%] ··purge·· Test [60.x% ── 70%]
Split 2: Train [0% ──────── 70%] ··purge·· Test [70.x% ── 80%]
Split 3: Train [0% ──────────── 80%] ··purge·· Test [80.x% ── 90%]
```

| Data délka | Splity | Train/Test |
|------------|--------|------------|
| >= 1.5yr (>= 13,140h) | 3 | 60/70/80% train, ~10% test each |
| < 1.5yr (>= 4,000h) | 2 | 70/85% train, 15% test |
| < 4,000h | Error | — |

**Hard Constraints** (trial = 0.0 pokud porušeno):

| Constraint | Hodnota |
|------------|---------|
| `MIN_TRADES_YEAR_HARD` | 30 trades/rok (train set) |
| `MIN_TRADES_TEST_HARD` | 5 trades (každý test split) |

---

## Log Calmar Objective

```
score = log(1 + raw_calmar) × trade_ramp × sharpe_decay

kde:
  raw_calmar     = max(0, annual_return / max(max_dd, 0.05))
  trade_ramp     = min(1.0, trades_per_year / 100)
  sharpe_decay   = 1 / (1 + 0.3 × (sharpe - 3.0))   # only if sharpe > 3.0
```

Anti-gaming mechanismy:

| Mechanismus | Co brání |
|-------------|----------|
| DD floor 5% | Minimalizace DD na ~0% |
| Log komprese | Extrémní Calmar hodnoty |
| Trade ramp | Cherry-picking pár obchodů (penalty pod 100/rok) |
| Sharpe decay | Přeoptimalizované params (nad Sharpe 3.0) |
| AWF průměr | Overfitting na jedno období |
| Hard constraints | Příliš málo obchodů (30/yr train, 5/split test) |

---

## Monte Carlo Validace

Testuje robustnost: "Jsou výsledky závislé na konkrétním pořadí obchodů?"

```
Pro každou z 1000 simulací:
  1. Zamíchej pořadí obchodů
  2. Sestav novou equity curve
  3. Spočítej Sharpe a Max DD

Výsledek:
  - 95% CI pro Sharpe a Max DD
  - Robustness score: 0.0 - 1.0
  - Confidence level: HIGH / MEDIUM / LOW
```

**Robustness Score** (průměr 4 faktorů): Sharpe CI width, Sharpe CI low, DD CI width, trade count.

**Agregace přes splity** — konzervativní (worst-case):
```
sharpe_ci_low = min(across splits)
robustness    = min(across splits)
confidence    = weakest(across splits)
```

---

## Metrics

### Primary

| Metrika | Výpočet | Annualizace |
|---------|---------|-------------|
| **sharpe_equity** | Daily equity returns | sqrt(365) — **PRIMARY** |
| **calmar** | Annual return / \|max_dd\| | Annualizovaný return |
| **max_drawdown** | Peak-to-trough na equity curve | — |

### Secondary

| Metrika | Výpočet |
|---------|---------|
| **sharpe_time** | Hourly price returns, sqrt(8760) — fees/slippage chybí → nadhodnocuje |
| **sortino** | Daily equity returns, downside vol only, sqrt(365) |
| **win_rate** | count(pnl > 0) / total_trades × 100 |
| **profit_factor** | gross_profit / gross_loss |
| **trades_per_year** | total_trades / (days / 365.25) |
| **expectancy** | (WR × avg_win) - ((1-WR) × avg_loss) |

---

## 12 Optuna Parameters

| Param | Typ | Range | Poznámka |
|-------|-----|-------|----------|
| `macd_fast` | float | 1.0 - 20.0 | Must be < macd_slow by >= 5 |
| `macd_slow` | int | 10 - 45 | Must be > macd_fast by >= 5 |
| `macd_signal` | int | 3 - 15 | Min=3 intentional floor |
| `rsi_period` | int | 3 - 30 | Min=3 intentional floor |
| `rsi_lower` | int | 20 - 40 | Oversold threshold |
| `rsi_upper` | int | 60 - 80 | Overbought threshold |
| `rsi_lookback` | int | 1 - 4 | RSI memory window |
| `trend_tf` | cat | 4h, 8h, 1d | HTF trend filter TF |
| `trend_strict` | int | 1 (fixed) | Vždy zapnuto |
| `allow_flip` | int | 0 (fixed) | 0=selective, 1=always-in |
| `trail_activation_mult` | float | 1.0 - 3.0 | ATR(14) multiples before trail activates |
| `trail_mult` | float | 1.5 - 4.0 | ATR(14) multiples for trail distance |

---

## Architektura

```
run.sh (presets)
  └→ python -m qre.optimize --symbol BTC/USDT --trials 30000 ...
       ├→ data/fetch.py       — stáhne OHLCV (lokální Parquet / Binance)
       ├→ optimize.py         — Optuna AWF studie (TPE + SHA pruner)
       │    ├→ strategy.py    — generuje buy/sell signály (12 params)
       │    ├→ backtest.py    — Numba trading loop
       │    └→ metrics.py     — Sharpe, drawdown, win rate, ...
       ├→ report.py           — HTML report (Plotly grafy)
       ├→ notify.py           — Discord notifikace (start/complete)
       └→ analyze.py          — post-run health check + suggestions + Discord embed
```

---

## Spuštění

```bash
cd ~/projects/qre
./run.sh 1               # Test: 5k trials, 1yr, BTC (~15 min)
./run.sh 2               # Quick: 15k trials, 2yr, BTC+SOL (~1-2 hr)
./run.sh 3               # Main: 40k trials, 3yr, all pairs (~4-8 hr)
./run.sh 4               # Deep: 50k trials, 5yr, all pairs (~12-24 hr)
./run.sh 5               # Marathon: 50k trials, max data, all pairs (~24-48 hr)
./run.sh 6               # Custom: interaktivní volba
```

| Preset | Trials | Hours | Splity | Symboly |
|--------|--------|-------|--------|---------|
| 1 Test | 5,000 | 8,760 (1yr) | 3 | BTC |
| 2 Quick | 15,000 | 17,520 (2yr) | 3 | BTC+SOL |
| 3 Main | 40,000 | 26,280 (3yr) | 3 | BTC+SOL+BNB+ETH+XRP |
| 4 Deep | 50,000 | 43,800 (5yr) | 5 | BTC+SOL+BNB+ETH+XRP |
| 5 Marathon | 50,000 | 87,600 (max) | 5 | BTC+SOL+BNB+ETH+XRP |
| 6 Custom | volba | volba | volba | volba |

Paralelní spuštění (2 páry najednou, `--eth`, `--xrp`, `--original` flagy pro výběr skupiny):
```bash
./run.sh 3 --original    # Main: BTC+SOL+BNB (původní páry)
./run.sh 3 --eth         # Main: ETH+XRP (nové páry, 2 páry paralelně)
```

### Process Management

```bash
./run.sh attach      # Připojit se k běžícímu runu
./run.sh logs        # Výpis log souborů
./run.sh kill        # Zastavit optimalizaci
```

### Přímé spuštění

```bash
python -m qre.optimize --symbol BTC/USDT --trials 5000 --hours 8760 --splits 3
```

CLI parametry: `--symbol`, `--hours`, `--trials`, `--splits`, `--seed`, `--timeout`, `--tag`, `--skip-recent`, `--results-dir`, `--test-size`.

---

## Live Monitor

```bash
python -m qre.monitor                           # auto-detect aktivní runy
python -m qre.monitor calmar-btc                # filtr na konkrétní run
python -m qre.monitor --interval 5              # refresh každých 5s
```

Zobrazuje: progress, best trial, metriky (Sharpe, DD, PnL%, trades), optimální params.

---

## Konfigurace

Klíčové konstanty v `config.py`:

| Konstanta | Hodnota | Popis |
|-----------|---------|-------|
| `SYMBOLS` | BTC/USDT, SOL/USDT, BNB/USDT, ETH/USDT, XRP/USDT | Obchodované páry |
| `STARTING_EQUITY` | $100,000 | Celkový účet (bez per-pair dělení) |
| `BACKTEST_POSITION_PCT` | 0.15 | 15% equity na jednu pozici (5 × 15% = 75% max) |
| `FEE` | 6 bps | Trading fee per side |
| `CATASTROPHIC_STOP_PCT` | BTC 8%, SOL 12%, BNB 10%, ETH 8%, XRP 10% | Per-symbol fixní |
| `LONG_ONLY` | False | Long + Short povoleno |
| `MIN_HOLD_HOURS` | 2 | Min bary před exit signálem |
| `MIN_TRADES_YEAR_HARD` | 30 | Hard constraint (train) |
| `MIN_TRADES_TEST_HARD` | 5 | Hard constraint (test split) |
| `MIN_DRAWDOWN_FLOOR` | 5% | DD floor pro anti-gaming |
| `TARGET_TRADES_YEAR` | 100 | Trade ramp target |
| `SHARPE_SUSPECT_THRESHOLD` | 3.0 | Sharpe decay trigger |
| `PURGE_GAP_BARS` | 50 | Purge gap train/test |
| `MONTE_CARLO_SIMULATIONS` | 1,000 | MC shuffles per split |
| `MIN_WARMUP_BARS` | 200 | Bars skipped at start |

### Optuna

| Konstanta | Hodnota |
|-----------|---------|
| Sampler | TPE |
| Startup ratio | 20% random |
| EI candidates | 24 |
| Pruner | SuccessiveHalving |
| Consider endpoints | True |

---

## Výstupy

```
results/<timestamp>_<tag>/<SYMBOL>/
  ├── best_params.json     # Optimální parametry + metriky
  ├── trades_*.csv         # Všechny obchody
  ├── report_*.html        # Interaktivní HTML report (Plotly)
  └── analysis.json        # Post-run diagnostika
```

---

## Post-Run Analýza

Auto-diagnose po každém runu — health check (8 metrik), verdikt (PASS/REVIEW/FAIL), suggestions.

Discord kanál `#qre-runs`: start, complete, run analysis notifikace.

---

## Autonomous Optimizer

Autonomní agent pro iterativní vylepšování QRE strategie. Analyzuje výsledky runů, implementuje změny, spouští nové runy a porovnává — v loop dokud nedosáhne TOP tier nebo max iterací.

### Spuštění

```bash
cd ~/projects/qre
claude --agent autonomous-optimizer
```

Agent se zeptá na:
- **Max iterations** (doporučeno 3-5)
- **Preset** (Quick/Main/Deep → mapuje na run.sh)
- **Páry** (All 5 / Original 3 / výběr)
- **Strategie změn** (Conservative / Aggressive)

### Architektura

```
AGENT (Claude Code session)
  │ Analyze → Decide → Implement → Launch run
  │ Session KONČÍ
  ▼
WATCHER (scripts/autonomous_watcher.sh)
  │ Poll results/ každých 10 min
  │ Detekce: best_params.json + .autonomous marker
  │ Timeout: 48h max
  ▼
AGENT (nová session, iterace N+1)
  │ Porovná výsledky → BETTER/WORSE/TOP/NEUTRAL
  │ → pokračuje nebo zastaví
```

### Evaluace

| Verdict | Podmínka | Akce |
|---------|----------|------|
| BETTER | Log Calmar +1.5%, no RED, PnL drop <10% | Pokračuje |
| WORSE | Log Calmar -3%, new RED, nebo PnL drop >20% | Rollback, jiný fix |
| TOP | Všechny metriky GREEN/TOP + 2× NEUTRAL | Zastaví, notifikace |
| NEUTRAL 2× | Diminishing returns | Zastaví |

### Co agent smí měnit

**Conservative:** Search space ranges, catastrophic stop %, trail stop ranges
**Aggressive:** + trial count, AWF splits, data window, slippage

**Nikdy nezmění:** strategy.py logiku, objective function, backtest engine, position sizing

### Git isolation

Agent pracuje na `autonomous/iter-N` branchích. `main` se nikdy nezmění.

```bash
git checkout main                        # zahodit vše
git merge autonomous/iter-3              # aplikovat iteraci 3
git branch -D $(git branch --list 'autonomous/*')  # smazat vše
```

### State soubory

```
results/autonomous/
  config.json          — startup konfigurace
  iteration_log.json   — strukturovaná data per-iterace
  changelog.md         — lidsky čitelný log všech změn
  watcher.pid          — PID watcher procesu
```

### Discord notifikace

5 per iterace do `#qre-control`: ANALYZING → IMPLEMENTING → RUN LAUNCHED → verdict → next/stop

### Zastavení

```bash
./run.sh kill                   # killne optimizer i watcher
kill $(cat results/autonomous/watcher.pid)  # jen watcher
```

---

## Adresářová struktura

```
qre/
├── src/qre/
│   ├── config.py
│   ├── optimize.py
│   ├── analyze.py
│   ├── autonomous.py   # Evaluace + state management pro autonomous optimizer
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
│   ├── unit/          # 315 testů
│   ├── integration/
│   └── conftest.py
├── results/           # Výstupy runů
├── logs/              # Log soubory (background runs)
├── .claude/agents/
│   └── autonomous-optimizer.md  # Agent definice
├── run.sh             # Entry point s presety
├── NOTES.md           # Session notes
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
- **Rich** — live monitor TUI
- **pytest** — 315 unit a integračních testů
