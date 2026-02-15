# QRE — Quantitative Research Engine

MACD+RSI optimizer pro BTC/USDC a SOL/USDC. Optuna TPE s Anchored Walk-Forward validací.

## Quick Start

```bash
./run.sh 3 --btc --fg                    # Production preset, BTC, foreground
./run.sh 6 --btc --trials 25000 --tag x  # Custom trials + tag
./run.sh attach                           # Attach k running runu
```

## Moduly

| Modul | Popis |
|-------|-------|
| `optimize.py` | AWF orchestrátor — Optuna TPE + SHA pruner |
| `core/strategy.py` | MACD+RSI strategie v2.0 (6-TF voting) |
| `core/backtest.py` | Event-driven backtest engine |
| `analyze.py` | Auto-analýza výsledků + Discord notifikace |
| `report.py` | Self-contained HTML report (Plotly) |
| `penalties.py` | 5 typů penalt pro objective funkci |
| `data/fetch.py` | Binance OHLCV fetch + Parquet cache |

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

### Recent Drawdowns (detail), 2022-2026

```
                    BTC/USD — Major Drawdowns (4Y)

  $130k ┤                                          ╭──╮
        │                                         ╱    ╲
  $110k ┤                                        ╱      ╲
        │                                       ╱    ③   ╲
   $90k ┤                                  ╭───╯  -47.6%  ╲
        │                                 ╱                 ╲
   $70k ┤                    ╭──╮        ╱                   ╰── $70k
        │                   ╱  ② ╲╭────╯
   $50k ┤╲                 ╱ -20.5%
        │  ╲               ╱
   $30k ┤   ╲  ①         ╱
        │    ╲ -65.4%   ╱
   $16k ┤     ╰───────╯
        ├────────┼────────┼────────┼────────┤
       2022    2023     2024     2025     2026
```

| # | Období | Peak | Trough | Propad | Trvání | Recovery |
|---|--------|------|--------|--------|--------|----------|
| 1 | Bear 2022 | $47,620 (Jan 2022) | $16,500 (Nov 2022) | **-65.4%** | 318 dní | 443 dní (Feb 2024) |
| 2 | Korekce Q2 2024 | $73,000 (Mar 2024) | $58,000 (Aug 2024) | **-20.5%** | 167 dní | 108 dní (Dec 2024) |
| 3 | Propad Q4 2025 | $126,000 (Oct 2025) | $66,000 (Feb 2026) | **-47.6%** | 123 dní | dosud nerecoverd |

Roční performance: 2022 -65.3% | 2023 +151.5% | 2024 +128.2% | 2025 -8.8%

*Zdroj: Binance/Yahoo Finance, generováno 2026-02-15*

### Rationale: Proč 2yr training window + skip-recent

**Problém:** Optuna optimalizuje na celém datasetu včetně crash období. Výsledek = parametry optimalizované na nerelevantní tržní režimy, vysoký overfit.

**Zjištění z diagnóz (Feb 2026):**

| Run | Trials | Data | Overfit | Train/Test diff | Verdict |
|-----|--------|------|---------|-----------------|---------|
| afterdiag-v7 | 25k | FULL | 0.60 | 2.31 | FAIL (5 RED) |
| 10k-v1 | 10k | FULL | 0.89 | 3.27 | FAIL |
| skip-recent-v1 | 10k | skip 1080h | **0.44** | **1.71** | **0 RED** |

**Klíčový poznatek:** Problém nebyl v počtu trials, ale v datech. Anomální Q4 2025 crash (-47.6%) otravoval trénink.

**Řešení — dvě vrstvy ochrany:**

1. **Death Cross Guard (EE):** SMA 50/200 crossover automaticky pausne trading při major crash (>25% drawdown). Strategie *nemusí* být trénována na crash období — EE to řeší na execution úrovni.

2. **Training window:** 2 roky (Jan 2024 - Dec 2025) + `--skip-recent 1080` (45 dní)
   - **Zahrnuje** korekci Q2 2024 (-20.5%) — normální tržní chování, strategie by ho měla zvládnout
   - **Vynechává** Bear 2022 (-65.4%) — příliš starý, jiný tržní režim
   - **Vynechává** Q4 2025 crash (-47.6%) — death cross guard to pokryje
   - **Skipuje** Jan-Feb 2026 — anomální post-crash období zkresluje optimalizaci

**Production run command:**

```bash
./run.sh 4 --btc --tag 2yr-skip-prod
```

- `--hours 18600` = ~2 roky dat (775 dní)
- `--skip-recent 1080` = ořízne posledních 45 dní z konce datasetu
- Efektivní tréninkové okno: ~Jan 2024 — Dec 2025

### Target metriky

| Metrika | Cíl | Poznámka |
|---------|-----|----------|
| Overfit score | < 0.40 | skip-recent-v1 dosáhl 0.44 |
| Train/Test diff | < 1.5 | skip-recent-v1 dosáhl 1.71 |
| Sharpe | 1.5 - 3.0 | Nižší = méně overfit |
| Trades/year | > 80 | Statisticky relevantní |
| Splits positive | všechny | Konzistentní OOS výkon |
