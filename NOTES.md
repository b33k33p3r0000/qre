# Session Notes

## 2026-02-27 — Přidání BNB/USDC páru

### Uděláno
- **Brainstorming:** ETH (korelace ~0.9 s BTC, redundantní) vs BNB (korelace ~0.6-0.8, vlastní dynamika) → BNB vybrán pro lepší diversifikaci funded accountu
- **config.py:** BNB/USDC v SYMBOLS, slippage 0.0012, catastrophic stop 10%
- **optimize.py:** BNB/USDC v CLI `--symbol` choices
- **run.sh:** BNB jako volba (4) v custom presetu, `--bnb` flag, `--both` teď zahrnuje všechny 3 páry
- **_export_params.py:** BNB v export symbols listu
- TOTAL_PAIRS zůstává 2 (alokace $50k/pár) — změna na 3 až při live nasazení BNB

### Poznatky
- Architektura QRE je dobře připravená na nové symboly — core moduly (strategy, backtest, metrics, monitor, analyze) nevyžadovaly žádné změny
- 227 testů PASS, žádné regrese

---

## 2026-02-26 — Report: Catastrophic Stop Detail + Sortino Fix

### Uděláno
- **report.py**: Nová sekce "Catastrophic Stop Events" pod Exit Reasons — tabulka per-event (Date, Symbol, Direction, Entry/Exit Price, Hold, P&L) + summary Total
- **metrics.py**: Sortino Ratio přepsán z per-trade returns + sqrt(252) na **daily equity returns + sqrt(365)** — konzistentní s equity-based Sharpe, průmyslový standard
- Regenerace všech 24 existujících reportů v results/
- 218 testů PASS

### Poznatky
- Starý Sortino nadhodnocoval ~40-50% (sqrt(252) na ~130 tradech/rok místo 252)
- Monte Carlo Sharpe to dělal správně (sqrt(trades_per_year)) — Sortino teď následuje lepší vzor (daily returns)

---

## 2026-02-23 (4) — MAIN Run Analýza + Metriky Audit

### Uděláno
- `/analyze-run` na MAIN runech (BTC 25k trials, SOL 25k trials)
- BTC: DEPLOY S VÝHRADOU (Sharpe 2.57, DD -4.92%, 116 trades/yr)
- SOL: REJECT (DD -16.5% > 15% threshold, funded account risk)
- Strategie v4.2.1 běží s `allow_flip=0` (selective mode) — potvrzeno jako stabilnější na základě flip-on vs flip-off A/B testů z 22.2.

### Metriky Audit — Nalezené Bugy
- **Calmar Ratio (metrics.py):** NEní annualizovaný — používá total return místo annual return. Fakticky počítá Recovery Factor, ne Calmar.
- **Sortino Ratio (metrics.py):** Annualizace sqrt(252) na per-trade returnech. Měl by sqrt(trades_per_year) nebo přepis na daily returns.
- **Sharpe Time (metrics.py):** Fees/slippage nejsou v hourly returnech, overwrite místo accumulate pro overlapping trades, np.std vs pandas .std inkonsistence. Nespolehlivý — zůstává secondary-only.
- **Catastrophic stop vs DD:** NENÍ bug — stop limituje per-trade ztrátu, DD je kumulativní přes equity curve. S position_pct=0.25 je max ztráta per trade ~3% equity (SOL: 12% × 25% = 3%). DD 16.5% = ~6 consecutive losing trades.
- **Reference docs (strategy_knowledge.md):** Zastaralé ranges vs aktuální kód (macd_slow 45 vs 50, rsi_lower 20-40 vs 15-45, rsi_upper 60-80 vs 55-85)

### Opraveno
- [x] Fix Calmar: annualizovaný v `calculate_calmar_ratio()` (+ backward compat pro <365d)
- [x] Aktualizovat strategy_knowledge.md (ranges, 10 params, catastrophic_stop info, metrics section)
- [x] Aktualizovat metric_thresholds.md (Calmar/Sortino/Sharpe time poznámky, edge detection ranges)
- [x] Aktualizovat MEMORY.md (10 Optuna params, metriky characteristics)

### Rozhodnutí (nechat jak je)
- Catastrophic stop vs DD: NENÍ bug (kumulativní equity DD vs per-trade stop, position_pct=0.25)
- ~~Sortino nadhodnocený ~40-50%~~ — **OPRAVENO 2026-02-26** (daily returns + sqrt(365))
- Sharpe Time: secondary-only, zdokumentované known issues, nechat

---

## 2026-02-23 (3) — Code Review Fixes (11 nálezů)

### Uděláno
- Kompletní `/review` celého QRE + parent repo → 26 nálezů (4C, 11W, 6I, 5 QRE-specific)
- Implementováno 11 oprav: C1-C4 (critical), W1-W4 (warning), I2-I6 (info), W6
- Design doc: `docs/plans/2026-02-23-qre-review-fixes.md`
- 11 commitů, 204 testů green, 0 regresí

### Klíčové opravy
- **C1-C3**: Smazán `compare_signals.py` (5 crash pointů, kompletně rozbitý)
- **C4**: RSI cache range 5-30 → 3-30 (performance fix, rsi_period=3 je intentional hard floor)
- **W1**: `get_slippage()` env var override odstraněn (per-symbol mapa vždy respektována)
- **W4**: `from __future__ import annotations` ve všech 9 src souborech
- **I3**: `strategy_param_keys` v output pro oddělení Optuna params od metadat
- **I4**: MC win rate loop optimalizace (1000 iterací pro nulový informační zisk)

---

## 2026-02-23 (2) — Post-Run Analýza + Search Space Tuning

### Uděláno
- `/analyze-run` na new-final BTC+SOL runech — oba RE-RUN (params z old search space)
- Search space tuning: `macd_signal` 2-15 → **3-15**, `rsi_lookback` 4-8 → **1-4**
- run.sh presety: BTC Main 20k/5 → 25k/3, SOL Main 40k/5 → 25k/3
- Skill reference docs aktualizovány (Sharpe equity = primary, intentional floors, 3 splits default)

### Poznatky
- rsi_lookback=8 = 9h paměť RSI podmínky — příliš volné, Optuna gamuje trade count přes lookback
- macd_signal=2 = de facto nefiltrovaný MACD — noise
- rsi_period=3 a macd_signal=3 na minimech = **záměrné hard floors**, ne edge param concern
- Sharpe (time) nespolehlivý výpočet → sekundární safety metrika, Sharpe (equity) = primary

---

## 2026-02-23 — /data-art Skill + 8 Vizualizačních Variant

### Uděláno
- Nový Claude Code skill `/data-art` — data-driven algoritmické umění z optimizer dat (optuna.db, trades CSV, best_params.json)
- Data pipeline: `data_extractor.py` → `art_data.json` → p5.js HTML (self-contained, inline data)
- 8 vizualizačních variant: Search Topology, Trade Waveform, Convergence Cosmos, Trial Flow Field, Time Iris, Trial Network, Trade Chords, Run Portrait
- Všech 8 vygenerováno pro flip-on-sol/SOL run jako demo
- Generator skripty v `scripts/gen_art_*.py` pro 3 komplexnější varianty

### Poznatky
- Skill v `~/.claude/skills/data-art/` (4 soubory: SKILL.md, data_extractor.py, viewer.html, visualization_variants.md)
- Art HTML v results/ (gitignored) — velké soubory s embedded daty (50KB-1.1MB)
- Slider UX: min=0 jako "OFF" pro každou vizuální vrstvu funguje dobře

---

## 2026-02-22 (2) — Always-in vs Selective Mode

### Uděláno
- Analýza whipsaw flipů z existujících trade dat: BTC 99% flipů, SOL 91%, krátké flipy (≤12h) konzistentně ztrátové
- A/B empirický test: dva paralelní optimizer runy per symbol (flip-on vs flip-off), identická data
- BTC 20k trials: flip-OFF lepší (+29.7% vs +21.9% P&L, Sharpe(eq) 3.11 vs 2.39, split std 0.26 vs 0.66)
- SOL 35k trials: P&L flip-ON vyšší (+51.7% vs +40.4%), ale Sharpe identický (2.23 vs 2.22), flip-OFF konzistentnější (split std 0.14 vs 0.35)
- Default přepnut na selective (`allow_flip=0`) pro oba symboly — priorita konzistence nad raw P&L
- `run.sh`: přidány `--allow-flip N` a `--always-in` flagy

### Poznatky
- Bimodální search space: Optuna najde dramaticky odlišné params pro flip-on vs flip-off (BTC macd_fast 9.72 vs 1.34)
- Potvrzeno že allow_flip 0-1 jako Optuna param v jednom runu nefunguje (divergentní regiony, TPE nekonverguje)
- MC INSUFFICIENT_DATA pro BTC: splity měly 20-28 trades, threshold byl 30 → snížen `MONTE_CARLO_MIN_TRADES` na 20
- Oba flip-off runy konvergují macd_fast k ~1.0 (edge) — strukturální vzorec, ne overfit

---

## 2026-02-22

### Uděláno
- Analýza catastrophic stop na existujících runech (`scripts/compare_stops.py`) — BTC optimum ~7%, SOL ~12%
- `catastrophic_stop_pct` jako 11. Optuna parametr (range 5-15%, step 0.01)
- Pipeline threading: strategy → optimize → backtest (6 commitů, 184 testů pass)
- Report: dynamický label + bullet chart, fix rsi_period range (5→3), fix default 0.15→0.10
- Strategy version bump 4.1.0 → 4.2.0, pushnuto na GitHub

### TODO / Rozděláno
- Spustit novou optimalizaci s `catastrophic_stop_pct` parametrem (BTC + SOL)
- Porovnat výsledky s předchozími runy (validovat joint optimization efekt)
- SOL 3yr-v2 run zastaven (43k/50k trials) — starý kód bez stop parametru

---

## 2026-02-21 - QRE Code Review & Fixes (Phase 1 + Phase 2)

**Uděláno:**
- Kompletní code review celého qre/ (40 nálezů: 9 critical, 18 warning, 13 info)
- Design doc: `docs/plans/2026-02-21-qre-review-fixes-design.md`
- Phase 1: Opraveno 9 critical nálezů (RSI verifikace, penalties.py smazán, study.best_trial guard, starting_equity do Numba, trades buffer, .get() defaults, webhook unifikace, scoring extrakce, vectorized edge test)
- Phase 2: Opraveno 8 warning nálezů (mrtvé config konstanty, np.random.seed fix, SQLite leak, ETA fix, bullet chart ranges, Discord truncation, dead code, shared conftest.py)
- 195 testů PASS, 6 skipped, 0 failed

**Commity (QRE):** c59c219 (Phase 1), 445e58f (Phase 2)

**TODO:**
- Phase 3 (deferred): I1 magic number 13140, I9 rolling().max() docstring, I10 noqa komentáře
- W17: Testy pro analyze.py (větší scope, samostatná session)

---

## 2026-02-21 - Live Optimizer Monitor

**Uděláno:**
- Brainstorming: live TUI dashboard pro sledování běžících optimizer runů
- Design doc: `docs/plans/2026-02-21-live-monitor-design.md`
- Implementation plan: `docs/plans/2026-02-21-live-monitor-plan.md` (5 tasků)
- Subagent-driven development: 5/5 tasků, 181 testů PASS (11 nových)

**Změny:**
- `monitor.py`: **NOVÝ** modul (~300 řádků) — `find_active_runs()`, `query_db_stats()`, Rich TUI rendering, auto-refresh smyčka
- `optimize.py`: +12 řádků — `trial.set_user_attr()` pro Sharpe/DD/P&L/trades, `study.set_user_attr()` pro n_trials_requested/symbol
- `pyproject.toml`: +2 řádky (rich dep, `qre-monitor` entry point)
- `tests/unit/test_monitor.py`: **NOVÝ** — 11 testů (find_active_runs + query_db_stats)
- `tests/unit/test_optimize.py`: +3 testy pro user_attrs

**Commity (QRE):** a582acc, fa02cd9, 3f71db2, abd3c98

---

## 2026-02-21 - Log Calmar Objective + Anti-Gaming Guards

**Uděláno:**
- Analýza `calmar-v11` runu: raw Calmar objective gamovatelný přes DD minimalizaci (Calmar 90, DD -2.2%, WR 81%, PF 13)
- Brainstorming + design: 3 přístupy → vybrán Log Calmar + DD floor + trade ramp
- Design doc: `docs/plans/2026-02-21-log-calmar-design.md`
- Implementation plan: `docs/plans/2026-02-21-log-calmar-implementation.md` (5 tasků)
- Subagent-driven development: 5/5 tasků, 189 testů PASS
- Validace na runu `calmar-big-v1` — dramatické zlepšení

**Commity (QRE):** f6b0449, 1aa28a4, 489f200, d623dd3

---

## 2026-02-20 - Calmar Objective + Overfitting Fixes

**Uděláno:**
- Analýza výsledků: Optuna konverguje na degenerované params (macd_fast≈1.0, rsi_period=3), Sharpe 14
- Brainstorming + design: `docs/plans/2026-02-20-calmar-objective-design.md`
- Subagent-driven development: 7/7 tasků
- Objective: Sharpe → Calmar ratio + smooth Sharpe decay penalty
- Search space: macd_fast≥2.0, rsi_period≥5, rsi_lookback 4-8, macd_signal≥2
- Purge gap: 50 barů mezi train/test splity (eliminace indicator leakage)

**Commity (QRE):** 26ee0d5, e8c0ad2, 458240e, d2f1c4d, e21fdec, bf3c5b4

---

## 2026-02-17 - /diagnose skill redesign pro v4.0

**Uděláno:**
- Brainstorming + design: audit /diagnose skillu, rebalancování 3 agentů
- Nový Agent 2 "Strategy & Params" — sloučení starého Signal System + param boundary/importance z Agent 3
- 3 nové SQL queries: RSI lookback korelace, trend filter impact, trend TF comparison
- Subagent-driven development: 6/6 tasků
- Design doc: `docs/plans/2026-02-17-diagnose-redesign-design.md`

---

## 2026-02-17 - v4.0 Tuning: RSI/MACD ranges + Sharpe capping + time-in-market

**Uděláno:**
- `/diagnose` na dvou v4.0 runech (BTC v2-1, SOL v2-2) — oba REVIEW
- Zjištěno: Optuna obchází RSI filtr (rsi_lower=51-55, lookback=23 → triviálně true)
- RSI ranges zúženy na [25-35]/[65-75], MACD dolní hranice rozšířeny
- Nová metrika `time_in_market`, Sharpe capping v report.py
- 231 testů PASS

---

## 2026-02-17 - Quant Whale Strategy v4.0 implementace

**Uděláno:**
- `/diagnose` na dvou runech — oba FAIL (overfit, málo trades, závislost na mega-trades)
- Design doc: `docs/plans/2026-02-17-chio-extreme-v4-design.md` — RSI lookback + multi-TF trend filter
- Subagent-driven development: 8/8 tasků, merge do main, 216 testů
- Přejmenování "Chio Extreme" → "Quant Whale Strategy"

**Klíčové změny v4.0:**
- RSI lookback window, Multi-TF trend filter, Data pipeline 1H + 4H/8H/1D
- 9 Optuna parametrů (6 original + rsi_lookback + trend_tf + trend_strict)

---

## 2026-02-16 - Strategy Redesign brainstorming

**Uděláno:**
- Kompletní audit QRE kódu (strategy, indicators, backtest, metrics, penalties, config, optimize)
- Revize všech 50+ parametrů — rozhodnutí keep/remove/defer
- Design doc: `docs/plans/2026-02-16-strategy-redesign-design.md`

---

## 2026-02-15 - Run Analysis notifikace redesign

**Uděláno:**
- Redesign `build_discord_embed()` — code block formát s `[ok]/[!!]/[XX]` tagy
- Sloučení kanálů: všechny notifikace do `#qre-runs`
- Design doc: `docs/plans/2026-02-15-run-analysis-notification-design.md`

---

## 2026-02-15 - AWF test_size parametr + preset update

**Uděláno:**
- test_size hardcoded na 10% = ~10 trades per test split (statisticky bezcenné)
- feat(optimize): parametr `--test-size` (default 0.15)
- run.sh: Deep + Über presety sníženy z 4 na 3 splity

---

## 2026-02-15 - BTC Market Analysis + Training Data Strategy

**Uděláno:**
- BTC 4Y drawdown analýza: 3 major propady (-65.4%, -20.5%, -47.6%)
- Rozhodnutí: 2yr training window (Jan 2024 - Dec 2025) + skip-recent 1080h
- Death Cross Guard (EE) pokrývá major crash (>25%)

---

## 2026-02-14 - MASTER PLAN finalizace + plány audit

**Uděláno:**
- MASTER PLAN označen jako COMPLETED (všech 6 fází done)
- 8 sub-plánů aktualizováno se statusy (COMPLETED / SUPERSEDED / PARTIAL)

---

## 2026-02-13 - EE Bugfix: MT5 symbol mapping + retry limit + ACCOUNT_SIZE

**Uděláno:**
- Fix MT5 symbol mapping: `BTCUSD` → `BTC/USD` (broker BrightFunded používá lomítko)
- Fix retry loop: `MAX_CONSECUTIVE_ERRORS = 3`
- Fix ACCOUNT_SIZE: revert 100k → 10k

---

## 2026-02-12 - Death Cross Guard + Phase 5 Finalizace

**Uděláno:**
- Death Cross Guard v EE: `guard/detector.py` (SMA 50/200) + `guard/actions.py`
- VPS cleanup: smazán `TradingBot-SignalServer` service, staré Scheduled Tasks
- MASTER-PLAN: EE DoD komplet

---

## 2026-02-12 - Phase 5 Cleanup

**Uděláno:**
- Archivace `C:\trading\trading-bot\` → `trading-bot-ARCHIVED` na VPS
- Fix EE failing test, přepis `vps-status.sh` pro EE
- MASTER-PLAN: Phase 5 DONE

---

## 2026-02-12 - Discord Control Bot (WHAL3 CONTROL)

**Uděláno:**
- EE Discord Bot: 5 slash commands (`/ee status`, `/ee pnl`, `/ee pause`, `/ee resume`, `/ee kill`)
- Flag-based IPC: PAUSE/STOP flag soubory v `state/`
- DNS fix: odinstalace `aiodns` (IPv6 DNS na VPS nefunkční)

---

## 2026-02-11 - INBOX fixy: run.sh časy, Sharpe bugfix, MASTER PLAN formát

**Uděláno:**
- run.sh: preset časy aktualizovány
- fix(metrics): MC annualizace sqrt(252) → sqrt(trades_per_year)
- fix(optimize): reporting přepnuto na time-based Sharpe

---

## 2026-02-11 - Phase 5: EE (Execution Engine) na VPS

**Uděláno:**
- EE repo na VPS, 12 modulů, 147 testů, GitHub `b33k33p3r0000/ee`
- Cutover: Task Scheduler (ONLOGON), BTC+SOL live na BrightFunded

---

## 2026-02-11 - QRE Phase 4: Stabilizace + v0.4.0

**Uděláno:**
- Signal comparison: BTC i SOL PASS (100% identické vs legacy optimizer)
- Config fix: MIN_TRADES_TEST_HARD 15→8, p_buy ranges zúženy

---

## 2026-02-11 - QRE Phase 3: Reporting

**Uděláno:**
- report.py: Self-contained HTML report s Plotly, 12 testů
- notify.py: Discord notifikace, 10 testů
- run.sh: CLI skript se 4 presety

---

## 2026-02-11 - QRE Phase 2: Orchestrace

**Uděláno:**
- optimize.py: AWF orchestrátor (~300 řádků rewrite z 2633-řádkového optimizeru)
- penalties.py, data/fetch.py, io.py, hooks — celkem 93 testů

---

## 2026-02-10 - QRE Phase 0+1: Repo + Core Audit

**Uděláno:**
- Vytvořeno QRE git repo + GitHub remote (private)
- Přeneseny 4 core moduly s auditem, 37 testů
- Golden baseline validace: identické výsledky s optimizerem

---

## 2026-02-10 - QUANT WHAL3 2.0 Master Plan + Drawdown Bug

**Uděláno:**
- Diagnostika max drawdown bugu, brainstorming QUANT WHAL3 2.0
- Master Plan vytvořen: `docs/plans/2026-02-10-quant-whale-2.0-MASTER-PLAN.md`
- Rozhodnutí: QRE first, EE later, sentinel vypnout

---
