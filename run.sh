#!/usr/bin/env bash
# QRE Run Script — MACD+RSI AWF Optimizer
# Usage: ./run.sh [preset] [options]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# =============================================================================
# HELP
# =============================================================================

show_help() {
    cat << 'EOF'
QRE Optimizer — MACD+RSI AWF
=============================

Presets:
  1) Test        —  5k trials, 1yr, 3 splits, BTC only (~15 min)
  2) Quick       — 15k trials, 2yr, 3 splits, BTC+SOL (~1-2 hr)
  3) Main        — 40k trials, 3yr, 3 splits, BTC+SOL+BNB (~4-8 hr)
  4) Deep        — 50k trials, 5yr, 5 splits, BTC+SOL+BNB (~12-24 hr)
  5) Custom      — You choose everything

All runs start in background by default (use --fg for foreground).

Pairs:
  --btc                BTC/USDT only
  --sol                SOL/USDT only
  --bnb                BNB/USDT only
  --all                All pairs (default for main/deep)

Options:
  --trials N           Override trial count
  --hours N            Override history length in hours
  --splits N           Override number of AWF splits
  --skip-recent N      Skip most recent N hours from training data
  --tag NAME           Run tag (e.g. 'test-v1')
  --allow-flip N       0=selective (default), 1=always-in
  --always-in          Shortcut for --allow-flip 1
  --warm-start PATH    Path to best_params.json for warm-starting optimization
  --fg                 Run in foreground (default: background)

Examples:
  ./run.sh 1                          # Test, BTC, background
  ./run.sh 2                          # Quick, BTC+SOL, background
  ./run.sh 3                          # Main, all pairs, background
  ./run.sh 4 --btc --fg               # Deep, BTC only, foreground
  ./run.sh 5 --btc --trials 8000      # Custom

Process management:
  ./run.sh attach          Attach to running/latest log
  ./run.sh kill            Kill running optimizer
  ./run.sh logs            List recent log files

EOF
}

# =============================================================================
# DEFAULTS
# =============================================================================

TRIALS=15000
HOURS=26280
SPLITS=""
PAIRS=""
TAG=""
PRESET=""
SKIP_RECENT=0
ALLOW_FLIP=0
WARM_START=""
WARM_BTC=""
WARM_SOL=""
WARM_BNB=""
FOREGROUND=false
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# =============================================================================
# WARM-START PICKER (per-symbol)
# =============================================================================

_WARM_RESULT=""
pick_warm_start() {
    local sym_filter="$1"  # BTC, SOL, BNB
    _WARM_RESULT=""
    local ws_table
    ws_table=$(venv/bin/python -c "
import json, sys
from pathlib import Path
sym_filter = sys.argv[1]
runs = sorted(Path('results').rglob('best_params.json'), reverse=True)
runs = [r for r in runs if 'checkpoints' not in str(r)]
runs = [r for r in runs if r.parts[-2] == sym_filter]
for i, r in enumerate(runs[:8]):
    d = json.loads(r.read_text())
    run_dir = r.parts[-3] if r.parts[-2] in ('BTC','SOL','BNB') else r.parts[-2]
    export = ' [LIVE]' if 'EXPORT' in str(r) else ''
    sharpe = d.get('sharpe_equity', 0)
    dd = d.get('max_drawdown', 0)
    trials = d.get('n_trials', 0)
    print(f'  {i+1:2d}) Sh={sharpe:.2f}  DD={dd:.1f}%  {trials//1000}k  {run_dir}{export}')
" "$sym_filter" 2>/dev/null)
    if [ -n "$ws_table" ]; then
        echo "$ws_table"
        echo "   0) Bez warm-start"
        echo ""
        read -p "  Vyber run [0]: " ws_pick
        ws_pick="${ws_pick:-0}"
        if [ "$ws_pick" != "0" ]; then
            _WARM_RESULT=$(venv/bin/python -c "
import sys
from pathlib import Path
sym_filter = sys.argv[2]
runs = sorted(Path('results').rglob('best_params.json'), reverse=True)
runs = [r for r in runs if 'checkpoints' not in str(r)]
runs = [r for r in runs if r.parts[-2] == sym_filter]
idx = int(sys.argv[1]) - 1
if 0 <= idx < len(runs):
    print(runs[idx])
else:
    sys.exit(1)
" "$ws_pick" "$sym_filter" 2>/dev/null)
            if [ -z "$_WARM_RESULT" ]; then
                echo "  Neplatná volba, pokračuji bez warm-start"
            fi
        fi
    else
        echo "  Žádné předchozí runy."
    fi
}

# =============================================================================
# PARSE ARGS
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        1) PRESET="test" ;;
        2) PRESET="quick" ;;
        3) PRESET="main" ;;
        4) PRESET="deep" ;;
        5) PRESET="custom" ;;
        --btc) PAIRS="btc" ;;
        --sol) PAIRS="sol" ;;
        --bnb) PAIRS="bnb" ;;
        --all) PAIRS="all" ;;
        --trials) TRIALS="$2"; shift ;;
        --hours) HOURS="$2"; shift ;;
        --splits) SPLITS="$2"; shift ;;
        --tag) TAG="$2"; shift ;;
        --skip-recent) SKIP_RECENT="$2"; shift ;;
        --allow-flip) ALLOW_FLIP="$2"; shift ;;
        --always-in) ALLOW_FLIP=1 ;;
        --warm-start) WARM_START="$2"; shift ;;
        --bg) FOREGROUND=false ;;
        --fg) FOREGROUND=true ;;
        attach)
            # Find actively written logs (modified in last 5 min)
            ACTIVE_LOGS=()
            while IFS= read -r f; do
                ACTIVE_LOGS+=("$f")
            done < <(find "$LOG_DIR" -name "*.log" -mmin -5 -type f 2>/dev/null | sort -r)

            if [ ${#ACTIVE_LOGS[@]} -eq 0 ]; then
                LATEST=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
                if [ -z "$LATEST" ]; then
                    echo "No log files found in $LOG_DIR"
                    exit 1
                fi
                echo "No active runs. Showing latest log:"
                echo "  $(basename "$LATEST")"
                echo "(Ctrl+C to detach)"
                echo ""
                tail -f "$LATEST"
            elif [ ${#ACTIVE_LOGS[@]} -eq 1 ]; then
                echo "Attaching to: $(basename "${ACTIVE_LOGS[0]}")"
                echo "(Ctrl+C to detach — run continues in background)"
                echo ""
                tail -f "${ACTIVE_LOGS[0]}"
            else
                echo "Multiple active runs detected:"
                echo ""
                for i in "${!ACTIVE_LOGS[@]}"; do
                    LOG_NAME=$(basename "${ACTIVE_LOGS[$i]}")
                    LAST_LINE=$(tail -1 "${ACTIVE_LOGS[$i]}" 2>/dev/null | head -c 80 || true)
                    echo "  $((i+1))) $LOG_NAME"
                    [ -n "$LAST_LINE" ] && echo "     $LAST_LINE"
                done
                echo ""
                read -p "Select run (1-${#ACTIVE_LOGS[@]}): " pick
                pick=$((pick - 1))
                if [ "$pick" -ge 0 ] && [ "$pick" -lt ${#ACTIVE_LOGS[@]} ]; then
                    echo ""
                    echo "Attaching to: $(basename "${ACTIVE_LOGS[$pick]}")"
                    echo "(Ctrl+C to detach — run continues in background)"
                    echo ""
                    tail -f "${ACTIVE_LOGS[$pick]}"
                else
                    echo "Invalid choice"
                    exit 1
                fi
            fi
            exit 0
            ;;
        logs)
            echo "Recent logs:"
            ls -lht "$LOG_DIR"/*.log 2>/dev/null | head -10
            exit 0
            ;;
        kill)
            PIDS=()
            CMDS=()
            while IFS= read -r pid; do
                cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
                PIDS+=("$pid")
                CMDS+=("$cmd")
            done < <(pgrep -f "python.*qre\.optimize" 2>/dev/null || true)

            _kill_pid() {
                local pid=$1
                kill "$pid" 2>/dev/null || true
                echo "Sent SIGTERM to PID $pid..."
                for i in 1 2 3; do
                    sleep 1
                    if ! kill -0 "$pid" 2>/dev/null; then
                        echo "PID $pid terminated."
                        return 0
                    fi
                done
                echo "Still running — sending SIGKILL..."
                kill -9 "$pid" 2>/dev/null || true
                sleep 0.5
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo "PID $pid killed."
                else
                    echo "WARNING: PID $pid may still be running."
                fi
            }

            if [ ${#PIDS[@]} -eq 0 ]; then
                echo "No QRE optimizer runs found."
                exit 0
            elif [ ${#PIDS[@]} -eq 1 ]; then
                echo "Killing QRE run (PID ${PIDS[0]}):"
                echo "  ${CMDS[0]}"
                echo ""
                _kill_pid "${PIDS[0]}"
            else
                echo "Multiple QRE runs detected:"
                echo ""
                for i in "${!PIDS[@]}"; do
                    echo "  $((i+1))) PID ${PIDS[$i]}: ${CMDS[$i]}"
                done
                echo "  a) Kill all"
                echo ""
                read -p "Select run to kill (1-${#PIDS[@]}, a=all): " pick
                if [ "$pick" = "a" ] || [ "$pick" = "A" ]; then
                    for pid in "${PIDS[@]}"; do
                        _kill_pid "$pid"
                    done
                else
                    idx=$((pick - 1))
                    if [ "$idx" -ge 0 ] && [ "$idx" -lt ${#PIDS[@]} ]; then
                        _kill_pid "${PIDS[$idx]}"
                    else
                        echo "Invalid choice"
                        exit 1
                    fi
                fi
            fi
            exit 0
            ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
    shift
done

# =============================================================================
# APPLY PRESET
# =============================================================================

case "$PRESET" in
    test)
        TRIALS=5000; HOURS=8760; SPLITS=3
        [ -z "$PAIRS" ] && PAIRS="btc"
        ;;
    quick)
        TRIALS=15000; HOURS=17520; SPLITS=3
        [ -z "$PAIRS" ] && PAIRS="btc-sol"
        ;;
    main)
        TRIALS=40000; HOURS=26280; SPLITS=3
        [ -z "$PAIRS" ] && PAIRS="all"
        ;;
    deep)
        TRIALS=50000; HOURS=43800; SPLITS=5
        [ -z "$PAIRS" ] && PAIRS="all"
        ;;
    custom) ;; # Use --trials, --hours, --splits from args
    "")
        # Interactive mode
        echo ""
        show_help
        read -p "Select preset (1-5): " choice
        case "$choice" in
            1) TRIALS=5000;  HOURS=8760;  SPLITS=3; [ -z "$PAIRS" ] && PAIRS="btc" ;;
            2) TRIALS=15000; HOURS=17520; SPLITS=3; [ -z "$PAIRS" ] && PAIRS="btc-sol" ;;
            3) TRIALS=40000; HOURS=26280; SPLITS=3; [ -z "$PAIRS" ] && PAIRS="all" ;;
            4) TRIALS=50000; HOURS=43800; SPLITS=5; [ -z "$PAIRS" ] && PAIRS="all" ;;
            5)
                read -p "Trials [15000]: " TRIALS; TRIALS="${TRIALS:-15000}"
                read -p "Hours [26280]: " HOURS; HOURS="${HOURS:-26280}"
                read -p "Splits [3]: " SPLITS; SPLITS="${SPLITS:-3}"
                read -p "Skip recent hours [0]: " SKIP_RECENT; SKIP_RECENT="${SKIP_RECENT:-0}"
                echo ""
                read -p "Pairs — (1) BTC, (2) SOL, (3) BNB, (4) All [4]: " pair_choice
                case "${pair_choice:-4}" in
                    1) PAIRS="btc" ;;
                    2) PAIRS="sol" ;;
                    3) PAIRS="bnb" ;;
                    4) PAIRS="all" ;;
                esac

                # Warm-start picker per symbol
                echo ""
                echo "Warm-start (enqueue known-good params as first trial):"
                case "$PAIRS" in
                    btc)
                        echo "— BTC:"
                        pick_warm_start "BTC"
                        WARM_BTC="$_WARM_RESULT"
                        ;;
                    sol)
                        echo "— SOL:"
                        pick_warm_start "SOL"
                        WARM_SOL="$_WARM_RESULT"
                        ;;
                    bnb)
                        echo "— BNB:"
                        pick_warm_start "BNB"
                        WARM_BNB="$_WARM_RESULT"
                        ;;
                    all|btc-sol)
                        for sym in BTC SOL BNB; do
                            [[ "$PAIRS" = "btc-sol" && "$sym" = "BNB" ]] && continue
                            echo "— $sym:"
                            pick_warm_start "$sym"
                            eval "WARM_$sym=\"\$_WARM_RESULT\""
                        done
                        ;;
                esac
                ;;
            *) echo "Invalid choice"; exit 1 ;;
        esac

        read -p "Run tag (optional, e.g. 'test-v1'): " TAG
        ;;
esac

# Default pairs if still empty
[ -z "$PAIRS" ] && PAIRS="all"

# =============================================================================
# BUILD COMMAND
# =============================================================================

build_cmd() {
    local symbol="$1"
    local cmd="venv/bin/python -m qre.optimize --symbol $symbol --trials $TRIALS --hours $HOURS"
    if [ -n "$SPLITS" ]; then
        cmd="$cmd --splits $SPLITS"
    fi
    if [ -n "$TAG" ]; then
        cmd="$cmd --tag $TAG"
    fi
    if [ "$SKIP_RECENT" -gt 0 ] 2>/dev/null; then
        cmd="$cmd --skip-recent $SKIP_RECENT"
    fi
    cmd="$cmd --allow-flip $ALLOW_FLIP"
    # Warm-start: CLI --warm-start overrides, then per-symbol from interactive picker
    local warm=""
    if [ -n "$WARM_START" ]; then
        warm="$WARM_START"
    else
        case "$symbol" in
            BTC/USDT) warm="$WARM_BTC" ;;
            SOL/USDT) warm="$WARM_SOL" ;;
            BNB/USDT) warm="$WARM_BNB" ;;
        esac
    fi
    if [ -n "$warm" ]; then
        cmd="$cmd --warm-start $warm"
    fi
    echo "$cmd"
}

# =============================================================================
# RUN
# =============================================================================

MODE_LABEL="background"
$FOREGROUND && MODE_LABEL="foreground"

# Build list of symbols to run
SYMBOLS=()
case "$PAIRS" in
    btc)     SYMBOLS=("BTC/USDT") ;;
    sol)     SYMBOLS=("SOL/USDT") ;;
    bnb)     SYMBOLS=("BNB/USDT") ;;
    btc-sol) SYMBOLS=("BTC/USDT" "SOL/USDT") ;;
    all)     SYMBOLS=("BTC/USDT" "SOL/USDT" "BNB/USDT") ;;
esac

echo ""
echo "═══════════════════════════════════════════"
echo "  QRE Optimizer — MACD+RSI AWF"
echo "═══════════════════════════════════════════"
echo "  Trials:  $TRIALS"
echo "  Hours:   $HOURS (~$((HOURS / 8760))yr)"
[ "$SKIP_RECENT" -gt 0 ] 2>/dev/null && echo "  Skip:    ${SKIP_RECENT}h (~$((SKIP_RECENT / 24)) days excluded)"
[ -n "$SPLITS" ] && echo "  Splits:  $SPLITS"
[ -n "$TAG" ] && echo "  Tag:     $TAG"
echo "  Pairs:   ${SYMBOLS[*]}"
echo "  Flip:    $( [ "$ALLOW_FLIP" = "1" ] && echo "always-in" || echo "selective" )"
if [ -n "$WARM_START" ]; then
    echo "  Warm:    $WARM_START (all symbols)"
else
    [ -n "$WARM_BTC" ] && echo "  Warm BTC: $WARM_BTC"
    [ -n "$WARM_SOL" ] && echo "  Warm SOL: $WARM_SOL"
    [ -n "$WARM_BNB" ] && echo "  Warm BNB: $WARM_BNB"
fi
echo "  Mode:    $MODE_LABEL"
echo "═══════════════════════════════════════════"
echo ""

if $FOREGROUND; then
    for sym in "${SYMBOLS[@]}"; do
        cmd=$(build_cmd "$sym")
        echo ">>> Running: $cmd"
        echo ""
        eval "$cmd"
        echo ""
        echo ">>> Done: $sym"
        echo ""
    done
    echo "═══════════════════════════════════════════"
    echo "  All runs complete!"
    echo "═══════════════════════════════════════════"
else
    TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
    LOG_FILE="$LOG_DIR/qre_${TIMESTAMP}.log"

    BG_CMDS=""
    for sym in "${SYMBOLS[@]}"; do
        cmd=$(build_cmd "$sym")
        if [ -n "$BG_CMDS" ]; then
            BG_CMDS="$BG_CMDS && echo '' && echo '>>> Done: $sym' && echo '' && "
        fi
        BG_CMDS="${BG_CMDS}echo '>>> Running: $cmd' && echo '' && $cmd && echo '' && echo '>>> Done: $sym'"
    done

    nohup bash -c "cd $SCRIPT_DIR && source venv/bin/activate && $BG_CMDS && echo '' && echo 'All runs complete!'" > "$LOG_FILE" 2>&1 &
    BG_PID=$!

    echo "Started in background (PID: $BG_PID)"
    echo "Log: $LOG_FILE"
    echo ""
    echo "Commands:"
    echo "  ./run.sh attach          # Watch live output"
    echo "  ./run.sh logs            # List log files"
    echo "  tail -f $LOG_FILE        # Direct tail"
    echo "  kill -2 $BG_PID          # Graceful stop"
fi
