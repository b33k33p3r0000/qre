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
  1) Test        —  5k trials, ~3yr, 3 splits, BTC+SOL
  2) BTC Main    — 30k trials, ~3yr, 3 splits, BTC only
  3) SOL Main    — 30k trials, ~3yr, 3 splits, SOL only
  4) Custom      — You choose everything

All presets use --hours 26280 --skip-recent 0 by default
(~3yr data). Override with --full or manual flags.
All runs start in background by default (use --fg for foreground).

Pairs:
  --btc                BTC/USDC only
  --sol                SOL/USDC only
  --both               Both pairs (default)

Options:
  --trials N           Override trial count
  --hours N            Override history length in hours
  --splits N           Override number of AWF splits
  --skip-recent N      Skip most recent N hours from training data
  --tag NAME           Run tag (e.g. 'test-v1')
  --full               Full data (--hours 26280 --skip-recent 0)
  --allow-flip N       0=selective (default), 1=always-in
  --always-in          Shortcut for --allow-flip 1
  --warm-start PATH    Path to best_params.json for warm-starting optimization
  --fg                 Run in foreground (default: background)

Examples:
  ./run.sh 1                          # Test, BTC+SOL, background
  ./run.sh 2                          # BTC Main, background
  ./run.sh 3                          # SOL Main, background
  ./run.sh 2 --fg                     # BTC Main, foreground
  ./run.sh 4 --btc --trials 8000      # Custom trials

Process management:
  ./run.sh attach          Attach to running/latest log
  ./run.sh kill            Kill running optimizer
  ./run.sh logs            List recent log files

EOF
}

# =============================================================================
# DEFAULTS (~3yr window)
# =============================================================================

TRIALS=15000
HOURS=26280
SPLITS=""
PAIRS="both"
TAG=""
PRESET=""
SKIP_RECENT=0
ALLOW_FLIP=0
WARM_START=""
FOREGROUND=false
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# =============================================================================
# PARSE ARGS
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        1) PRESET="test" ;;
        2) PRESET="btc-main" ;;
        3) PRESET="sol-main" ;;
        4) PRESET="custom" ;;
        --btc) PAIRS="btc" ;;
        --sol) PAIRS="sol" ;;
        --both) PAIRS="both" ;;
        --trials) TRIALS="$2"; shift ;;
        --hours) HOURS="$2"; shift ;;
        --splits) SPLITS="$2"; shift ;;
        --tag) TAG="$2"; shift ;;
        --skip-recent) SKIP_RECENT="$2"; shift ;;
        --full) HOURS=26280; SKIP_RECENT=0 ;;
        --allow-flip) ALLOW_FLIP="$2"; shift ;;
        --always-in) ALLOW_FLIP=1 ;;
        --warm-start) WARM_START="$2"; shift ;;
        --bg) FOREGROUND=false ;;  # already default, kept for explicitness
        --fg) FOREGROUND=true ;;
        attach)
            # Find actively written logs (modified in last 5 min)
            ACTIVE_LOGS=()
            while IFS= read -r f; do
                ACTIVE_LOGS+=("$f")
            done < <(find "$LOG_DIR" -name "*.log" -mmin -5 -type f 2>/dev/null | sort -r)

            if [ ${#ACTIVE_LOGS[@]} -eq 0 ]; then
                # No active logs — fall back to latest
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
            # Find QRE optimizer processes (macOS-compatible)
            PIDS=()
            CMDS=()
            while IFS= read -r pid; do
                cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
                PIDS+=("$pid")
                CMDS+=("$cmd")
            done < <(pgrep -f "python.*qre\.optimize" 2>/dev/null || true)

            # Escalating kill: SIGTERM -> wait 3s -> SIGKILL
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
    test)       TRIALS=5000;  SPLITS=3 ;;
    btc-main)   TRIALS=30000; SPLITS=3; PAIRS="btc" ;;
    sol-main)   TRIALS=30000; SPLITS=3; PAIRS="sol" ;;
    custom)     ;; # Use --trials, --hours, --splits from args
    "")
        # Interactive mode
        echo ""
        show_help
        read -p "Select preset (1-4): " choice
        case "$choice" in
            1) TRIALS=5000;  SPLITS=3 ;;
            2) TRIALS=30000; SPLITS=3; PAIRS="btc" ;;
            3) TRIALS=30000; SPLITS=3; PAIRS="sol" ;;
            4)
                read -p "Trials [15000]: " TRIALS; TRIALS="${TRIALS:-15000}"
                read -p "Hours [26280]: " HOURS; HOURS="${HOURS:-26280}"
                read -p "Splits [3]: " SPLITS; SPLITS="${SPLITS:-3}"
                read -p "Skip recent hours [0]: " SKIP_RECENT; SKIP_RECENT="${SKIP_RECENT:-0}"
                read -p "Warm-start (path to best_params.json, empty=none): " WARM_START
                echo ""
                read -p "Pairs — (1) BTC only, (2) SOL only, (3) Both [3]: " pair_choice
                case "${pair_choice:-3}" in
                    1) PAIRS="btc" ;;
                    2) PAIRS="sol" ;;
                    3) PAIRS="both" ;;
                esac
                ;;
            *) echo "Invalid choice"; exit 1 ;;
        esac

        # Only ask for pairs in interactive if preset doesn't set them
        if [ "$choice" = "1" ]; then
            echo ""
            read -p "Pairs — (1) BTC only, (2) SOL only, (3) Both [3]: " pair_choice
            case "${pair_choice:-3}" in
                1) PAIRS="btc" ;;
                2) PAIRS="sol" ;;
                3) PAIRS="both" ;;
            esac
        fi

        read -p "Run tag (optional, e.g. 'test-v1'): " TAG
        ;;
esac

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
    if [ -n "$WARM_START" ]; then
        cmd="$cmd --warm-start $WARM_START"
    fi
    echo "$cmd"
}

# =============================================================================
# RUN
# =============================================================================

MODE_LABEL="background"
$FOREGROUND && MODE_LABEL="foreground"

echo ""
echo "═══════════════════════════════════════════"
echo "  QRE Optimizer — MACD+RSI AWF"
echo "═══════════════════════════════════════════"
echo "  Trials:  $TRIALS"
echo "  Hours:   $HOURS (~$((HOURS / 24)) days)"
[ "$SKIP_RECENT" -gt 0 ] 2>/dev/null && echo "  Skip:    ${SKIP_RECENT}h (~$((SKIP_RECENT / 24)) days recent data excluded)"
[ -n "$SPLITS" ] && echo "  Splits:  $SPLITS"
[ -n "$TAG" ] && echo "  Tag:     $TAG"
echo "  Pairs:   $PAIRS"
echo "  Flip:    $( [ "$ALLOW_FLIP" = "1" ] && echo "always-in" || echo "selective" )"
[ -n "$WARM_START" ] && echo "  Warm:    $WARM_START"
echo "  Mode:    $MODE_LABEL"
echo "═══════════════════════════════════════════"
echo ""

# Build list of symbols to run
SYMBOLS=()
case "$PAIRS" in
    btc)  SYMBOLS=("BTC/USDC") ;;
    sol)  SYMBOLS=("SOL/USDC") ;;
    both) SYMBOLS=("BTC/USDC" "SOL/USDC") ;;
esac

if $FOREGROUND; then
    # Foreground mode: run directly in terminal
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
    # Background mode: nohup + log file
    TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
    LOG_FILE="$LOG_DIR/qre_${TIMESTAMP}.log"

    # Build combined command for all symbols
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
    echo "  kill -2 $BG_PID          # Graceful stop (finishes report + notifications)"
    echo "  kill $BG_PID             # Hard kill (no report)"
fi
