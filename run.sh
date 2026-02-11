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
# PRESETS
# =============================================================================

show_presets() {
    cat << 'EOF'
QRE Optimizer — MACD+RSI AWF
=============================

Presets:
  1) Quick      —  2k trials, 1yr, 2 splits     (~5 min)
  2) Standard   —  5k trials, 1yr, 3 splits     (~15 min)
  3) Production — 10k trials, 1yr, 3 splits     (~30 min)
  4) Deep       — 15k trials, 2yr, 4 splits     (~60 min)
  5) Custom     — You choose everything

Pairs: BTC/USDC, SOL/USDC (or both)

Overrides (apply to any preset):
  --hours N          History length (default from preset)
  --trials N         Trial count (e.g., --trials 25000 for SOL)
  --skip-recent N    Skip most recent N hours (720 = 1 month)
  --fg               Run in foreground (default: background)

Usage:
  ./run.sh 3 --btc              # Background, Production BTC
  ./run.sh 3 --btc --fg         # Foreground (watch live)
  ./run.sh 3 --sol --trials 25000          # SOL, 25k trials
  ./run.sh 3 --hours 17520                 # 2yr history
  ./run.sh 3 --skip-recent 720            # Skip last month
  ./run.sh attach               # Attach to running/latest log
  ./run.sh logs                 # List recent log files

EOF
}

# =============================================================================
# DEFAULTS
# =============================================================================

TRIALS=10000
HOURS=8760
SPLITS=""
PAIRS="both"
TAG=""
PRESET=""
SKIP_RECENT=0
FOREGROUND=false
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# =============================================================================
# PARSE ARGS
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        1) PRESET="quick" ;;
        2) PRESET="standard" ;;
        3) PRESET="production" ;;
        4) PRESET="deep" ;;
        5) PRESET="custom" ;;
        --btc) PAIRS="btc" ;;
        --sol) PAIRS="sol" ;;
        --both) PAIRS="both" ;;
        --trials) TRIALS="$2"; shift ;;
        --hours) HOURS="$2"; shift ;;
        --splits) SPLITS="$2"; shift ;;
        --tag) TAG="$2"; shift ;;
        --skip-recent) SKIP_RECENT="$2"; shift ;;
        --fg) FOREGROUND=true ;;
        attach)
            LATEST=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
            if [ -z "$LATEST" ]; then
                echo "No log files found in $LOG_DIR"
                exit 1
            fi
            echo "Attaching to: $LATEST"
            echo "(Ctrl+C to detach — run continues in background)"
            echo ""
            tail -f "$LATEST"
            exit 0
            ;;
        logs)
            echo "Recent logs:"
            ls -lht "$LOG_DIR"/*.log 2>/dev/null | head -10
            exit 0
            ;;
        -h|--help) show_presets; exit 0 ;;
        *) echo "Unknown option: $1"; show_presets; exit 1 ;;
    esac
    shift
done

# =============================================================================
# APPLY PRESET
# =============================================================================

case "$PRESET" in
    quick)      TRIALS=2000;  HOURS=8760;  SPLITS=2 ;;
    standard)   TRIALS=5000;  HOURS=8760;  SPLITS=3 ;;
    production) TRIALS=10000; HOURS=8760;  SPLITS=3 ;;
    deep)       TRIALS=15000; HOURS=17520; SPLITS=4 ;;
    custom)     ;; # Use --trials, --hours, --splits from args
    "")
        # Interactive mode
        echo ""
        show_presets
        read -p "Select preset (1-5): " choice
        case "$choice" in
            1) TRIALS=2000;  HOURS=8760;  SPLITS=2 ;;
            2) TRIALS=5000;  HOURS=8760;  SPLITS=3 ;;
            3) TRIALS=10000; HOURS=8760;  SPLITS=3 ;;
            4) TRIALS=15000; HOURS=17520; SPLITS=4 ;;
            5)
                read -p "Trials [10000]: " TRIALS; TRIALS="${TRIALS:-10000}"
                read -p "Hours [8760]: " HOURS; HOURS="${HOURS:-8760}"
                read -p "Splits [3]: " SPLITS; SPLITS="${SPLITS:-3}"
                read -p "Skip recent hours [0]: " SKIP_RECENT; SKIP_RECENT="${SKIP_RECENT:-0}"
                ;;
            *) echo "Invalid choice"; exit 1 ;;
        esac

        echo ""
        read -p "Pairs — (1) BTC only, (2) SOL only, (3) Both [3]: " pair_choice
        case "${pair_choice:-3}" in
            1) PAIRS="btc" ;;
            2) PAIRS="sol" ;;
            3) PAIRS="both" ;;
        esac

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
    echo "  kill $BG_PID             # Stop run"
fi
