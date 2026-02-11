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

Usage:
  ./run.sh              # Interactive menu
  ./run.sh 2            # Standard preset, both pairs
  ./run.sh 3 --btc      # Production, BTC only
  ./run.sh 3 --sol --trials 25000          # Production SOL, 25k trials
  ./run.sh 3 --hours 17520                 # Production, 2yr history
  ./run.sh 3 --skip-recent 720            # Production, skip last month
  ./run.sh 3 --hours 13140 --skip-recent 720  # 15mo history, skip last month

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
echo "═══════════════════════════════════════════"
echo ""

run_symbol() {
    local symbol="$1"
    local cmd=$(build_cmd "$symbol")
    echo ">>> Running: $cmd"
    echo ""
    eval "$cmd"
    echo ""
    echo ">>> Done: $symbol"
    echo ""
}

case "$PAIRS" in
    btc)  run_symbol "BTC/USDC" ;;
    sol)  run_symbol "SOL/USDC" ;;
    both)
        run_symbol "BTC/USDC"
        run_symbol "SOL/USDC"
        ;;
esac

echo "═══════════════════════════════════════════"
echo "  All runs complete!"
echo "═══════════════════════════════════════════"
