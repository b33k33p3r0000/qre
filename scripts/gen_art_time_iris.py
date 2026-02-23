#!/usr/bin/env python3
"""Generate Time Iris data art HTML from embedded trade data."""

import json
from pathlib import Path


def generate_html(data_path: str, output_path: str) -> None:
    with open(data_path) as f:
        data = json.load(f)

    symbol = data["symbol"]
    run = data["run"]
    trades = data["trades"]
    metrics = data.get("metrics", {})

    # Pre-compute stats
    total_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    losses = sum(1 for t in trades if t["pnl_pct"] < 0)
    total_pnl = sum(t["pnl_abs"] for t in trades)
    max_pnl = max(t["pnl_pct"] for t in trades) if trades else 0
    min_pnl = min(t["pnl_pct"] for t in trades) if trades else 0
    win_rate = metrics.get("win_rate", wins / total_trades if total_trades > 0 else 0)
    log_calmar = metrics.get("log_calmar", 0)
    max_dd = metrics.get("max_drawdown", 0)

    trades_json = json.dumps(trades)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Art — Time Iris — {symbol}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.7.0/p5.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #1c2129;
            --border: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #484f58;
            --accent-blue: #58a6ff;
            --accent-red: #f85149;
            --accent-green: #3fb950;
            --accent-cyan: #76e3ea;
            --accent-teal: #4ec9b0;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            min-height: 100vh;
            color: var(--text-primary);
        }}
        .container {{
            display: flex;
            min-height: 100vh;
            padding: 16px;
            gap: 16px;
        }}
        .sidebar {{
            width: 300px;
            flex-shrink: 0;
            background: var(--bg-secondary);
            padding: 20px;
            border-radius: 8px;
            border: 1px solid var(--border);
            overflow-y: auto;
            overflow-x: hidden;
        }}
        .sidebar h1 {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 18px;
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 4px;
        }}
        .sidebar .subtitle {{
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-muted);
            font-size: 11px;
            margin-bottom: 24px;
            letter-spacing: 0.5px;
        }}
        .sidebar .run-info {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-secondary);
            margin-bottom: 24px;
            padding: 10px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            border: 1px solid var(--border);
            line-height: 1.6;
        }}
        .control-section {{ margin-bottom: 24px; }}
        .control-section h3 {{
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .control-group {{ margin-bottom: 16px; }}
        .control-group label {{
            display: block;
            font-size: 12px;
            font-weight: 400;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }}
        .slider-container {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .slider-container input[type="range"] {{
            flex: 1;
            height: 3px;
            background: var(--border);
            border-radius: 2px;
            outline: none;
            -webkit-appearance: none;
        }}
        .slider-container input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            background: var(--accent-blue);
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .slider-container input[type="range"]::-webkit-slider-thumb:hover {{
            transform: scale(1.15);
            box-shadow: 0 0 8px rgba(88, 166, 255, 0.4);
        }}
        .value-display {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-muted);
            min-width: 50px;
            text-align: right;
        }}
        .button {{
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border);
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
            width: 100%;
            font-family: 'Inter', sans-serif;
        }}
        .button:hover {{
            background: var(--border);
            border-color: var(--text-muted);
        }}
        .button.primary {{
            background: rgba(88, 166, 255, 0.15);
            border-color: rgba(88, 166, 255, 0.3);
            color: var(--accent-blue);
        }}
        .button.primary:hover {{ background: rgba(88, 166, 255, 0.25); }}
        .button.active {{
            background: rgba(88, 166, 255, 0.2);
            border-color: var(--accent-blue);
            color: var(--accent-blue);
        }}
        .button-row {{ display: flex; gap: 8px; }}
        .button-row .button {{ flex: 1; }}
        .canvas-area {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 0;
        }}
        #canvas-container {{
            width: 100%;
            max-width: 1200px;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
            background: var(--bg-primary);
        }}
        #canvas-container canvas {{
            display: block;
            width: 100% !important;
            height: auto !important;
        }}
        .color-group {{
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .color-group label {{
            font-size: 11px;
            color: var(--text-secondary);
            min-width: 80px;
        }}
        .color-group input[type="color"] {{
            width: 28px;
            height: 28px;
            border: 1px solid var(--border);
            border-radius: 4px;
            cursor: pointer;
            background: none;
            padding: 2px;
        }}
        .color-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: var(--text-muted);
        }}
        @media (max-width: 768px) {{
            .container {{ flex-direction: column; }}
            .sidebar {{ width: 100%; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h1>TIME IRIS</h1>
            <div class="subtitle">TRADE CHRONOGRAPH // {symbol}</div>

            <div class="run-info">
                <div>RUN: {run}</div>
                <div>SYMBOL: {symbol}</div>
                <div>TRADES: {total_trades}</div>
                <div>WIN RATE: {win_rate:.0%}</div>
                <div>LOG CALMAR: {log_calmar:.4f}</div>
                <div>MAX DD: {max_dd:.2f}%</div>
                <div>P&amp;L: ${total_pnl:,.0f}</div>
            </div>

            <div class="control-section">
                <h3>Geometry</h3>
                <div class="control-group">
                    <label>Inner Radius</label>
                    <div class="slider-container">
                        <input type="range" id="innerRadius" min="0" max="200" step="5" value="80" oninput="updateSlider('innerRadius', this.value); scheduleRedraw()">
                        <span class="value-display" id="innerRadius-value">80</span>
                    </div>
                </div>
                <div class="control-group">
                    <label>Outer Radius</label>
                    <div class="slider-container">
                        <input type="range" id="outerRadius" min="0" max="550" step="5" value="450" oninput="updateSlider('outerRadius', this.value); scheduleRedraw()">
                        <span class="value-display" id="outerRadius-value">450</span>
                    </div>
                </div>
                <div class="control-group">
                    <label>Gap (degrees)</label>
                    <div class="slider-container">
                        <input type="range" id="gapDeg" min="0" max="3" step="0.1" value="0.5" oninput="updateSlider('gapDeg', this.value); scheduleRedraw()">
                        <span class="value-display" id="gapDeg-value">0.5</span>
                    </div>
                </div>
            </div>

            <div class="control-section">
                <h3>Rendering</h3>
                <div class="control-group">
                    <label>Strip Opacity</label>
                    <div class="slider-container">
                        <input type="range" id="stripOpacity" min="0" max="255" step="1" value="220" oninput="updateSlider('stripOpacity', this.value); scheduleRedraw()">
                        <span class="value-display" id="stripOpacity-value">220</span>
                    </div>
                </div>
            </div>

            <div class="control-section">
                <h3>Colors</h3>
                <div class="color-group">
                    <label>Profit</label>
                    <input type="color" id="colorProfit" value="#4ec9b0" onchange="scheduleRedraw()">
                    <span class="color-value" id="colorProfit-hex">#4ec9b0</span>
                </div>
                <div class="color-group">
                    <label>Loss</label>
                    <input type="color" id="colorLoss" value="#f85149" onchange="scheduleRedraw()">
                    <span class="color-value" id="colorLoss-hex">#f85149</span>
                </div>
                <div class="color-group">
                    <label>Neutral</label>
                    <input type="color" id="colorNeutral" value="#484f58" onchange="scheduleRedraw()">
                    <span class="color-value" id="colorNeutral-hex">#484f58</span>
                </div>
            </div>

            <div class="control-section">
                <h3>Layout</h3>
                <div class="button-row" style="margin-bottom: 16px;">
                    <button class="button active" id="btn-circ" onclick="setLayout('circ')">Circular</button>
                    <button class="button" id="btn-rect" onclick="setLayout('rect')">Rectangular</button>
                </div>
            </div>

            <div class="control-section">
                <h3>Actions</h3>
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    <button class="button primary" onclick="exportPNG()">Export PNG</button>
                    <button class="button" onclick="resetDefaults()">Reset</button>
                </div>
            </div>
        </div>

        <div class="canvas-area">
            <div id="canvas-container"></div>
        </div>
    </div>

    <script>
    // Embedded data
    const TRADES = {trades_json};
    const SYMBOL = "{symbol}";
    const METRICS = {{
        trades: {total_trades},
        winRate: {win_rate:.4f},
        logCalmar: {log_calmar:.4f},
        maxDD: {max_dd:.2f},
        totalPnl: {total_pnl:.2f}
    }};

    // State
    let layout = 'circ';
    let needsRedraw = true;

    // Defaults
    const DEFAULTS = {{
        innerRadius: 80, outerRadius: 450, gapDeg: 0.5, stripOpacity: 220,
        colorProfit: '#4ec9b0', colorLoss: '#f85149', colorNeutral: '#484f58'
    }};

    function updateSlider(id, val) {{
        document.getElementById(id + '-value').textContent = val;
    }}

    function getVal(id) {{ return parseFloat(document.getElementById(id).value); }}
    function getColor(id) {{ return document.getElementById(id).value; }}

    function setLayout(l) {{
        layout = l;
        document.getElementById('btn-circ').classList.toggle('active', l === 'circ');
        document.getElementById('btn-rect').classList.toggle('active', l === 'rect');
        scheduleRedraw();
    }}

    function scheduleRedraw() {{
        needsRedraw = true;
        ['colorProfit', 'colorLoss', 'colorNeutral'].forEach(id => {{
            let hex = document.getElementById(id).value;
            let el = document.getElementById(id + '-hex');
            if (el) el.textContent = hex;
        }});
    }}

    function resetDefaults() {{
        document.getElementById('innerRadius').value = DEFAULTS.innerRadius;
        document.getElementById('outerRadius').value = DEFAULTS.outerRadius;
        document.getElementById('gapDeg').value = DEFAULTS.gapDeg;
        document.getElementById('stripOpacity').value = DEFAULTS.stripOpacity;
        document.getElementById('colorProfit').value = DEFAULTS.colorProfit;
        document.getElementById('colorLoss').value = DEFAULTS.colorLoss;
        document.getElementById('colorNeutral').value = DEFAULTS.colorNeutral;
        layout = 'circ';
        document.getElementById('btn-circ').classList.add('active');
        document.getElementById('btn-rect').classList.remove('active');
        ['innerRadius','outerRadius','gapDeg','stripOpacity'].forEach(id => {{
            updateSlider(id, document.getElementById(id).value);
        }});
        ['colorProfit','colorLoss','colorNeutral'].forEach(id => {{
            document.getElementById(id + '-hex').textContent = document.getElementById(id).value;
        }});
        scheduleRedraw();
    }}

    function exportPNG() {{
        saveCanvas('time_iris_{symbol}', 'png');
    }}

    // Color helpers
    function hexToRgb(hex) {{
        let r = parseInt(hex.slice(1,3), 16);
        let g = parseInt(hex.slice(3,5), 16);
        let b = parseInt(hex.slice(5,7), 16);
        return [r, g, b];
    }}
    function lerpColor3(c1, c2, t) {{
        return [
            c1[0] + (c2[0]-c1[0]) * t,
            c1[1] + (c2[1]-c1[1]) * t,
            c1[2] + (c2[2]-c1[2]) * t
        ];
    }}

    // Pre-compute P&L range
    let maxAbsPnl = 0;
    TRADES.forEach(t => {{
        let a = Math.abs(t.pnl_pct);
        if (a > maxAbsPnl) maxAbsPnl = a;
    }});

    // Total hold bars for angle proportions
    let totalHoldBars = 0;
    TRADES.forEach(t => {{ totalHoldBars += t.hold_bars; }});

    const CW = 1200;
    const CH = 1200;

    function setup() {{
        pixelDensity(2);
        let canvas = createCanvas(CW, CH);
        canvas.parent('canvas-container');
        noLoop();
        scheduleRedraw();
    }}

    function draw() {{
        if (!needsRedraw) return;
        needsRedraw = false;

        background(13, 17, 23);

        const innerR = getVal('innerRadius');
        const outerR = getVal('outerRadius');
        const gapDeg = getVal('gapDeg');
        const stripOp = getVal('stripOpacity');
        const cProfit = hexToRgb(getColor('colorProfit'));
        const cLoss = hexToRgb(getColor('colorLoss'));
        const cNeutral = hexToRgb(getColor('colorNeutral'));

        const gapRad = radians(gapDeg);
        const totalGap = gapRad * TRADES.length;
        const availableAngle = TWO_PI - totalGap;

        if (layout === 'circ') {{
            drawCircular(innerR, outerR, gapRad, availableAngle, stripOp, cProfit, cLoss, cNeutral);
        }} else {{
            drawRectangular(stripOp, cProfit, cLoss, cNeutral, gapDeg);
        }}

        // Watermark
        fill(48, 63, 61, 60);
        noStroke();
        textFont('JetBrains Mono');
        textSize(10);
        textAlign(RIGHT, BOTTOM);
        text('QRE // TIME IRIS // ' + SYMBOL, CW - 20, CH - 12);
    }}

    function getTradeColor(pnlPct, cProfit, cLoss, cNeutral) {{
        if (maxAbsPnl === 0) return cNeutral;
        // Normalize to -1..1
        let norm = pnlPct / maxAbsPnl;
        // Clamp
        norm = Math.max(-1, Math.min(1, norm));

        if (norm > 0.005) {{
            // Profit: neutral -> profit
            let t = Math.pow(norm, 0.6); // gamma for better gradient
            return lerpColor3(cNeutral, cProfit, t);
        }} else if (norm < -0.005) {{
            let t = Math.pow(Math.abs(norm), 0.6);
            return lerpColor3(cNeutral, cLoss, t);
        }} else {{
            return cNeutral;
        }}
    }}

    function drawCircular(innerR, outerR, gapRad, availableAngle, stripOp, cProfit, cLoss, cNeutral) {{
        const cx = CW / 2;
        const cy = CH / 2;

        let currentAngle = -HALF_PI; // Start from top

        noStroke();

        TRADES.forEach((trade, i) => {{
            const proportion = totalHoldBars > 0 ? trade.hold_bars / totalHoldBars : 1 / TRADES.length;
            const arcWidth = proportion * availableAngle;

            if (arcWidth <= 0 || outerR <= 0) return;

            const startA = currentAngle;
            const endA = currentAngle + arcWidth;
            const c = getTradeColor(trade.pnl_pct, cProfit, cLoss, cNeutral);

            // Draw filled arc (strip)
            fill(c[0], c[1], c[2], stripOp);

            beginShape();
            // Outer arc
            const steps = Math.max(8, Math.ceil(arcWidth / 0.02));
            for (let s = 0; s <= steps; s++) {{
                let a = startA + (s / steps) * arcWidth;
                vertex(cx + cos(a) * outerR, cy + sin(a) * outerR);
            }}
            // Inner arc (reverse)
            for (let s = steps; s >= 0; s--) {{
                let a = startA + (s / steps) * arcWidth;
                vertex(cx + cos(a) * innerR, cy + sin(a) * innerR);
            }}
            endShape(CLOSE);

            // Bright edge on outer rim for profitable trades
            if (trade.pnl_pct > 0) {{
                stroke(c[0], c[1], c[2], Math.min(255, stripOp + 35));
                strokeWeight(1.5);
                noFill();
                beginShape();
                for (let s = 0; s <= steps; s++) {{
                    let a = startA + (s / steps) * arcWidth;
                    vertex(cx + cos(a) * outerR, cy + sin(a) * outerR);
                }}
                endShape();
                noStroke();
            }}

            currentAngle = endA + gapRad;
        }});

        // Inner circle metrics text
        if (innerR > 30) {{
            fill(230, 237, 243, 200);
            noStroke();
            textFont('JetBrains Mono');
            textAlign(CENTER, CENTER);

            textSize(28);
            text(SYMBOL, cx, cy - 30);

            textSize(12);
            fill(139, 148, 158, 200);
            text(METRICS.trades + ' trades', cx, cy + 5);
            text((METRICS.winRate * 100).toFixed(0) + '% win', cx, cy + 22);
            text('DD ' + METRICS.maxDD.toFixed(1) + '%', cx, cy + 39);
        }}
    }}

    function drawRectangular(stripOp, cProfit, cLoss, cNeutral, gapDeg) {{
        // Vertical barcode layout
        const margin = 40;
        const availWidth = CW - 2 * margin;
        const totalGapPx = gapDeg * TRADES.length; // gapDeg as pixel gap in rect mode
        const drawableWidth = availWidth - totalGapPx;

        let currentX = margin;
        const topY = margin;
        const bottomY = CH - margin;
        const barHeight = bottomY - topY;

        noStroke();

        TRADES.forEach((trade, i) => {{
            const proportion = totalHoldBars > 0 ? trade.hold_bars / totalHoldBars : 1 / TRADES.length;
            const barWidth = proportion * drawableWidth;

            if (barWidth <= 0) return;

            const c = getTradeColor(trade.pnl_pct, cProfit, cLoss, cNeutral);

            fill(c[0], c[1], c[2], stripOp);
            rect(currentX, topY, barWidth, barHeight);

            // Bright top edge for profitable trades
            if (trade.pnl_pct > 0) {{
                stroke(c[0], c[1], c[2], Math.min(255, stripOp + 35));
                strokeWeight(2);
                line(currentX, topY, currentX + barWidth, topY);
                noStroke();
            }}

            currentX += barWidth + gapDeg;
        }});

        // Labels
        fill(139, 148, 158, 160);
        noStroke();
        textFont('JetBrains Mono');
        textSize(11);
        textAlign(LEFT, TOP);
        text('FIRST TRADE', margin, bottomY + 8);
        textAlign(RIGHT, TOP);
        text('LAST TRADE', CW - margin, bottomY + 8);
        textAlign(CENTER, BOTTOM);
        textSize(20);
        fill(230, 237, 243, 200);
        text(SYMBOL + ' — ' + METRICS.trades + ' trades — ' + (METRICS.winRate * 100).toFixed(0) + '% win', CW / 2, topY - 10);
    }}

    // Animation loop
    function animLoop() {{
        if (needsRedraw) {{
            redraw();
        }}
        requestAnimationFrame(animLoop);
    }}
    requestAnimationFrame(animLoop);
    </script>
</body>
</html>"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(f"Written: {output_path} ({len(html):,} bytes)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Time Iris data art")
    parser.add_argument("data_path", type=str, help="Path to art data JSON")
    parser.add_argument("output_path", type=str, help="Output HTML path")
    args = parser.parse_args()
    generate_html(args.data_path, args.output_path)
