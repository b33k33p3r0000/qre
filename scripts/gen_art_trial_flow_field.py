#!/usr/bin/env python3
"""Generate Trial Flow Field data art HTML from embedded Optuna trial data."""

import json
import sys
from pathlib import Path


def generate_html(data_path: str, output_path: str) -> None:
    with open(data_path) as f:
        data = json.load(f)

    symbol = data["symbol"]
    run = data["run"]
    trials = data["optuna"]["trials"]
    distributions = data["optuna"]["distributions"]
    total = data["optuna"]["total"]

    # Pre-compute some stats
    values = [t["v"] for t in trials]
    nonzero = [v for v in values if v > 0]
    best_val = max(values) if values else 0
    best_nz = max(nonzero) if nonzero else 0
    min_nz = min(nonzero) if nonzero else 0

    # Serialize data for embedding
    trials_json = json.dumps(trials)
    dists_json = json.dumps(distributions)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Art — Trial Flow Field — {symbol}</title>
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
        select {{
            width: 100%;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-family: 'Inter', sans-serif;
            cursor: pointer;
            outline: none;
        }}
        select:focus {{ border-color: var(--accent-blue); }}
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
            <h1>TRIAL FLOW FIELD</h1>
            <div class="subtitle">OPTUNA SEARCH VECTORS // {symbol}</div>

            <div class="run-info">
                <div>RUN: {run}</div>
                <div>SYMBOL: {symbol}</div>
                <div>TRIALS: {len(trials)} / {total}</div>
                <div>BEST: {best_val:.4f}</div>
            </div>

            <div class="control-section">
                <h3>Axis Mapping</h3>
                <div class="control-group">
                    <label>X Axis</label>
                    <select id="xParam" onchange="scheduleRedraw()"></select>
                </div>
                <div class="control-group">
                    <label>Y Axis</label>
                    <select id="yParam" onchange="scheduleRedraw()"></select>
                </div>
            </div>

            <div class="control-section">
                <h3>Flow</h3>
                <div class="control-group">
                    <label>Flow Strength</label>
                    <div class="slider-container">
                        <input type="range" id="flowStrength" min="0" max="60" step="1" value="30" oninput="updateSlider('flowStrength', this.value); scheduleRedraw()">
                        <span class="value-display" id="flowStrength-value">30</span>
                    </div>
                </div>
            </div>

            <div class="control-section">
                <h3>Dots</h3>
                <div class="control-group">
                    <label>Dot Size</label>
                    <div class="slider-container">
                        <input type="range" id="dotSize" min="0" max="5" step="0.1" value="1.5" oninput="updateSlider('dotSize', this.value); scheduleRedraw()">
                        <span class="value-display" id="dotSize-value">1.5</span>
                    </div>
                </div>
                <div class="control-group">
                    <label>Dot Opacity</label>
                    <div class="slider-container">
                        <input type="range" id="dotOpacity" min="0" max="100" step="1" value="40" oninput="updateSlider('dotOpacity', this.value); scheduleRedraw()">
                        <span class="value-display" id="dotOpacity-value">40</span>
                    </div>
                </div>
            </div>

            <div class="control-section">
                <h3>Trails</h3>
                <div class="control-group">
                    <label>Trail Opacity</label>
                    <div class="slider-container">
                        <input type="range" id="trailOpacity" min="0" max="20" step="0.5" value="6" oninput="updateSlider('trailOpacity', this.value); scheduleRedraw()">
                        <span class="value-display" id="trailOpacity-value">6</span>
                    </div>
                </div>
                <div class="control-group">
                    <label>Trail Weight</label>
                    <div class="slider-container">
                        <input type="range" id="trailWeight" min="0" max="2" step="0.05" value="0.4" oninput="updateSlider('trailWeight', this.value); scheduleRedraw()">
                        <span class="value-display" id="trailWeight-value">0.4</span>
                    </div>
                </div>
            </div>

            <div class="control-section">
                <h3>Colors</h3>
                <div class="color-group">
                    <label>High Score</label>
                    <input type="color" id="colorHigh" value="#e6edf3" onchange="scheduleRedraw()">
                    <span class="color-value" id="colorHigh-hex">#e6edf3</span>
                </div>
                <div class="color-group">
                    <label>Low Score</label>
                    <input type="color" id="colorLow" value="#484f58" onchange="scheduleRedraw()">
                    <span class="color-value" id="colorLow-hex">#484f58</span>
                </div>
            </div>

            <div class="control-section">
                <h3>Layout</h3>
                <div class="button-row" style="margin-bottom: 16px;">
                    <button class="button active" id="btn-rect" onclick="setLayout('rect')">Rectangular</button>
                    <button class="button" id="btn-circ" onclick="setLayout('circ')">Circular</button>
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
    const DATA_TRIALS = {trials_json};
    const DATA_DISTS = {dists_json};
    const SYMBOL = "{symbol}";
    const TOTAL_TRIALS = {total};

    // Available params from distributions
    const PARAM_NAMES = Object.keys(DATA_DISTS).sort();

    // State
    let layout = 'rect';
    let needsRedraw = true;
    let redrawTimer = null;

    // Defaults
    const DEFAULTS = {{
        flowStrength: 30, dotSize: 1.5, dotOpacity: 40,
        trailOpacity: 6, trailWeight: 0.4,
        colorHigh: '#e6edf3', colorLow: '#484f58',
        xParam: 'macd_fast', yParam: 'rsi_period'
    }};

    function populateSelects() {{
        const xSel = document.getElementById('xParam');
        const ySel = document.getElementById('yParam');
        PARAM_NAMES.forEach(p => {{
            let ox = document.createElement('option');
            ox.value = p; ox.textContent = p;
            if (p === DEFAULTS.xParam) ox.selected = true;
            xSel.appendChild(ox);
            let oy = document.createElement('option');
            oy.value = p; oy.textContent = p;
            if (p === DEFAULTS.yParam) oy.selected = true;
            ySel.appendChild(oy);
        }});
    }}
    populateSelects();

    function updateSlider(id, val) {{
        document.getElementById(id + '-value').textContent = val;
    }}

    function getVal(id) {{ return parseFloat(document.getElementById(id).value); }}
    function getColor(id) {{ return document.getElementById(id).value; }}

    function setLayout(l) {{
        layout = l;
        document.getElementById('btn-rect').classList.toggle('active', l === 'rect');
        document.getElementById('btn-circ').classList.toggle('active', l === 'circ');
        scheduleRedraw();
    }}

    function scheduleRedraw() {{
        needsRedraw = true;
        // Update color hex displays
        ['colorHigh', 'colorLow'].forEach(id => {{
            let hex = document.getElementById(id).value;
            let el = document.getElementById(id + '-hex');
            if (el) el.textContent = hex;
        }});
    }}

    function resetDefaults() {{
        document.getElementById('flowStrength').value = DEFAULTS.flowStrength;
        document.getElementById('dotSize').value = DEFAULTS.dotSize;
        document.getElementById('dotOpacity').value = DEFAULTS.dotOpacity;
        document.getElementById('trailOpacity').value = DEFAULTS.trailOpacity;
        document.getElementById('trailWeight').value = DEFAULTS.trailWeight;
        document.getElementById('colorHigh').value = DEFAULTS.colorHigh;
        document.getElementById('colorLow').value = DEFAULTS.colorLow;
        document.getElementById('xParam').value = DEFAULTS.xParam;
        document.getElementById('yParam').value = DEFAULTS.yParam;
        layout = 'rect';
        document.getElementById('btn-rect').classList.add('active');
        document.getElementById('btn-circ').classList.remove('active');
        ['flowStrength','dotSize','dotOpacity','trailOpacity','trailWeight'].forEach(id => {{
            updateSlider(id, document.getElementById(id).value);
        }});
        ['colorHigh','colorLow'].forEach(id => {{
            document.getElementById(id + '-hex').textContent = document.getElementById(id).value;
        }});
        scheduleRedraw();
    }}

    function exportPNG() {{
        saveCanvas('trial_flow_field_{symbol}', 'png');
    }}

    // Normalization helper
    function norm(val, low, high) {{
        if (high === low) return 0.5;
        return (val - low) / (high - low);
    }}

    // Color interpolation
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

    // Pre-process trials
    let processedTrials = [];
    let maxVal = 0;
    let minPosVal = Infinity;

    DATA_TRIALS.forEach(t => {{
        if (t.v > maxVal) maxVal = t.v;
        if (t.v > 0 && t.v < minPosVal) minPosVal = t.v;
    }});
    if (minPosVal === Infinity) minPosVal = 0;

    // Canvas dimensions
    const CW = 1200;
    const CH = 1200;
    const MARGIN = 60;

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

        background(13, 17, 23); // #0d1117

        const flowStr = getVal('flowStrength');
        const dotSz = getVal('dotSize');
        const dotOp = getVal('dotOpacity');
        const trailOp = getVal('trailOpacity');
        const trailW = getVal('trailWeight');
        const cHigh = hexToRgb(getColor('colorHigh'));
        const cLow = hexToRgb(getColor('colorLow'));
        const xP = document.getElementById('xParam').value;
        const yP = document.getElementById('yParam').value;

        const xDist = DATA_DISTS[xP];
        const yDist = DATA_DISTS[yP];
        const xLow = xDist.attributes.low;
        const xHigh = xDist.attributes.high;
        const yLow = yDist.attributes.low;
        const yHigh = yDist.attributes.high;

        // Get all remaining params (not x or y) for flow vector
        const flowParams = PARAM_NAMES.filter(p => p !== xP && p !== yP);

        // Sort trials by trial number
        let sorted = [...DATA_TRIALS].sort((a, b) => a.n - b.n);

        // Compute positions
        let positions = [];
        sorted.forEach((t, i) => {{
            const xRaw = t.p[xP] !== undefined ? t.p[xP] : 0;
            const yRaw = t.p[yP] !== undefined ? t.p[yP] : 0;
            let xNorm = norm(xRaw, xLow, xHigh);
            let yNorm = norm(yRaw, yLow, yHigh);

            // Compute flow angle from remaining params
            let flowSum = 0;
            let flowCount = 0;
            flowParams.forEach(fp => {{
                const fd = DATA_DISTS[fp];
                if (t.p[fp] !== undefined) {{
                    flowSum += norm(t.p[fp], fd.attributes.low, fd.attributes.high);
                    flowCount++;
                }}
            }});
            const flowAvg = flowCount > 0 ? flowSum / flowCount : 0.5;
            const flowAngle = flowAvg * TWO_PI;
            const flowMag = flowStr;

            // Score normalized (0-1)
            let scoreNorm = 0;
            if (maxVal > 0 && t.v > 0) {{
                scoreNorm = t.v / maxVal;
            }}

            if (layout === 'rect') {{
                // Rectangular layout
                let px = MARGIN + xNorm * (CW - 2 * MARGIN);
                let py = MARGIN + (1 - yNorm) * (CH - 2 * MARGIN);
                px += cos(flowAngle) * flowMag;
                py += sin(flowAngle) * flowMag;
                positions.push({{ x: px, y: py, score: scoreNorm, val: t.v }});
            }} else {{
                // Circular layout
                const cx = CW / 2;
                const cy = CH / 2;
                const maxR = min(CW, CH) / 2 - MARGIN;
                // Map xNorm to angle, yNorm to radius
                const angle = xNorm * TWO_PI - HALF_PI;
                const radius = yNorm * maxR * 0.9 + maxR * 0.05;
                let px = cx + cos(angle) * radius;
                let py = cy + sin(angle) * radius;
                px += cos(flowAngle) * flowMag;
                py += sin(flowAngle) * flowMag;
                positions.push({{ x: px, y: py, score: scoreNorm, val: t.v }});
            }}
        }});

        // Draw trails (connecting lines between consecutive trials)
        if (trailOp > 0 && trailW > 0) {{
            for (let i = 1; i < positions.length; i++) {{
                const p0 = positions[i - 1];
                const p1 = positions[i];
                const avgScore = (p0.score + p1.score) / 2;
                const c = lerpColor3(cLow, cHigh, avgScore);
                stroke(c[0], c[1], c[2], trailOp * 2.55);
                strokeWeight(trailW);
                line(p0.x, p0.y, p1.x, p1.y);
            }}
        }}

        // Draw dots
        noStroke();
        positions.forEach(p => {{
            if (dotSz <= 0) return;
            const c = lerpColor3(cLow, cHigh, p.score);
            const sz = dotSz * (0.3 + p.score * 0.7);
            const alpha = dotOp * 2.55 * (0.3 + p.score * 0.7);
            fill(c[0], c[1], c[2], alpha);
            ellipse(p.x, p.y, sz * 2, sz * 2);
        }});

        // Axis labels
        if (layout === 'rect') {{
            fill(72, 79, 88); // text-muted
            noStroke();
            textFont('JetBrains Mono');
            textSize(11);
            textAlign(CENTER, TOP);
            text(xP, CW / 2, CH - 20);
            push();
            translate(18, CH / 2);
            rotate(-HALF_PI);
            textAlign(CENTER, BOTTOM);
            text(yP, 0, 0);
            pop();

            // Axis tick marks
            stroke(48, 63, 61, 40);
            strokeWeight(0.5);
            for (let i = 0; i <= 4; i++) {{
                let tx = MARGIN + (i / 4) * (CW - 2 * MARGIN);
                line(tx, CH - MARGIN + 5, tx, CH - MARGIN + 12);
                let ty = MARGIN + (i / 4) * (CH - 2 * MARGIN);
                line(MARGIN - 12, ty, MARGIN - 5, ty);
            }}

            // Tick labels
            noStroke();
            fill(72, 79, 88);
            textSize(9);
            textAlign(CENTER, TOP);
            for (let i = 0; i <= 4; i++) {{
                let tx = MARGIN + (i / 4) * (CW - 2 * MARGIN);
                let val = xLow + (i / 4) * (xHigh - xLow);
                text(val.toFixed(1), tx, CH - MARGIN + 14);
            }}
            textAlign(RIGHT, CENTER);
            for (let i = 0; i <= 4; i++) {{
                let ty = MARGIN + (i / 4) * (CH - 2 * MARGIN);
                let val = yHigh - (i / 4) * (yHigh - yLow);
                text(val.toFixed(1), MARGIN - 14, ty);
            }}
        }}

        // Watermark
        fill(48, 63, 61, 60);
        noStroke();
        textFont('JetBrains Mono');
        textSize(10);
        textAlign(RIGHT, BOTTOM);
        text('QRE // TRIAL FLOW FIELD // ' + SYMBOL, CW - 20, CH - 12);
    }}

    // Use requestAnimationFrame loop for responsiveness
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
    data_path = "/tmp/art_data_embed.json"
    output_path = (
        "/Users/davidxbinko/projects/qre/results/"
        "2026-02-22_17-58-58_flip-on-sol/SOL/art_trial_flow_field.html"
    )
    generate_html(data_path, output_path)
