#!/usr/bin/env python3
"""Generate Run Portrait — 5-layer radial data art visualization of a QRE optimizer run."""

import json
import sys
from pathlib import Path

DATA_PATH = Path("/tmp/art_data_embed.json")
OUTPUT_DIR = Path.home() / "projects/qre/results/2026-02-22_17-58-58_flip-on-sol/SOL"
OUTPUT_FILE = OUTPUT_DIR / "art_run_portrait.html"


def main():
    # Load data
    with open(DATA_PATH) as f:
        data = json.load(f)

    symbol = data["symbol"]
    run_name = data["run"]
    trials = data["optuna"]["trials"]
    distributions = data["optuna"]["distributions"]
    trades = data["trades"]
    metrics = data["metrics"]

    # Serialize data compactly for embedding
    data_json = json.dumps(data, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Run Portrait — {symbol} | {run_name}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.7.0/p5.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; display: flex; overflow: hidden; height: 100vh; }}
  #canvas-wrap {{ flex: 1; display: flex; align-items: center; justify-content: center; }}
  #sidebar {{
    width: 300px; min-width: 300px; background: #161b22; border-left: 1px solid #21262d;
    padding: 16px; overflow-y: auto; display: flex; flex-direction: column; gap: 6px;
    font-size: 12px;
  }}
  #sidebar h2 {{ font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #58a6ff; margin-bottom: 4px; letter-spacing: 0.5px; }}
  #sidebar h3 {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #8b949e; margin-top: 10px; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 1px; }}
  .ctrl {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 3px 0; }}
  .ctrl label {{ flex: 0 0 auto; color: #8b949e; font-size: 11px; white-space: nowrap; }}
  .ctrl input[type=range] {{ flex: 1; accent-color: #58a6ff; height: 4px; cursor: pointer; }}
  .ctrl input[type=color] {{ width: 28px; height: 20px; border: 1px solid #30363d; border-radius: 3px; background: transparent; cursor: pointer; padding: 0; }}
  .ctrl .val {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #58a6ff; min-width: 32px; text-align: right; }}
  .toggle {{ display: flex; align-items: center; gap: 8px; padding: 2px 0; cursor: pointer; }}
  .toggle input {{ accent-color: #58a6ff; cursor: pointer; }}
  .toggle label {{ color: #c9d1d9; font-size: 11px; cursor: pointer; }}
  .btn {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 6px 12px;
    border: 1px solid #30363d; border-radius: 4px; cursor: pointer; background: #21262d;
    color: #c9d1d9; transition: all 0.15s;
  }}
  .btn:hover {{ background: #30363d; border-color: #58a6ff; }}
  .btn-row {{ display: flex; gap: 8px; margin-top: 8px; }}
  .meta {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #484f58; margin-top: auto; padding-top: 12px; border-top: 1px solid #21262d; line-height: 1.6; }}
</style>
</head>
<body>
<div id="canvas-wrap"></div>
<div id="sidebar">
  <h2>RUN PORTRAIT</h2>
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#8b949e;margin-bottom:8px;">
    {symbol} &middot; {run_name.split('_')[-1]}
  </div>

  <h3>Layers</h3>
  <div class="toggle"><input type="checkbox" id="L1" checked><label for="L1">1 &mdash; Objective Glow</label></div>
  <div class="toggle"><input type="checkbox" id="L2" checked><label for="L2">2 &mdash; Param Star</label></div>
  <div class="toggle"><input type="checkbox" id="L3" checked><label for="L3">3 &mdash; Trade Timeline</label></div>
  <div class="toggle"><input type="checkbox" id="L4" checked><label for="L4">4 &mdash; Trial Density</label></div>
  <div class="toggle"><input type="checkbox" id="L5" checked><label for="L5">5 &mdash; Score Dust</label></div>

  <h3>Layer 2 &mdash; Param Star</h3>
  <div class="ctrl"><label>Scale</label><input type="range" id="paramScale" min="0" max="200" value="100"><span class="val" id="paramScaleV">1.00</span></div>

  <h3>Layer 3 &mdash; Trades</h3>
  <div class="ctrl"><label>Amplitude</label><input type="range" id="tradeAmp" min="0" max="200" value="80"><span class="val" id="tradeAmpV">80</span></div>
  <div class="ctrl"><label>Equity Line</label><input type="range" id="eqLine" min="0" max="300" value="150"><span class="val" id="eqLineV">1.50</span></div>

  <h3>Layer 4 &mdash; Density</h3>
  <div class="ctrl"><label>Scale</label><input type="range" id="densScale" min="0" max="100" value="50"><span class="val" id="densScaleV">50</span></div>

  <h3>Layer 5 &mdash; Dust</h3>
  <div class="ctrl"><label>Opacity</label><input type="range" id="dustOp" min="0" max="80" value="25"><span class="val" id="dustOpV">25</span></div>
  <div class="ctrl"><label>Size</label><input type="range" id="dustSz" min="0" max="300" value="100"><span class="val" id="dustSzV">1.00</span></div>

  <h3>Colors</h3>
  <div class="ctrl"><label>Core</label><input type="color" id="cCore" value="#f0c040"></div>
  <div class="ctrl"><label>Profit</label><input type="color" id="cProfit" value="#3fb950"></div>
  <div class="ctrl"><label>Loss</label><input type="color" id="cLoss" value="#f85149"></div>
  <div class="ctrl"><label>High Score</label><input type="color" id="cHigh" value="#58a6ff"></div>
  <div class="ctrl"><label>Low Score</label><input type="color" id="cLow" value="#484f58"></div>

  <div class="btn-row">
    <button class="btn" id="btnExport">Export PNG</button>
    <button class="btn" id="btnReset">Reset</button>
  </div>

  <div class="meta">
    Log Calmar: {metrics['log_calmar']}<br>
    Sharpe: {metrics['sharpe_equity']}<br>
    Max DD: {metrics['max_drawdown']}%<br>
    Win Rate: {int(metrics['win_rate']*100)}%<br>
    Trades: {metrics['trades']}<br>
    Trades/yr: {metrics['trades_per_year']}
  </div>
</div>

<script>
// ── Embedded data ──
const DATA = {data_json};

// ── Parse helpers ──
function hexToRgb(h) {{
  const r = parseInt(h.slice(1,3),16), g = parseInt(h.slice(3,5),16), b = parseInt(h.slice(5,7),16);
  return [r,g,b];
}}

// ── Controls state ──
const C = {{
  L1: true, L2: true, L3: true, L4: true, L5: true,
  paramScale: 1.0, tradeAmp: 80, eqLine: 1.5, densScale: 50, dustOp: 25, dustSz: 1.0,
  cCore: '#f0c040', cProfit: '#3fb950', cLoss: '#f85149', cHigh: '#58a6ff', cLow: '#484f58'
}};

const DEFAULTS = JSON.parse(JSON.stringify(C));

function readControls() {{
  C.L1 = document.getElementById('L1').checked;
  C.L2 = document.getElementById('L2').checked;
  C.L3 = document.getElementById('L3').checked;
  C.L4 = document.getElementById('L4').checked;
  C.L5 = document.getElementById('L5').checked;
  C.paramScale = parseInt(document.getElementById('paramScale').value) / 100;
  C.tradeAmp = parseInt(document.getElementById('tradeAmp').value);
  C.eqLine = parseInt(document.getElementById('eqLine').value) / 100;
  C.densScale = parseInt(document.getElementById('densScale').value);
  C.dustOp = parseInt(document.getElementById('dustOp').value);
  C.dustSz = parseInt(document.getElementById('dustSz').value) / 100;
  C.cCore = document.getElementById('cCore').value;
  C.cProfit = document.getElementById('cProfit').value;
  C.cLoss = document.getElementById('cLoss').value;
  C.cHigh = document.getElementById('cHigh').value;
  C.cLow = document.getElementById('cLow').value;

  document.getElementById('paramScaleV').textContent = C.paramScale.toFixed(2);
  document.getElementById('tradeAmpV').textContent = C.tradeAmp;
  document.getElementById('eqLineV').textContent = C.eqLine.toFixed(2);
  document.getElementById('densScaleV').textContent = C.densScale;
  document.getElementById('dustOpV').textContent = C.dustOp;
  document.getElementById('dustSzV').textContent = C.dustSz.toFixed(2);
}}

function resetControls() {{
  document.getElementById('L1').checked = true;
  document.getElementById('L2').checked = true;
  document.getElementById('L3').checked = true;
  document.getElementById('L4').checked = true;
  document.getElementById('L5').checked = true;
  document.getElementById('paramScale').value = 100;
  document.getElementById('tradeAmp').value = 80;
  document.getElementById('eqLine').value = 150;
  document.getElementById('densScale').value = 50;
  document.getElementById('dustOp').value = 25;
  document.getElementById('dustSz').value = 100;
  document.getElementById('cCore').value = '#f0c040';
  document.getElementById('cProfit').value = '#3fb950';
  document.getElementById('cLoss').value = '#f85149';
  document.getElementById('cHigh').value = '#58a6ff';
  document.getElementById('cLow').value = '#484f58';
  readControls();
}}

// ── Precomputed data ──
const trials = DATA.optuna.trials;
const dists = DATA.optuna.distributions;
const trades = DATA.trades;
const metrics = DATA.metrics;

// Strategy params (8 tunable, skip trend_strict and allow_flip)
const PARAM_NAMES = ['macd_fast','macd_slow','macd_signal','rsi_period','rsi_lower','rsi_upper','rsi_lookback','trend_tf'];
const PARAM_LABELS = ['MACD Fast','MACD Slow','MACD Signal','RSI Period','RSI Lower','RSI Upper','RSI Lookback','Trend TF'];

// Best trial params
let bestTrial = trials[0];
for (const t of trials) {{ if (t.v > bestTrial.v) bestTrial = t; }}
const bestParams = bestTrial.p;

// Normalize param value to 0-1
function normParam(name, val) {{
  const d = dists[name];
  if (!d) return 0.5;
  if (d.name === 'CategoricalDistribution') {{
    const choices = d.attributes.choices;
    return choices.length > 1 ? val / (choices.length - 1) : 0.5;
  }}
  const lo = d.attributes.low, hi = d.attributes.high;
  if (hi === lo) return 0.5;
  return (val - lo) / (hi - lo);
}}

// Trial score range
let minScore = Infinity, maxScore = -Infinity;
for (const t of trials) {{
  if (t.v > maxScore) maxScore = t.v;
  if (t.v < minScore) minScore = t.v;
}}
const scoreRange = maxScore - minScore || 1;

// Trade pnl range
let maxAbsPnl = 0;
for (const t of trades) {{ if (Math.abs(t.pnl_pct) > maxAbsPnl) maxAbsPnl = Math.abs(t.pnl_pct); }}

// Equity curve (cumulative pnl)
let cumPnl = [0];
for (let i = 0; i < trades.length; i++) {{ cumPnl.push(cumPnl[i] + trades[i].pnl_pct); }}
let eqMin = Infinity, eqMax = -Infinity;
for (const v of cumPnl) {{ if (v < eqMin) eqMin = v; if (v > eqMax) eqMax = v; }}
const eqRange = eqMax - eqMin || 1;

// Trial density bins (120 bins)
const N_BINS = 120;
const binCounts = new Array(N_BINS).fill(0);
const binScoreSums = new Array(N_BINS).fill(0);
for (const t of trials) {{
  const angle = (t.n / trials.length) * N_BINS;
  const bin = Math.min(Math.floor(angle), N_BINS - 1);
  binCounts[bin]++;
  binScoreSums[bin] += t.v;
}}
const binAvgScores = binCounts.map((c, i) => c > 0 ? binScoreSums[i] / c : 0);
let maxBinCount = 0;
for (const c of binCounts) {{ if (c > maxBinCount) maxBinCount = c; }}

// Dust sample (800 trials uniformly sampled)
const DUST_N = 800;
const dustTrials = [];
const step = Math.max(1, Math.floor(trials.length / DUST_N));
for (let i = 0; i < trials.length && dustTrials.length < DUST_N; i += step) {{
  dustTrials.push(trials[i]);
}}

// Hold bars range for width scaling
let maxHold = 0;
for (const t of trades) {{ if (t.hold_bars > maxHold) maxHold = t.hold_bars; }}

// ── p5.js sketch ──
const sketch = (p) => {{
  let cx, cy, baseR;
  let pg; // offscreen graphics for glow

  p.setup = () => {{
    const wrap = document.getElementById('canvas-wrap');
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    const cnv = p.createCanvas(w, h);
    cnv.parent('canvas-wrap');
    p.pixelDensity(2);
    p.textFont('JetBrains Mono');
    p.angleMode(p.RADIANS);
    p.smooth();
    cx = w / 2;
    cy = h / 2;
    baseR = Math.min(w, h) / 2 - 40;

    // Offscreen buffer for glow
    pg = p.createGraphics(w, h);
    pg.pixelDensity(2);
  }};

  p.windowResized = () => {{
    const wrap = document.getElementById('canvas-wrap');
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    p.resizeCanvas(w, h);
    cx = w / 2;
    cy = h / 2;
    baseR = Math.min(w, h) / 2 - 40;
    pg = p.createGraphics(w, h);
    pg.pixelDensity(2);
  }};

  function scaleR(nominalR) {{
    // Scale nominal radii (designed for ~600px radius) to actual canvas
    return nominalR * (baseR / 560);
  }}

  p.draw = () => {{
    readControls();
    p.background(13, 17, 23);

    // Subtle radial grid lines
    p.push();
    p.translate(cx, cy);
    p.noFill();
    p.stroke(33, 38, 45, 40);
    p.strokeWeight(0.5);
    const ringRadii = [80, 180, 200, 320, 340, 440, 460, 540];
    for (const r of ringRadii) {{ p.circle(0, 0, scaleR(r) * 2); }}
    // Angle guides
    for (let a = 0; a < p.TWO_PI; a += p.PI / 4) {{
      p.line(0, 0, Math.cos(a) * scaleR(560), Math.sin(a) * scaleR(560));
    }}
    p.pop();

    // Layer 1: Objective Score Glow
    if (C.L1) drawLayer1();
    // Layer 2: Strategy Parameter Star
    if (C.L2 && C.paramScale > 0) drawLayer2();
    // Layer 3: Trade Timeline
    if (C.L3) drawLayer3();
    // Layer 4: Trial Density Heatmap
    if (C.L4 && C.densScale > 0) drawLayer4();
    // Layer 5: Score Scatter Dust
    if (C.L5 && C.dustOp > 0) drawLayer5();

    // Center text
    p.push();
    p.translate(cx, cy);
    p.fill(200, 200, 200, 180);
    p.noStroke();
    p.textAlign(p.CENTER, p.CENTER);
    p.textSize(10);
    p.textFont('JetBrains Mono');
    p.text(DATA.symbol, 0, -8);
    p.textSize(7);
    p.fill(139, 148, 158, 150);
    p.text('log calmar ' + metrics.log_calmar.toFixed(2), 0, 6);
    p.pop();
  }};

  function drawLayer1() {{
    // Radial gradient glow at center
    const coreRgb = hexToRgb(C.cCore);
    const score = metrics.log_calmar;
    // score range roughly 0-8; normalize to 0-1
    const norm = Math.min(score / 7, 1);
    const maxR = scaleR(70) * (0.4 + 0.6 * norm);

    p.push();
    p.translate(cx, cy);
    p.noStroke();
    const steps = 40;
    for (let i = steps; i >= 0; i--) {{
      const t = i / steps;
      const r = maxR * t;
      // Core is brighter white, outer is colored
      const alpha = (1 - t * t) * 120 * (0.3 + 0.7 * norm);
      if (t < 0.3) {{
        // White hot center
        const w = 1 - t / 0.3;
        p.fill(
          coreRgb[0] + (255 - coreRgb[0]) * w,
          coreRgb[1] + (255 - coreRgb[1]) * w,
          coreRgb[2] + (255 - coreRgb[2]) * w,
          alpha * 1.5
        );
      }} else {{
        p.fill(coreRgb[0], coreRgb[1], coreRgb[2], alpha);
      }}
      p.circle(0, 0, r * 2);
    }}
    p.pop();
  }}

  function drawLayer2() {{
    const scale = C.paramScale;
    const innerR = scaleR(80);
    const outerR = scaleR(180);
    const rayLen = (outerR - innerR) * scale;

    p.push();
    p.translate(cx, cy);

    const pts = [];
    const nParams = PARAM_NAMES.length;

    for (let i = 0; i < nParams; i++) {{
      const angle = (i / nParams) * p.TWO_PI - p.HALF_PI;
      const name = PARAM_NAMES[i];
      const val = bestParams[name];
      const norm = normParam(name, val);
      const r = innerR + rayLen * norm;

      const x = Math.cos(angle) * r;
      const y = Math.sin(angle) * r;
      pts.push({{ x, y, angle, r, name: PARAM_LABELS[i], norm }});

      // Ray line
      p.stroke(88, 166, 255, 60);
      p.strokeWeight(1);
      p.line(
        Math.cos(angle) * innerR, Math.sin(angle) * innerR,
        x, y
      );

      // Ray tip dot
      p.noStroke();
      p.fill(88, 166, 255, 180);
      p.circle(x, y, 4);

      // Label
      p.fill(139, 148, 158, 160);
      p.textSize(7);
      p.textAlign(p.CENTER, p.CENTER);
      const labelR = r + 14;
      p.text(PARAM_LABELS[i], Math.cos(angle) * labelR, Math.sin(angle) * labelR);
    }}

    // Polygon
    p.noStroke();
    p.fill(88, 166, 255, 18);
    p.beginShape();
    for (const pt of pts) {{ p.vertex(pt.x, pt.y); }}
    p.endShape(p.CLOSE);

    // Polygon outline
    p.stroke(88, 166, 255, 50);
    p.strokeWeight(1);
    p.noFill();
    p.beginShape();
    for (const pt of pts) {{ p.vertex(pt.x, pt.y); }}
    p.endShape(p.CLOSE);

    p.pop();
  }}

  function drawLayer3() {{
    const amp = C.tradeAmp;
    const eqW = C.eqLine;
    if (amp === 0 && eqW === 0) return;

    const baseRing = scaleR(260);
    const maxBarH = scaleR(60) * (amp / 80);

    p.push();
    p.translate(cx, cy);

    // Compute total "width" for proportional spacing
    let totalHold = 0;
    for (const t of trades) totalHold += t.hold_bars;

    // Draw trade bars
    let cumAngle = 0;
    const profitRgb = hexToRgb(C.cProfit);
    const lossRgb = hexToRgb(C.cLoss);

    if (amp > 0) {{
      for (let i = 0; i < trades.length; i++) {{
        const t = trades[i];
        const barAngleWidth = (t.hold_bars / totalHold) * p.TWO_PI;
        const midAngle = cumAngle + barAngleWidth / 2 - p.HALF_PI;

        const pnlNorm = Math.abs(t.pnl_pct) / (maxAbsPnl || 1);
        const barH = maxBarH * pnlNorm;

        const isProfit = t.pnl_pct >= 0;
        const col = isProfit ? profitRgb : lossRgb;

        // Bar from base ring outward (profit) or inward (loss)
        const r1 = isProfit ? baseRing : baseRing - barH;
        const r2 = isProfit ? baseRing + barH : baseRing;

        // Draw as angular segment (arc wedge)
        const nSeg = Math.max(2, Math.ceil(barAngleWidth / 0.02));
        p.noStroke();
        p.fill(col[0], col[1], col[2], 140);
        p.beginShape();
        for (let s = 0; s <= nSeg; s++) {{
          const a = cumAngle - p.HALF_PI + (s / nSeg) * barAngleWidth;
          p.vertex(Math.cos(a) * r1, Math.sin(a) * r1);
        }}
        for (let s = nSeg; s >= 0; s--) {{
          const a = cumAngle - p.HALF_PI + (s / nSeg) * barAngleWidth;
          p.vertex(Math.cos(a) * r2, Math.sin(a) * r2);
        }}
        p.endShape(p.CLOSE);

        // Thin border
        p.stroke(col[0], col[1], col[2], 40);
        p.strokeWeight(0.3);
        p.noFill();
        p.beginShape();
        for (let s = 0; s <= nSeg; s++) {{
          const a = cumAngle - p.HALF_PI + (s / nSeg) * barAngleWidth;
          p.vertex(Math.cos(a) * r1, Math.sin(a) * r1);
        }}
        for (let s = nSeg; s >= 0; s--) {{
          const a = cumAngle - p.HALF_PI + (s / nSeg) * barAngleWidth;
          p.vertex(Math.cos(a) * r2, Math.sin(a) * r2);
        }}
        p.endShape(p.CLOSE);

        cumAngle += barAngleWidth;
      }}
    }}

    // Equity curve overlay
    if (eqW > 0) {{
      cumAngle = 0;
      p.noFill();
      p.stroke(255, 255, 255, 100);
      p.strokeWeight(eqW);
      p.beginShape();
      for (let i = 0; i <= trades.length; i++) {{
        const angle = (i / trades.length) * p.TWO_PI - p.HALF_PI;
        const eqNorm = (cumPnl[i] - eqMin) / eqRange; // 0 to 1
        const r = baseRing - scaleR(50) + scaleR(100) * eqNorm;
        p.vertex(Math.cos(angle) * r, Math.sin(angle) * r);
      }}
      p.endShape();
    }}

    // Base ring (subtle)
    p.noFill();
    p.stroke(48, 54, 61, 60);
    p.strokeWeight(0.5);
    p.circle(0, 0, baseRing * 2);

    p.pop();
  }}

  function drawLayer4() {{
    const scale = C.densScale;
    if (scale === 0) return;

    const innerR = scaleR(340);
    const maxH = scaleR(100) * (scale / 50);
    const hiRgb = hexToRgb(C.cHigh);
    const loRgb = hexToRgb(C.cLow);

    p.push();
    p.translate(cx, cy);

    const binAngle = p.TWO_PI / N_BINS;

    for (let i = 0; i < N_BINS; i++) {{
      const a1 = i * binAngle - p.HALF_PI;
      const a2 = (i + 1) * binAngle - p.HALF_PI;
      const countNorm = maxBinCount > 0 ? binCounts[i] / maxBinCount : 0;
      const scoreNorm = (binAvgScores[i] - minScore) / scoreRange;
      const barH = maxH * countNorm;

      // Lerp color by score
      const r = loRgb[0] + (hiRgb[0] - loRgb[0]) * scoreNorm;
      const g = loRgb[1] + (hiRgb[1] - loRgb[1]) * scoreNorm;
      const b = loRgb[2] + (hiRgb[2] - loRgb[2]) * scoreNorm;

      p.noStroke();
      p.fill(r, g, b, 100 + 80 * scoreNorm);

      const nSeg = Math.max(2, Math.ceil(binAngle / 0.03));
      p.beginShape();
      for (let s = 0; s <= nSeg; s++) {{
        const a = a1 + (s / nSeg) * binAngle;
        p.vertex(Math.cos(a) * innerR, Math.sin(a) * innerR);
      }}
      for (let s = nSeg; s >= 0; s--) {{
        const a = a1 + (s / nSeg) * binAngle;
        p.vertex(Math.cos(a) * (innerR + barH), Math.sin(a) * (innerR + barH));
      }}
      p.endShape(p.CLOSE);
    }}

    p.pop();
  }}

  function drawLayer5() {{
    const opacity = C.dustOp;
    const sz = C.dustSz;
    if (opacity === 0 || sz === 0) return;

    const centerR = scaleR(500);
    const jitter = scaleR(40);
    const hiRgb = hexToRgb(C.cHigh);
    const loRgb = hexToRgb(C.cLow);

    p.push();
    p.translate(cx, cy);
    p.noStroke();

    for (const t of dustTrials) {{
      const angle = (t.n / trials.length) * p.TWO_PI - p.HALF_PI;
      const scoreNorm = (t.v - minScore) / scoreRange;

      // Deterministic jitter from trial id
      const seed = t.id * 2654435761;
      const jx = ((seed & 0xFFFF) / 0xFFFF - 0.5) * 2;
      const jy = (((seed >> 16) & 0xFFFF) / 0xFFFF - 0.5) * 2;
      const r = centerR + jitter * jx;

      const cr = loRgb[0] + (hiRgb[0] - loRgb[0]) * scoreNorm;
      const cg = loRgb[1] + (hiRgb[1] - loRgb[1]) * scoreNorm;
      const cb = loRgb[2] + (hiRgb[2] - loRgb[2]) * scoreNorm;
      const alpha = (0.2 + 0.8 * scoreNorm) * opacity * (255 / 80);

      p.fill(cr, cg, cb, alpha);
      const dotR = (0.5 + 1.5 * scoreNorm) * sz;
      p.circle(Math.cos(angle) * r, Math.sin(angle) * r, dotR * 2);
    }}

    p.pop();
  }}

  // Export
  document.getElementById('btnExport').addEventListener('click', () => {{
    p.saveCanvas('run_portrait_{symbol}', 'png');
  }});

  document.getElementById('btnReset').addEventListener('click', () => {{
    resetControls();
  }});
}};

new p5(sketch);
</script>
</body>
</html>"""

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"Generated: {OUTPUT_FILE}")
    print(f"Size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
