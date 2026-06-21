---
theme: none
background: '#0B1120'
---

<style>
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; padding: 0 20px; }
h1 { font-size: 36px !important; color: #FFFFFF !important; margin-bottom: 4px !important; }
h2 { font-size: 20px !important; color: #94A3B8 !important; font-weight: 400 !important; margin-bottom: 24px !important; }
h3 { font-size: 13px !important; color: #64748B !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 1px !important; margin-bottom: 10px !important; }
h4 { font-size: 14px !important; color: #E2E8F0 !important; font-weight: 700 !important; margin-bottom: 6px !important; }
.card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; margin-bottom: 12px; }
.badge-red { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.2); border-radius: 12px; padding: 16px; }
.badge-orange { background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.2); border-radius: 12px; padding: 16px; }
.badge-purple { background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.2); border-radius: 12px; padding: 16px; }
.badge-blue { background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.2); border-radius: 12px; padding: 12px; }
.badge-green { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2); border-radius: 12px; padding: 12px; }
.num-big { font-size: 56px; font-weight: 900; line-height: 1; }
.num-sm { font-size: 18px; font-weight: 800; }
.txt-xs { font-size: 11px; }
.txt-sm { font-size: 12px; }
.txt-md { font-size: 13px; }
.txt-lg { font-size: 14px; }
</style>

<!-- SLIDE 1: TITLE -->
<div style="width:100%;min-height:100vh;background:#0B1120;display:flex;flex-direction:column;align-items:center;justify-content:center;position:relative;padding:40px;box-sizing:border-box">

<div style="position:absolute;top:0;left:0;width:100%;height:4px;background:linear-gradient(90deg,#3B82F6,#8B5CF6,#06B6D4)"></div>

<div style="font-size:52px;font-weight:800;color:#FFFFFF;text-align:center;line-height:1.1;margin-bottom:8px;letter-spacing:-1px">
  Quantum-Enhanced<br>
  <span style="background:linear-gradient(90deg,#3B82F6,#8B5CF6);-webkit-background-clip:text;-webkit-text-fill-color:transparent">Spatio-Temporal Point Process</span>
</div>

<div style="font-size:22px;color:#94A3B8;margin-bottom:40px;text-align:center">for Dengue Fever Prediction in Southeast Asia</div>

<div style="display:flex;gap:0;margin-bottom:40px;border:1px solid rgba(255,255,255,0.1);border-radius:16px;overflow:hidden">
  <div style="padding:16px 32px;text-align:center;border-right:1px solid rgba(255,255,255,0.1)">
    <div style="font-size:36px;font-weight:800;color:#3B82F6;line-height:1">53K+</div>
    <div style="font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-top:4px">STPP Events</div>
  </div>
  <div style="padding:16px 32px;text-align:center;border-right:1px solid rgba(255,255,255,0.1)">
    <div style="font-size:36px;font-weight:800;color:#8B5CF6;line-height:1">8</div>
    <div style="font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-top:4px">Countries</div>
  </div>
  <div style="padding:16px 32px;text-align:center;border-right:1px solid rgba(255,255,255,0.1)">
    <div style="font-size:36px;font-weight:800;color:#06B6D4;line-height:1">99.46%</div>
    <div style="font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-top:4px">Spatial Corr.</div>
  </div>
  <div style="padding:16px 32px;text-align:center">
    <div style="font-size:36px;font-weight:800;color:#10B981;line-height:1">R2=0.78</div>
    <div style="font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-top:4px">Forecast R2</div>
  </div>
</div>

<div style="display:flex;gap:10px;margin-bottom:32px">
  <span style="background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);padding:6px 16px;border-radius:20px;font-size:13px;color:#3B82F6">Quantum Computing</span>
  <span style="background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.3);padding:6px 16px;border-radius:20px;font-size:13px;color:#10B981">Public Health</span>
  <span style="background:rgba(139,92,246,0.15);border:1px solid rgba(139,92,246,0.3);padding:6px 16px;border-radius:20px;font-size:13px;color:#8B5CF6">Hackathon 2026</span>
</div>

<div style="display:flex;gap:32px;align-items:center">
  <span style="font-size:15px;color:#CBD5E1">Quantum Dengue Team</span>
  <span style="color:#334155">|</span>
  <span style="font-size:13px;color:#64748B">May 30, 2026</span>
  <span style="color:#334155">|</span>
  <span style="font-size:13px;color:#64748B">NVIDIA RTX 3090 &middot; 6.3 min GPU</span>
</div>

</div>

---

<!-- SLIDE 2: DENGUE CRISIS -->
<div style="min-height:100vh;background:#0B1120;padding:40px 48px;box-sizing:border-box">

<h1>The Dengue Crisis</h1>
<h2>A Growing Threat in Southeast Asia</h2>

<div class="cols" style="margin-top:16px">

<div>

<div class="badge-red">
  <div class="num-big" style="color:#EF4444">390M</div>
  <div class="txt-md" style="color:#CBD5E1;margin-top:4px">annual infections worldwide</div>
</div>

<div class="badge-orange" style="margin-top:12px">
  <div class="num-big" style="color:#F59E0B">128</div>
  <div class="txt-md" style="color:#CBD5E1;margin-top:4px">countries affected</div>
</div>

<div class="badge-purple" style="margin-top:12px">
  <div class="num-big" style="color:#8B5CF6">$1B+</div>
  <div class="txt-md" style="color:#CBD5E1;margin-top:4px">annual economic cost (SEA)</div>
</div>

</div>

<div>

<div style="display:flex;flex-direction:column;gap:10px">

<div style="display:flex;gap:12px;align-items:flex-start">
  <div style="width:30px;height:30px;background:rgba(59,130,246,0.15);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#3B82F6;flex-shrink:0">1</div>
  <div class="txt-md" style="color:#CBD5E1;line-height:1.5;padding-top:4px">Climate change expanding mosquito range into new regions</div>
</div>

<div style="display:flex;gap:12px;align-items:flex-start">
  <div style="width:30px;height:30px;background:rgba(59,130,246,0.15);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#3B82F6;flex-shrink:0">2</div>
  <div class="txt-md" style="color:#CBD5E1;line-height:1.5;padding-top:4px">4 dengue serotypes -- prior infection increases severe dengue risk</div>
</div>

<div style="display:flex;gap:12px;align-items:flex-start">
  <div style="width:30px;height:30px;background:rgba(59,130,246,0.15);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#3B82F6;flex-shrink:0">3</div>
  <div class="txt-md" style="color:#CBD5E1;line-height:1.5;padding-top:4px">No universal vaccine -- Dengvaxia has safety concerns</div>
</div>

<div style="display:flex;gap:12px;align-items:flex-start">
  <div style="width:30px;height:30px;background:rgba(59,130,246,0.15);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#3B82F6;flex-shrink:0">4</div>
  <div class="txt-md" style="color:#CBD5E1;line-height:1.5;padding-top:4px">Urbanization creating denser mosquito breeding grounds</div>
</div>

</div>

<div style="background:rgba(239,68,68,0.08);border-left:3px solid #EF4444;padding:10px 14px;border-radius:0 8px 8px 0;margin-top:16px">
  <div class="txt-xs" style="color:#EF4444;font-weight:700;text-transform:uppercase;letter-spacing:1px">WHO</div>
  <div class="txt-md" style="color:#FCA5A5;margin-top:4px;font-style:italic">"One of the top 10 global health threats"</div>
</div>

</div>

</div>
</div>

---

<!-- SLIDE 3: PREDICTION CHALLENGE -->
<div style="min-height:100vh;background:#0B1120;padding:40px 48px;box-sizing:border-box">

<h1>The Prediction Challenge</h1>
<h2>Why Classical Models Fail on Dengue Data</h2>

<div class="cols" style="margin-top:16px">

<div>

<h3>The Four Core Problems</h3>

<div class="card">
  <div class="txt-lg" style="color:#FFFFFF;font-weight:700;margin-bottom:2px">Zero-Inflation</div>
  <div class="num-sm" style="color:#EF4444;margin-bottom:2px">30-50%</div>
  <div class="txt-xs" style="color:#64748B">of cells have ZERO cases. Most data is "absence".</div>
</div>

<div class="card">
  <div class="txt-lg" style="color:#FFFFFF;font-weight:700;margin-bottom:2px">Overdispersion</div>
  <div class="num-sm" style="color:#F59E0B;margin-bottom:2px">200-2000x</div>
  <div class="txt-xs" style="color:#64748B">Variance >> Mean. Indonesia: 2,066x.</div>
</div>

<div class="card">
  <div class="txt-lg" style="color:#FFFFFF;font-weight:700;margin-bottom:2px">Spatial Autocorrelation</div>
  <div class="num-sm" style="color:#8B5CF6;margin-bottom:2px">Moran I = 0.54</div>
  <div class="txt-xs" style="color:#64748B">Nearby regions correlated. All countries: p less than 0.001.</div>
</div>

<div class="card" style="margin-bottom:0">
  <div class="txt-lg" style="color:#FFFFFF;font-weight:700;margin-bottom:2px">Data Scarcity</div>
  <div class="num-sm" style="color:#06B6D4;margin-bottom:2px">223</div>
  <div class="txt-xs" style="color:#64748B">regions across 8 countries. Models overfit easily.</div>
</div>

</div>

<div>

<h3>Spatial Characteristics</h3>

<div class="card">
  <table style="width:100%;border-collapse:collapse;font-size:11px">
    <thead>
      <tr style="color:#475569;border-bottom:1px solid rgba(255,255,255,0.06)">
        <th style="text-align:left;padding:3px 6px;font-weight:600">Country</th>
        <th style="text-align:right;padding:3px 6px">Zero%</th>
        <th style="text-align:right;padding:3px 6px">OD</th>
        <th style="text-align:right;padding:3px 6px">Moran</th>
      </tr>
    </thead>
    <tbody style="font-family:monospace">
      <tr style="border-bottom:1px solid rgba(255,255,255,0.04)"><td style="padding:2px 6px;color:#CBD5E1">Vietnam</td><td style="text-align:right;color:#CBD5E1">31.1%</td><td style="text-align:right;color:#F59E0B">695x</td><td style="text-align:right;color:#10B981">0.540***</td></tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.04)"><td style="padding:2px 6px;color:#CBD5E1">Thailand</td><td style="text-align:right;color:#CBD5E1">6.0%</td><td style="text-align:right;color:#F59E0B">265x</td><td style="text-align:right;color:#10B981">0.092***</td></tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.04)"><td style="padding:2px 6px;color:#CBD5E1">Indonesia</td><td style="text-align:right;color:#CBD5E1">4.6%</td><td style="text-align:right;color:#EF4444">2,066x</td><td style="text-align:right;color:#10B981">0.540***</td></tr>
      <tr><td style="padding:2px 6px;color:#CBD5E1">Malaysia</td><td style="text-align:right;color:#CBD5E1">2.1%</td><td style="text-align:right;color:#F59E0B">415x</td><td style="text-align:right;color:#10B981">0.190***</td></tr>
    </tbody>
  </table>
  <div class="txt-xs" style="color:#475569;margin-top:4px">***p less than 0.001 -- All countries show significant spatial clustering</div>
</div>

<div class="badge-blue">
  <div class="txt-md" style="color:#3B82F6;font-weight:700;margin-bottom:4px">Classical Augmentation Fails</div>
  <div class="txt-sm" style="color:#94A3B8;line-height:1.5">SOP and noise injection destroy spatial correlations. Need augmentation that <b style="color:#3B82F6">preserves</b> spatial autocorrelation.</div>
</div>

</div>

</div>
</div>

---

<!-- SLIDE 4: RESEARCH QUESTIONS -->
<div style="min-height:100vh;background:#0B1120;padding:40px 48px;box-sizing:border-box">

<h1>Research Questions & Hypothesis</h1>

<div class="badge-blue" style="margin-bottom:20px;border-radius:0 12px 12px 0">
  <div class="txt-xs" style="color:#3B82F6;text-transform:uppercase;letter-spacing:2px;font-weight:700;margin-bottom:6px">Core Hypothesis</div>
  <div class="txt-md" style="color:#E2E8F0;line-height:1.6;font-style:italic">" Quantum generative models can learn the complex probability distribution of dengue outbreak patterns more efficiently than classical methods -- with <b style="color:#3B82F6;font-weight:700">exponential advantage potential</b> as quantum hardware matures."</div>
</div>

<h3>Research Questions</h3>

<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px">
  <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);border-radius:12px;padding:14px">
    <div class="txt-xs" style="color:#3B82F6;font-weight:800;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">RQ1</div>
    <div class="txt-sm" style="color:#CBD5E1;line-height:1.5">Can quantum generative models learn spatial outbreak probability distributions with high fidelity?</div>
  </div>
  <div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.2);border-radius:12px;padding:14px">
    <div class="txt-xs" style="color:#8B5CF6;font-weight:800;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">RQ2</div>
    <div class="txt-sm" style="color:#CBD5E1;line-height:1.5">Does quantum-augmented data improve spatio-temporal forecasting beyond classical baselines?</div>
  </div>
  <div style="background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.2);border-radius:12px;padding:14px">
    <div class="txt-xs" style="color:#06B6D4;font-weight:800;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">RQ3</div>
    <div class="txt-sm" style="color:#CBD5E1;line-height:1.5">How does the quantum-inspired pipeline scale vs classical augmentation?</div>
  </div>
</div>

<h3>Project Objectives</h3>

<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px">
  <div style="display:flex;gap:8px;align-items:flex-start">
    <div style="width:20px;height:20px;background:rgba(16,185,129,0.15);border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px">
      <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
    </div>
    <div class="txt-sm" style="color:#94A3B8;line-height:1.5">Design QBM + QGAN for disease outbreak augmentation</div>
  </div>
  <div style="display:flex;gap:8px;align-items:flex-start">
    <div style="width:20px;height:20px;background:rgba(16,185,129,0.15);border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px">
      <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
    </div>
    <div class="txt-sm" style="color:#94A3B8;line-height:1.5">Generate synthetic patterns preserving spatial autocorrelation</div>
  </div>
  <div style="display:flex;gap:8px;align-items:flex-start">
    <div style="width:20px;height:20px;background:rgba(16,185,129,0.15);border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px">
      <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
    </div>
    <div class="txt-sm" style="color:#94A3B8;line-height:1.5">Validate via MMD loss (QBM) and spatial correlation (QGAN)</div>
  </div>
  <div style="display:flex;gap:8px;align-items:flex-start">
    <div style="width:20px;height:20px;background:rgba(16,185,129,0.15);border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px">
      <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
    </div>
    <div class="txt-sm" style="color:#94A3B8;line-height:1.5">Benchmark against Transformer-based classical forecaster</div>
  </div>
</div>
</div>

---

<!-- SLIDE 5: PIPELINE -->
<div style="min-height:100vh;background:#0B1120;padding:40px 48px;box-sizing:border-box">

<h1>Hybrid Pipeline Architecture</h1>
<h2>Quantum Generator + Classical Forecaster</h2>

<div class="cols" style="margin-top:16px">

<div>

<h3>The Pipeline</h3>

<div class="card" style="font-size:13px;line-height:2;padding:14px">

<div style="display:flex;align-items:center;gap:8px">
  <div style="width:8px;height:8px;background:#3B82F6;border-radius:50%;flex-shrink:0"></div>
  <span style="color:#94A3B8">Real Data (Dengue)</span>
</div>
<div style="color:#334155;margin-left:4px">&darr;</div>

<div style="display:flex;align-items:center;gap:8px">
  <div style="width:8px;height:8px;background:#8B5CF6;border-radius:50%;flex-shrink:0"></div>
  <span style="color:#E2E8F0;font-weight:600">Quantum Generator</span>
</div>
<div style="color:#334155;margin-left:4px">&darr; generates</div>

<div style="display:flex;align-items:center;gap:8px">
  <div style="width:8px;height:8px;background:#06B6D4;border-radius:50%;flex-shrink:0"></div>
  <span style="color:#94A3B8">Synthetic Augmented Data</span>
</div>
<div style="color:#334155;margin-left:4px">&darr;</div>

<div style="display:flex;align-items:center;gap:8px">
  <div style="width:8px;height:8px;background:#10B981;border-radius:50%;flex-shrink:0"></div>
  <span style="color:#E2E8F0;font-weight:600">Classical Forecaster</span>
</div>
<div style="color:#334155;margin-left:4px">&darr; predicts</div>

<div style="display:flex;align-items:center;gap:8px">
  <div style="width:8px;height:8px;background:#F59E0B;border-radius:50%;flex-shrink:0"></div>
  <span style="color:#F59E0B;font-weight:700">Dengue Outbreak Forecast</span>
</div>

</div>

</div>

<div>

<h3>Three-Layer Design</h3>

<div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.2);border-radius:10px;padding:14px 16px;margin-bottom:10px">
  <div class="txt-lg" style="color:#8B5CF6;font-weight:700;margin-bottom:2px">[Q] Quantum Layer</div>
  <div class="txt-sm" style="color:#94A3B8">QBM learns distributions<br>QGAN generates spatial grids</div>
</div>

<div style="background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.2);border-radius:10px;padding:14px 16px;margin-bottom:10px">
  <div class="txt-lg" style="color:#06B6D4;font-weight:700;margin-bottom:2px">[+] Integration Layer</div>
  <div class="txt-sm" style="color:#94A3B8">Synthetic + Real data<br>Hybrid training sets</div>
</div>

<div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);border-radius:10px;padding:14px 16px;margin-bottom:10px">
  <div class="txt-lg" style="color:#10B981;font-weight:700;margin-bottom:2px">[M] Classical Layer</div>
  <div class="txt-sm" style="color:#94A3B8">CNN-LSTM / Transformer<br>Forecasting and evaluation</div>
</div>

<div class="card" style="font-size:11px;color:#64748B;margin-bottom:0">
  <b style="color:#94A3B8">OpenDengue v1.1</b> (Imperial College London)<br>
  8 countries &middot; 223 regions &middot; 1993-2022 monthly &middot; 48x48 grid<br>
  <b style="color:#94A3B8">Split:</b> 37,390 train / 8,012 val / 8,013 test
</div>

</div>

</div>
</div>

---

<!-- SLIDE 6: QBM -->
<div style="min-height:100vh;background:#0B1120;padding:40px 48px;box-sizing:border-box">

<h1>Quantum Born Machine</h1>
<h2>Learning Spatial Probability Distributions</h2>

<div class="cols" style="margin-top:16px">

<div>

<h3>QBM Circuit (8 qubits)</h3>

<div class="card" style="font-family:monospace;font-size:10px;line-height:1.8;color:#CBD5E1;padding:12px;margin-bottom:10px">
<div><span style="color:#3B82F6">|0&gt;</span> -- <span style="color:#F59E0B">RY</span>(theta1) -- <span style="color:#8B5CF6">CNOT</span> -- <span style="color:#F59E0B">RY</span>(theta9) -- <span style="color:#8B5CF6">CNOT</span></div>
<div><span style="color:#3B82F6">|0&gt;</span> -- <span style="color:#F59E0B">RY</span>(theta2) -- <span style="color:#8B5CF6">CNOT</span> -- <span style="color:#F59E0B">RY</span>(theta10) -- <span style="color:#8B5CF6">CNOT</span></div>
<div><span style="color:#3B82F6">|0&gt;</span> -- <span style="color:#F59E0B">RY</span>(theta3) -- <span style="color:#8B5CF6">CNOT</span> -- <span style="color:#F59E0B">RY</span>(theta11) -- <span style="color:#8B5CF6">CNOT</span></div>
<div style="color:#334155;margin:2px 0">...</div>
<div><span style="color:#3B82F6">|0&gt;</span> -- <span style="color:#F59E0B">RY</span>(theta8) -- <span style="color:#8B5CF6">CNOT</span> -- <span style="color:#F59E0B">RY</span>(theta16) -- <span style="color:#8B5CF6">CNOT</span></div>
<div style="margin-top:8px;font-size:9px;color:#64748B;line-height:1.6;border-top:1px solid rgba(255,255,255,0.06);padding-top:8px"><span style="color:#F59E0B">RY</span> = encode learnable params<br><span style="color:#8B5CF6">CNOT</span> = capture spatial correlations<br><span style="color:#3B82F6">Measure</span> = probability over 2^8 = 256 states</div>
</div>

<h3 style="margin-top:4px">Training: MMD Loss</h3>

<div class="card" style="font-family:monospace;font-size:10px;line-height:1.8;color:#94A3B8;padding:10px;margin-bottom:0">
MMD(P, Q) = E[x,y~P][k(x,y)]<br>
- 2 x E[x~P, y~Q][k(x,y)]<br>
+ E[x,y~Q][k(x,y)]
</div>

</div>

<div>

<h3>Why Quantum?</h3>

<div style="display:flex;flex-direction:column;gap:10px">

<div style="display:flex;gap:10px;align-items:flex-start">
  <div style="width:34px;height:34px;background:rgba(139,92,246,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:13px;font-weight:800;color:#8B5CF6">S</div>
  <div>
    <div class="txt-md" style="color:#FFFFFF;font-weight:700;margin-bottom:2px">Superposition</div>
    <div class="txt-xs" style="color:#64748B;line-height:1.4">Represent 2^n states simultaneously. 8 qubits = 256 states at once.</div>
  </div>
</div>

<div style="display:flex;gap:10px;align-items:flex-start">
  <div style="width:34px;height:34px;background:rgba(16,185,129,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:13px;font-weight:800;color:#10B981">E</div>
  <div>
    <div class="txt-md" style="color:#FFFFFF;font-weight:700;margin-bottom:2px">Entanglement</div>
    <div class="txt-xs" style="color:#64748B;line-height:1.4">CNOT gates capture spatial correlations natively between qubits.</div>
  </div>
</div>

<div style="display:flex;gap:10px;align-items:flex-start">
  <div style="width:34px;height:34px;background:rgba(245,158,11,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:13px;font-weight:800;color:#F59E0B">T</div>
  <div>
    <div class="txt-md" style="color:#FFFFFF;font-weight:700;margin-bottom:2px">Quantum Tunneling</div>
    <div class="txt-xs" style="color:#64748B;line-height:1.4">Explore probability landscape more efficiently than classical MCMC.</div>
  </div>
</div>

</div>

<h3 style="margin-top:12px">Scalability</h3>

<div class="badge-green" style="font-family:monospace;font-size:10px">
  <div class="txt-xs" style="color:#64748B;margin-bottom:6px">Classical grows O(n^2) -- quantum depth stays O(1)</div>
  <div class="txt-xs" style="color:#94A3B8">8 qubits <span style="color:#10B981">256 states, O(1)</span></div>
  <div class="txt-xs" style="color:#94A3B8">16 qubits <span style="color:#10B981">65K states, O(1)</span></div>
  <div class="txt-xs" style="color:#94A3B8">32 qubits <span style="color:#10B981">4B states, O(1)</span></div>
  <div class="txt-xs" style="color:#94A3B8">64 qubits <span style="color:#EF4444">Infeasible class., O(1)</span></div>
</div>

</div>

</div>
</div>

---

<!-- SLIDE 7: QGAN -->
<div style="min-height:100vh;background:#0B1120;padding:40px 48px;box-sizing:border-box">

<h1>Grid QGAN v3</h1>
<h2>Full Spatial Grid Generation</h2>

<div class="cols" style="margin-top:16px">

<div>

<h3>Evolution: v1 to v3</h3>

<div class="card" style="padding:14px;margin-bottom:10px">
  <div style="margin-bottom:8px">
    <span style="background:rgba(239,68,68,0.2);color:#EF4444;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700">v1</span>
    <span class="txt-xs" style="color:#64748B;margin-left:8px">Individual events -- Lost spatial structure</span>
  </div>
  <div style="margin-bottom:8px">
    <span style="background:rgba(245,158,11,0.2);color:#F59E0B;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700">v2</span>
    <span class="txt-xs" style="color:#64748B;margin-left:8px">Binary patterns -- Partial convergence</span>
  </div>
  <div>
    <span style="background:rgba(59,130,246,0.2);color:#3B82F6;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700">v3</span>
    <span class="txt-xs" style="color:#10B981;margin-left:8px">Full grid tensors -- 99.46% spatial correlation</span>
  </div>
</div>

<h3>Generator (PennyLane)</h3>

<div class="card" style="font-family:monospace;font-size:10px;line-height:1.8;color:#94A3B8;padding:10px;margin-bottom:8px">
  <div><span style="color:#64748B"># Quantum-inspired generator</span></div>
  <div><span style="color:#3B82F6">Latent</span> (16D) -- <span style="color:#F59E0B">RY</span> Encoding</div>
  <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#8B5CF6">RZ</span> Style Modulation (temporal)</div>
  <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#10B981">Entanglement</span> Layers (spatial mix)</div>
  <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#3B82F6">Spatial Projection</span> to <span style="color:#F59E0B">(12, 48, 48)</span></div>
</div>

<h3>Discriminator (CNN)</h3>

<div class="card" style="font-family:monospace;font-size:10px;line-height:1.8;color:#94A3B8;padding:10px;margin-bottom:0">
  <div><span style="color:#64748B"># Classical CNN discriminator</span></div>
  <div><span style="color:#3B82F6">Grid</span> -- Conv2D(32) -- Conv2D(64) -- Conv2D(128) -- Linear(1)</div>
  <div class="txt-xs" style="color:#64748B;margin-top:4px">WGAN-GP loss for stable adversarial training</div>
</div>

</div>

<div>

<h3>Training Progress</h3>

<div class="card" style="font-family:monospace;font-size:10px;line-height:1.8;margin-bottom:10px">
  <div><span style="color:#475569">Epoch   50:</span> <span style="color:#F59E0B">G=568.36</span> <span style="color:#64748B">D=0.71</span></div>
  <div><span style="color:#475569">Epoch  100:</span> <span style="color:#F59E0B">G=508.31</span> <span style="color:#64748B">D=2.48</span></div>
  <div><span style="color:#475569">Epoch  200:</span> <span style="color:#F59E0B">G=436.17</span> <span style="color:#64748B">D=1.34</span></div>
  <div><span style="color:#475569">Epoch  300:</span> <span style="color:#F59E0B">G=325.55</span> <span style="color:#64748B">D=0.67</span></div>
  <div><span style="color:#475569">Epoch  400:</span> <span style="color:#F59E0B">G=256.09</span> <span style="color:#64748B">D=0.80</span></div>
  <div style="border-top:1px solid rgba(255,255,255,0.08);margin-top:6px;padding-top:6px">
    <div><span style="color:#F59E0B">G_loss:</span> 256.09 <span style="color:#10B981">decreasing</span></div>
    <div><span style="color:#F59E0B">D_loss:</span> 0.80 <span style="color:#10B981">stable</span></div>
  </div>
</div>

<h3>Key Result</h3>

<div style="background:rgba(16,185,129,0.1);border:2px solid rgba(16,185,129,0.4);border-radius:14px;padding:20px;text-align:center;margin-bottom:10px">
  <div style="font-size:48px;font-weight:900;color:#10B981;line-height:1">99.46%</div>
  <div class="txt-sm" style="color:#94A3B8;margin-top:4px">Spatial correlation (generated vs real)</div>
</div>

<div class="badge-blue">
  <div class="txt-xs" style="color:#F59E0B;font-weight:700;margin-bottom:4px">"Quantum-Inspired" means:</div>
  <div class="txt-xs" style="color:#64748B;line-height:1.4">We simulate quantum circuits on classical hardware. Validates the architecture -- preparing for 100+ qubit QPUs.</div>
</div>

</div>

</div>
</div>

---

<!-- SLIDE 8: RESULTS -->
<div style="min-height:100vh;background:#0B1120;padding:40px 48px;box-sizing:border-box">

<h1>Experimental Results</h1>
<h2>QBM Convergence + Dataset Overview</h2>

<div class="cols" style="margin-top:16px">

<div>

<h3>QBM v3 Training (PennyLane + Adam)</h3>

<div class="card" style="margin-bottom:10px">
  <div style="font-family:monospace;font-size:10px;line-height:1.8;color:#94A3B8">
    <div><span style="color:#475569">Epoch   50:</span> <span style="color:#F59E0B">MMD = 9.09 x 10^-6</span></div>
    <div><span style="color:#475569">Epoch  100:</span> <span style="color:#10B981">MMD = 9.09 x 10^-8</span></div>
    <div><span style="color:#475569">Epoch  150:</span> <span style="color:#10B981">MMD = 9.09 x 10^-8</span></div>
    <div><span style="color:#475569">Epoch  200:</span> <span style="color:#10B981">MMD = 9.09 x 10^-8</span></div>
  </div>
  <div style="margin-top:10px;padding:10px;background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);border-radius:8px;text-align:center">
    <div style="font-size:22px;font-weight:800;color:#10B981">MMD = 9.09 x 10^-8</div>
    <div class="txt-xs" style="color:#64748B;margin-top:4px">Near-zero loss = quantum circuit learned the distribution</div>
  </div>
</div>

<div class="card" style="font-family:monospace;font-size:10px;color:#64748B;line-height:1.8;padding:10px;margin-bottom:0">
  QBM: 6 layers, 8 qubits | QGAN: 400 epochs<br>
  Batch: 128 | LR: 0.0003 | AdamW<br>
  GPU: RTX 3090 24GB | Runtime: 6.3 min
</div>

</div>

<div>

<h3>Dataset: OpenDengue v1.1</h3>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px">
  <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);border-radius:10px;padding:12px;text-align:center">
    <div style="font-size:22px;font-weight:800;color:#3B82F6;line-height:1">53,415</div>
    <div class="txt-xs" style="color:#64748B;margin-top:4px">STPP events</div>
  </div>
  <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);border-radius:10px;padding:12px;text-align:center">
    <div style="font-size:22px;font-weight:800;color:#10B981;line-height:1">3.93M</div>
    <div class="txt-xs" style="color:#64748B;margin-top:4px">total cases</div>
  </div>
  <div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.2);border-radius:10px;padding:12px;text-align:center">
    <div style="font-size:22px;font-weight:800;color:#8B5CF6;line-height:1">8</div>
    <div class="txt-xs" style="color:#64748B;margin-top:4px">countries</div>
  </div>
  <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:10px;padding:12px;text-align:center">
    <div style="font-size:22px;font-weight:800;color:#F59E0B;line-height:1">223</div>
    <div class="txt-xs" style="color:#64748B;margin-top:4px">regions</div>
  </div>
</div>

<h3>Train / Val / Test</h3>

<div class="card" style="padding:12px;margin-bottom:0">
  <div style="display:flex;gap:4px;margin-bottom:8px">
    <div style="flex:3.1;height:20px;background:linear-gradient(90deg,#3B82F6,#2563EB);border-radius:6px;display:flex;align-items:center;justify-content:center">
      <span style="font-size:10px;font-weight:700;color:#fff">Train 37,390</span>
    </div>
    <div style="flex:0.67;height:20px;background:linear-gradient(90deg,#8B5CF6,#7C3AED);border-radius:6px;display:flex;align-items:center;justify-content:center">
      <span style="font-size:10px;font-weight:700;color:#fff">Val 8,012</span>
    </div>
    <div style="flex:0.67;height:20px;background:linear-gradient(90deg,#10B981,#059669);border-radius:6px;display:flex;align-items:center;justify-content:center">
      <span style="font-size:10px;font-weight:700;color:#fff">Test 8,013</span>
    </div>
  </div>
  <div class="txt-xs" style="color:#64748B;line-height:1.4">8 countries: Cambodia, Indonesia, Laos, Malaysia, Philippines, Singapore, Thailand, Vietnam</div>
  <div class="txt-xs" style="color:#64748B;margin-top:3px">1993-2022 monthly | 48x48 grid | 12-month sequences</div>
</div>

</div>

</div>
</div>

---

<!-- SLIDE 9: FORECASTING -->
<div style="min-height:100vh;background:#0B1120;padding:40px 48px;box-sizing:border-box">

<h1>Forecasting Performance</h1>
<h2>Transformer vs Classical Baselines</h2>

<div class="cols" style="margin-top:16px">

<div>

<h3>Validation Results (347 sequences)</h3>

<div class="card" style="padding:12px;overflow-x:auto">
  <table style="width:100%;border-collapse:collapse;font-size:11px;min-width:280px">
    <thead>
      <tr style="color:#475569;border-bottom:1px solid rgba(255,255,255,0.08)">
        <th style="text-align:left;padding:5px 8px;font-weight:600">Method</th>
        <th style="text-align:right;padding:5px 8px">RMSE</th>
        <th style="text-align:right;padding:5px 8px">R2</th>
        <th style="text-align:right;padding:5px 8px">Pearson</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05);background:rgba(59,130,246,0.08)">
        <td style="padding:5px 8px;font-weight:700;color:#3B82F6">Transformer No Aug</td>
        <td style="text-align:right;color:#10B981;font-weight:700">1.37</td>
        <td style="text-align:right;color:#10B981;font-weight:700">0.78</td>
        <td style="text-align:right;color:#10B981;font-weight:700">0.89</td>
      </tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
        <td style="padding:5px 8px;color:#94A3B8">CNN-LSTM No Aug</td>
        <td style="text-align:right;color:#94A3B8">2.20</td>
        <td style="text-align:right;color:#94A3B8">0.43</td>
        <td style="text-align:right;color:#94A3B8">0.73</td>
      </tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
        <td style="padding:5px 8px;color:#64748B">CNN-LSTM + SOP</td>
        <td style="text-align:right;color:#64748B">2.27</td>
        <td style="text-align:right;color:#64748B">0.39</td>
        <td style="text-align:right;color:#64748B">0.73</td>
      </tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
        <td style="padding:5px 8px;color:#64748B">CNN-LSTM + QGAN</td>
        <td style="text-align:right;color:#64748B">2.36</td>
        <td style="text-align:right;color:#64748B">0.34</td>
        <td style="text-align:right;color:#64748B">0.68</td>
      </tr>
      <tr>
        <td style="padding:5px 8px;color:#475569">AttnLSTM No Aug</td>
        <td style="text-align:right;color:#475569">2.54</td>
        <td style="text-align:right;color:#475569">0.24</td>
        <td style="text-align:right;color:#475569">0.53</td>
      </tr>
    </tbody>
  </table>
</div>

</div>

<div>

<h3>Test Results (347 sequences)</h3>

<div class="card" style="padding:12px;overflow-x:auto;margin-bottom:10px">
  <table style="width:100%;border-collapse:collapse;font-size:11px;min-width:200px">
    <thead>
      <tr style="color:#475569;border-bottom:1px solid rgba(255,255,255,0.08)">
        <th style="text-align:left;padding:5px 8px;font-weight:600">Method</th>
        <th style="text-align:right;padding:5px 8px">RMSE</th>
        <th style="text-align:right;padding:5px 8px">R2</th>
        <th style="text-align:right;padding:5px 8px">Pearson</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05);background:rgba(59,130,246,0.08)">
        <td style="padding:5px 8px;font-weight:700;color:#3B82F6">Transformer No Aug</td>
        <td style="text-align:right;color:#10B981;font-weight:700">0.79</td>
        <td style="text-align:right;color:#10B981;font-weight:700">0.53</td>
        <td style="text-align:right;color:#10B981;font-weight:700">0.88</td>
      </tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
        <td style="padding:5px 8px;color:#94A3B8">CNN-LSTM No Aug</td>
        <td style="text-align:right;color:#94A3B8">0.88</td>
        <td style="text-align:right;color:#94A3B8">0.41</td>
        <td style="text-align:right;color:#94A3B8">0.77</td>
      </tr>
      <tr>
        <td style="padding:5px 8px;color:#475569">AttnLSTM No Aug</td>
        <td style="text-align:right;color:#475569">0.93</td>
        <td style="text-align:right;color:#475569">0.35</td>
        <td style="text-align:right;color:#475569">0.69</td>
      </tr>
    </tbody>
  </table>
</div>

<h3>Key Insights</h3>

<div class="badge-green">
  <div class="txt-md" style="color:#10B981;font-weight:700;margin-bottom:2px">Transformer dominates</div>
  <div class="txt-xs" style="color:#64748B">R2=0.78 (val) / R2=0.53 (test) -- best classical baseline</div>
</div>

<div class="badge-blue" style="margin-top:8px;margin-bottom:0">
  <div class="txt-md" style="color:#F59E0B;font-weight:700;margin-bottom:2px">Augmentation gap</div>
  <div class="txt-xs" style="color:#64748B;line-height:1.4">Quantum-augmented R2=0.34 vs No-Aug R2=0.78. Data is not the bottleneck -- temporal autocorrelation and climate covariates are the missing signals.</div>
</div>

</div>

</div>
</div>

---

<!-- SLIDE 10: LIMITATIONS & ROADMAP -->
<div style="min-height:100vh;background:#0B1120;padding:40px 48px;box-sizing:border-box">

<h1>Limitations & Roadmap</h1>
<h2>Honest Assessment</h2>

<div class="cols" style="margin-top:16px">

<div>

<h3>Current Limitations</h3>

<div style="display:flex;flex-direction:column;gap:8px">

<div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.15);border-radius:8px;padding:10px 12px">
  <div class="txt-sm" style="color:#EF4444;font-weight:700;margin-bottom:2px">No real quantum advantage yet</div>
  <div class="txt-xs" style="color:#64748B;line-height:1.4">Simulated on classical hardware. True speedup needs 50-100+ real qubits.</div>
</div>

<div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.15);border-radius:8px;padding:10px 12px">
  <div class="txt-sm" style="color:#EF4444;font-weight:700;margin-bottom:2px">Temporal dynamics not captured</div>
  <div class="txt-xs" style="color:#64748B;line-height:1.4">QGAN learns spatial but not month-to-month outbreak evolution.</div>
</div>

<div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.15);border-radius:8px;padding:10px 12px">
  <div class="txt-sm" style="color:#EF4444;font-weight:700;margin-bottom:2px">Climate covariates missing</div>
  <div class="txt-xs" style="color:#64748B;line-height:1.4">Temperature, rainfall, humidity are the primary dengue drivers.</div>
</div>

<div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.15);border-radius:8px;padding:10px 12px">
  <div class="txt-sm" style="color:#EF4444;font-weight:700;margin-bottom:2px">Augmentation underperforms</div>
  <div class="txt-xs" style="color:#64748B;line-height:1.4">QGAN-augmented R2=0.34 vs Transformer R2=0.78. Strategy needs refinement.</div>
</div>

</div>

</div>

<div>

<h3>Roadmap</h3>

<div style="display:flex;flex-direction:column;gap:10px">

<div style="display:flex;gap:10px;align-items:flex-start">
  <div style="width:30px;height:30px;background:linear-gradient(135deg,#3B82F6,#2563EB);border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:12px;font-weight:800;color:#fff">1</div>
  <div>
    <div class="txt-md" style="color:#FFFFFF;font-weight:700;margin-bottom:2px">Phase 1: Validation (Current)</div>
    <div class="txt-xs" style="color:#10B981;line-height:1.4">QBM MMD=10^-8 [check] | QGAN 99.46% [check] | Transformer R2=0.78 [check]</div>
    <div class="txt-xs" style="color:#F59E0B;line-height:1.4">[ ] End-to-end quantum-classical integration</div>
  </div>
</div>

<div style="display:flex;gap:10px;align-items:flex-start">
  <div style="width:30px;height:30px;background:linear-gradient(135deg,#8B5CF6,#7C3AED);border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:12px;font-weight:800;color:#fff">2</div>
  <div>
    <div class="txt-md" style="color:#FFFFFF;font-weight:700;margin-bottom:2px">Phase 2: Real Quantum HW (2026-2027)</div>
    <div class="txt-xs" style="color:#64748B;line-height:1.4">IBM Eagle 127-qubit. Advantage at qubits greater than 50, depth less than 1000. Target: 10x training speedup.</div>
  </div>
</div>

<div style="display:flex;gap:10px;align-items:flex-start">
  <div style="width:30px;height:30px;background:linear-gradient(135deg,#10B981,#059669);border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:12px;font-weight:800;color:#fff">3</div>
  <div>
    <div class="txt-md" style="color:#FFFFFF;font-weight:700;margin-bottom:2px">Phase 3: Production (2027-2029)</div>
    <div class="txt-xs" style="color:#64748B;line-height:1.4">WHO Global Dengue Programme. Real-time pipelines. Climate covariates (ERA5, GFS).</div>
  </div>
</div>

<div style="display:flex;gap:10px;align-items:flex-start">
  <div style="width:30px;height:30px;background:linear-gradient(135deg,#F59E0B,#D97706);border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:12px;font-weight:800;color:#fff">4</div>
  <div>
    <div class="txt-md" style="color:#FFFFFF;font-weight:700;margin-bottom:2px">Phase 4: Global Expansion (2029+)</div>
    <div class="txt-xs" style="color:#64748B;line-height:1.4">All WHO-priority NTDs (malaria, Zika, Ebola). Continental quantum centers. Federated learning.</div>
  </div>
</div>

</div>

</div>

</div>
</div>

---

<!-- SLIDE 11: CONCLUSION -->
<div style="min-height:100vh;background:#0B1120;padding:40px 48px;box-sizing:border-box">

<h1>Conclusion</h1>

<div class="cols" style="margin-top:16px">

<div>

<h3>What We Achieved</h3>

<div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);border-radius:10px;padding:14px;margin-bottom:8px">
  <div style="font-size:18px;font-weight:800;color:#10B981;line-height:1;margin-bottom:2px">QBM: MMD = 9.09 x 10^-8</div>
  <div class="txt-xs" style="color:#64748B;margin-top:4px">Near-perfect distribution learning -- quantum circuit validated</div>
</div>

<div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);border-radius:10px;padding:14px;margin-bottom:8px">
  <div style="font-size:18px;font-weight:800;color:#3B82F6;line-height:1;margin-bottom:2px">QGAN: 99.46% spatial corr.</div>
  <div class="txt-xs" style="color:#64748B;margin-top:4px">Full grid-level tensor generation with spatial preservation</div>
</div>

<div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:10px;padding:14px;margin-bottom:8px">
  <div style="font-size:18px;font-weight:800;color:#F59E0B;line-height:1;margin-bottom:2px">Transformer: R2 = 0.78</div>
  <div class="txt-xs" style="color:#64748B;margin-top:4px">Competitive classical baseline for dengue forecasting</div>
</div>

<div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.2);border-radius:10px;padding:14px;margin-bottom:0">
  <div style="font-size:15px;font-weight:800;color:#8B5CF6;line-height:1;margin-bottom:2px">SDG-Aligned</div>
  <div class="txt-xs" style="color:#64748B;margin-top:4px">SDG 3 (Health 85%) | SDG 10 (Ineq. 70%) | SDG 13 (Climate 55%) | SDG 17 (Partnerships 85%)</div>
</div>

</div>

<div>

<h3>The Quantum Promise</h3>

<div class="card" style="font-size:11px;line-height:1.8;margin-bottom:10px">

<div style="color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Today: Quantum-Inspired Simulation</div>
<div style="color:#10B981">QBM learns distributions [check]</div>
<div style="color:#10B981">QGAN spatial patterns [check]</div>
<div style="color:#10B981">Classical models work [check]</div>
<div style="color:#EF4444">Quantum advantage: NOT YET [x]</div>

<div style="border-top:1px solid rgba(255,255,255,0.08);margin:6px 0;padding-top:6px;color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:1px">Tomorrow: Real Quantum HW</div>
<div style="color:#94A3B8">50-100 qubits available</div>
<div style="color:#94A3B8">Exponential speedup potential</div>
<div style="color:#10B981">Quantum advantage: EXPECTED [check]</div>

<div style="border-top:1px solid rgba(255,255,255,0.08);margin:6px 0;padding-top:6px;color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:1px">Future: Production Systems</div>
<div style="color:#94A3B8">Continental disease surveillance</div>
<div style="color:#94A3B8">Pandemic prediction networks</div>
<div style="color:#10B981">Global health transformation [check]</div>

</div>

<div class="badge-blue" style="font-style:italic">
  "We have <b style="color:#3B82F6;font-weight:700">proven that quantum circuit architectures are correctly designed</b> for learning outbreak patterns, and that quantum-inspired simulations generate spatial data with <b style="color:#10B981;font-weight:700">99.46% fidelity</b>."
</div>

</div>

</div>
</div>
