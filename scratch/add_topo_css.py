# -*- coding: utf-8 -*-

css_rules = r"""
/* ── Enhanced Topological Sort Interactive Visualizer ──────────────── */
.topo-interactive-root {
  margin: 1.8rem 0 2.2rem;
  border: 1px solid rgba(26, 67, 84, 0.2);
  border-radius: 12px;
  background: #ffffff;
  box-shadow: 0 10px 30px rgba(15, 35, 45, 0.07);
  overflow: hidden;
  font-family: inherit;
}

.topo-int-header {
  padding: 1.25rem 1.4rem 1.1rem;
  background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
  border-bottom: 1px solid rgba(226, 232, 240, 0.9);
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
}

.topo-int-title-group {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.topo-int-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  align-self: flex-start;
  padding: 0.2rem 0.65rem;
  background: rgba(37, 99, 235, 0.1);
  color: #1d4ed8;
  font-size: 0.75rem;
  font-weight: 700;
  border-radius: 999px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.topo-int-badge-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #2563eb;
  animation: topo-pulse-dot 1.8s infinite;
}

@keyframes topo-pulse-dot {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.4); opacity: 0.6; }
}

.topo-int-title-group h3 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 800;
  color: #0f172a;
}

.topo-int-title-group p {
  margin: 0;
  font-size: 0.88rem;
  color: #475569;
  line-height: 1.5;
}

.topo-mode-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.2rem;
}

.topo-tab-btn {
  padding: 0.45rem 0.9rem;
  border-radius: 8px;
  border: 1px solid #cbd5e1;
  background: #ffffff;
  color: #334155;
  font-size: 0.84rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.18s ease;
}

.topo-tab-btn:hover {
  background: #f8fafc;
  border-color: #94a3b8;
  color: #0f172a;
}

.topo-tab-btn.is-active {
  background: #1e293b;
  border-color: #1e293b;
  color: #ffffff;
  box-shadow: 0 4px 12px rgba(15, 23, 42, 0.18);
}

/* Controls Toolbar */
.topo-int-controls {
  padding: 0.75rem 1.4rem;
  background: #ffffff;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.8rem;
}

.topo-step-indicator {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  flex-wrap: wrap;
}

.topo-step-pill {
  padding: 0.25rem 0.65rem;
  background: #e2e8f0;
  color: #1e293b;
  border-radius: 6px;
  font-size: 0.82rem;
  font-family: "IBM Plex Mono", monospace;
}

.topo-step-title {
  font-size: 0.92rem;
  font-weight: 700;
  color: #1e293b;
}

.topo-btn-group {
  display: flex;
  align-items: center;
  gap: 0.45rem;
}

.topo-ctrl-btn {
  padding: 0.38rem 0.75rem;
  border-radius: 6px;
  border: 1px solid #cbd5e1;
  background: #ffffff;
  color: #1e293b;
  font-size: 0.82rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s ease;
}

.topo-ctrl-btn:hover:not(:disabled) {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.topo-ctrl-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.topo-ctrl-primary {
  background: #2563eb;
  border-color: #2563eb;
  color: #ffffff;
}

.topo-ctrl-primary:hover:not(:disabled) {
  background: #1d4ed8;
  border-color: #1d4ed8;
  color: #ffffff;
}

/* Stage Area */
.topo-int-stage {
  padding: 1.25rem 1.4rem 1.4rem;
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
  background: #fafafa;
}

.topo-step-banner {
  display: flex;
  align-items: flex-start;
  gap: 0.9rem;
  padding: 0.9rem 1.1rem;
  border-radius: 8px;
  border-left: 4px solid #3b82f6;
  background: #eff6ff;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
}

.topo-step-banner.status-success {
  border-left-color: #10b981;
  background: #ecfdf5;
}

.topo-step-banner.status-cycle_error {
  border-left-color: #ef4444;
  background: #fef2f2;
}

.topo-step-banner-icon {
  font-size: 1.3rem;
  line-height: 1;
}

.topo-step-banner-text p {
  margin: 0;
  font-size: 0.9rem;
  color: #1e293b;
  line-height: 1.55;
  font-weight: 500;
}

/* SVG Graph */
.topo-canvas-wrapper {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 0.8rem;
  box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.02);
}

.topo-graph-svg {
  width: 100%;
  height: auto;
  display: block;
}

.topo-edge-line {
  stroke: #cbd5e1;
  stroke-width: 2.5;
  stroke-linecap: round;
  transition: all 0.3s ease;
}

.topo-edge-line.is-active {
  stroke: #3b82f6;
  stroke-width: 3.5;
  stroke-dasharray: 4;
  animation: topo-dash 1s linear infinite;
}

.topo-edge-line.is-cycle {
  stroke: #ef4444;
  stroke-width: 3.5;
  stroke-dasharray: 6;
  animation: topo-dash 0.8s linear infinite;
}

.topo-edge-line.is-done {
  stroke: #94a3b8;
  stroke-width: 1.8;
  opacity: 0.45;
  stroke-dasharray: 3 3;
}

@keyframes topo-dash {
  to {
    stroke-dashoffset: -20;
  }
}

.topo-node-group {
  cursor: default;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.topo-node-circle {
  fill: #ffffff;
  stroke: #94a3b8;
  stroke-width: 2.5;
  transition: all 0.3s ease;
}

.topo-node-name {
  font-size: 13px;
  font-weight: 800;
  fill: #1e293b;
  font-family: "IBM Plex Mono", monospace;
}

.topo-node-sub {
  font-size: 8.5px;
  font-weight: 600;
  fill: #64748b;
}

.topo-indegree-badge {
  fill: #f1f5f9;
  stroke: #cbd5e1;
  stroke-width: 1.5;
  transition: all 0.25s ease;
}

.topo-indegree-badge.is-zero {
  fill: #10b981;
  stroke: #059669;
}

.topo-indegree-text {
  font-size: 9.5px;
  font-weight: 800;
  fill: #334155;
  font-family: "IBM Plex Mono", monospace;
}

.topo-indegree-badge.is-zero + .topo-indegree-text {
  fill: #ffffff;
}

/* Node States */
.node-in-queue .topo-node-circle {
  stroke: #f59e0b;
  stroke-width: 3.5;
  fill: #fffbeb;
  animation: topo-pulse-queue 1.5s infinite;
}

.node-active .topo-node-circle {
  stroke: #3b82f6;
  stroke-width: 3.5;
  fill: #eff6ff;
}

.node-done .topo-node-circle {
  stroke: #10b981;
  stroke-width: 2;
  fill: #ecfdf5;
  opacity: 0.85;
}

.node-done .topo-node-name {
  fill: #059669;
}

.node-cycle-locked .topo-node-circle {
  stroke: #ef4444;
  stroke-width: 3.5;
  fill: #fef2f2;
  animation: topo-pulse-cycle 1.2s infinite;
}

.node-cycle-locked .topo-node-name {
  fill: #b91c1c;
}

@keyframes topo-pulse-queue {
  0%, 100% { stroke-width: 3.5; transform: scale(1); }
  50% { stroke-width: 4.5; transform: scale(1.04); }
}

@keyframes topo-pulse-cycle {
  0%, 100% { stroke-width: 3.5; }
  50% { stroke-width: 5; }
}

/* Real-time State Inspectors Grid */
.topo-inspectors-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
}

.topo-inspector-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 0.9rem 1rem;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.topo-inspector-header {
  display: flex;
  align-items: center;
  gap: 0.45rem;
}

.topo-insp-icon {
  font-size: 1.1rem;
}

.topo-inspector-header h4 {
  margin: 0;
  font-size: 0.88rem;
  font-weight: 700;
  color: #1e293b;
}

.topo-indegree-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.topo-indegree-chip {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.25rem 0.55rem;
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.8rem;
  font-weight: 600;
}

.topo-indegree-chip.is-zero {
  background: #ecfdf5;
  border-color: #10b981;
  color: #047857;
}

.topo-indegree-chip.is-done {
  opacity: 0.5;
}

.topo-indegree-chip.is-locked {
  background: #fef2f2;
  border-color: #ef4444;
  color: #b91c1c;
}

.topo-chip-node {
  color: #475569;
}

.topo-chip-val {
  font-weight: 800;
}

.topo-queue-lane,
.topo-order-lane {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  min-height: 40px;
}

.topo-empty-tip {
  color: #94a3b8;
  font-size: 0.8rem;
  font-style: italic;
}

.topo-queue-box,
.topo-order-box {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.3rem 0.65rem;
  border-radius: 6px;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.84rem;
  font-weight: 700;
}

.topo-queue-box {
  background: #fffbeb;
  border: 1px solid #f59e0b;
  color: #b45309;
}

.topo-front-tag {
  font-size: 0.65rem;
  background: #f59e0b;
  color: #ffffff;
  padding: 0.1rem 0.35rem;
  border-radius: 3px;
}

.topo-order-box {
  background: #ecfdf5;
  border: 1px solid #10b981;
  color: #047857;
}

.topo-order-box small {
  color: #059669;
  font-size: 0.7rem;
}

/* Alien Dictionary Mode */
.topo-alien-stage {
  padding: 1.3rem 1.4rem;
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
  background: #fafafa;
}

.topo-alien-explain {
  background: #eff6ff;
  border-left: 4px solid #3b82f6;
  border-radius: 8px;
  padding: 0.85rem 1.1rem;
}

.topo-alien-explain h4 {
  margin: 0 0 0.35rem;
  font-size: 0.95rem;
  font-weight: 700;
  color: #1e293b;
}

.topo-alien-explain p {
  margin: 0;
  font-size: 0.88rem;
  color: #334155;
  line-height: 1.5;
}

.topo-alien-pairs-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.8rem;
}

.topo-alien-pair-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 0.75rem 0.9rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.topo-alien-words {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.95rem;
}

.topo-w-first,
.topo-w-second {
  font-weight: 700;
  color: #1e293b;
}

.topo-w-arrow {
  color: #94a3b8;
  font-size: 0.75rem;
}

.topo-alien-edge-badge {
  font-size: 0.82rem;
  color: #2563eb;
  background: #eff6ff;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  display: inline-flex;
}

.topo-alien-chain-box {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 1rem 1.2rem;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
}

.topo-chain-label {
  font-size: 0.85rem;
  font-weight: 700;
  color: #475569;
}

.topo-chain-nodes {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.topo-alien-char-node {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: #f8fafc;
  border: 1.5px solid #3b82f6;
  border-radius: 8px;
  padding: 0.35rem 0.65rem;
  font-family: "IBM Plex Mono", monospace;
}

.topo-alien-char-node b {
  font-size: 1.1rem;
  color: #1e293b;
}

.topo-alien-char-node small {
  font-size: 0.65rem;
  color: #64748b;
}

.topo-alien-arrow {
  color: #3b82f6;
  font-weight: 800;
}

.topo-alien-result-box {
  margin-left: auto;
  font-size: 0.9rem;
  font-weight: 700;
  color: #047857;
  background: #ecfdf5;
  border: 1px solid #10b981;
  padding: 0.4rem 0.8rem;
  border-radius: 6px;
}

.topo-alien-result-box code {
  font-size: 1rem;
  color: #047857;
  font-weight: 800;
}

/* Footer Memory Rhyme */
.topo-memory-card {
  padding: 1.2rem 1.4rem;
  background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
  border-top: 1px solid #e2e8f0;
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
}

.topo-memory-header {
  display: flex;
  align-items: center;
  gap: 0.45rem;
}

.topo-mem-icon {
  font-size: 1.15rem;
}

.topo-memory-header h4 {
  margin: 0;
  font-size: 0.95rem;
  font-weight: 800;
  color: #0f172a;
}

.topo-rhymes-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 0.85rem;
}

.topo-rhyme-item {
  display: flex;
  align-items: flex-start;
  gap: 0.65rem;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 0.75rem 0.85rem;
}

.topo-rhyme-num {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: #2563eb;
  color: #ffffff;
  font-size: 0.75rem;
  font-weight: 800;
  flex-shrink: 0;
}

.topo-rhyme-item b {
  display: block;
  font-size: 0.85rem;
  color: #0f172a;
  margin-bottom: 0.2rem;
}

.topo-rhyme-item p {
  margin: 0;
  font-size: 0.78rem;
  color: #64748b;
  line-height: 1.4;
}
"""

with open("src/App.css", "r", encoding="utf-8") as f:
    app_css = f.read()

app_css += "\n" + css_rules

with open("src/App.css", "w", encoding="utf-8") as f:
    f.write(app_css)

print("CSS appended to src/App.css!")
