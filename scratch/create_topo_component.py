# -*- coding: utf-8 -*-

code_topo = r"""
function TopologicalSortVisual() {
  const { isEnglish, t } = useUiCopy();
  const [activeTab, setActiveTab] = useState('dag'); // 'dag' | 'cycle' | 'alien'
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1400);

  // -------------------------------------------------------------
  // Mode 1: DAG Course Schedule Data
  // -------------------------------------------------------------
  const dagNodes = [
    { id: 0, name: '0: CS101', short: '0', x: 70, y: 70, cat: 'Base' },
    { id: 1, name: '1: Math101', short: '1', x: 70, y: 210, cat: 'Base' },
    { id: 2, name: '2: CS201', short: '2', x: 250, y: 70, cat: 'Core' },
    { id: 3, name: '3: Algo300', short: '3', x: 250, y: 210, cat: 'Core' },
    { id: 4, name: '4: ML400', short: '4', x: 450, y: 150, cat: 'Adv' },
    { id: 5, name: '5: Sys500', short: '5', x: 450, y: 40, cat: 'Adv' },
  ];
  const dagEdges = [
    { from: 0, to: 2, id: '0->2' },
    { from: 1, to: 2, id: '1->2' },
    { from: 1, to: 3, id: '1->3' },
    { from: 2, to: 4, id: '2->4' },
    { from: 2, to: 5, id: '2->5' },
    { from: 3, to: 4, id: '3->4' },
  ];
  const dagSteps = [
    {
      title: isEnglish ? 'Step 0: Initialize Indegrees & Enqueue In-0 Nodes' : 'Step 0: 初始化入度表并入队 0 入度节点',
      desc: isEnglish
        ? 'Scan all edges to compute indegrees. Nodes 0 (CS101) and 1 (Math101) have indegree=0 (no prerequisites). Enqueue [0, 1] as initial sources.'
        : '遍历全图统计入度。节点 0 (CS101) 和节点 1 (Math101) 入度为 0（无任何先修依赖），作为源点率先压入待处理队列 [0, 1]！',
      indegrees: { 0: 0, 1: 0, 2: 2, 3: 1, 4: 2, 5: 1 },
      queue: [0, 1],
      order: [],
      activeNode: null,
      activeEdges: [],
      doneNodes: [],
      status: 'init',
    },
    {
      title: isEnglish ? 'Step 1: Pop Node 0 & Decrement Successor 2' : 'Step 1: 弹出节点 0，消减后继 2 的入度',
      desc: isEnglish
        ? 'Pop 0 from queue, append to Order = [0]. Traverse successor 2: decrement indegree[2] from 2 to 1 (not 0 yet, stays pending).'
        : '从队列弹出节点 0 写入拓扑序。遍历后继节点 2，消减边 0->2：indegree[2] 从 2 减为 1（未归零，暂不入队）。',
      indegrees: { 0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 1 },
      queue: [1],
      order: [0],
      activeNode: 0,
      activeEdges: ['0->2'],
      doneNodes: [0],
      status: 'running',
    },
    {
      title: isEnglish ? 'Step 2: Pop Node 1 & Enqueue Nodes 2 and 3!' : 'Step 2: 弹出节点 1，节点 2 和 3 入度归零同时入队！',
      desc: isEnglish
        ? 'Pop 1, Order = [0, 1]. Successor 2 indegree drops 1->0 (Enqueued!). Successor 3 indegree drops 1->0 (Enqueued!). Queue = [2, 3].'
        : '弹出节点 1 写入拓扑序。后继节点 2 入度降为 0（入队！），后继节点 3 入度降为 0（入队！）。所有先修依赖清除，队列变为 [2, 3]！',
      indegrees: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 2, 5: 1 },
      queue: [2, 3],
      order: [0, 1],
      activeNode: 1,
      activeEdges: ['1->2', '1->3'],
      doneNodes: [0, 1],
      status: 'running',
    },
    {
      title: isEnglish ? 'Step 3: Pop Node 2 & Enqueue Node 5' : 'Step 3: 弹出节点 2，节点 5 入度归零入队',
      desc: isEnglish
        ? 'Pop 2, Order = [0, 1, 2]. Successor 4 indegree drops 2->1. Successor 5 indegree drops 1->0 (Enqueued!). Queue = [3, 5].'
        : '弹出节点 2 写入拓扑序。遍历后继 4 和 5：indegree[4] 从 2 减为 1；indegree[5] 从 1 降为 0（入队！）。队列变为 [3, 5]。',
      indegrees: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0 },
      queue: [3, 5],
      order: [0, 1, 2],
      activeNode: 2,
      activeEdges: ['2->4', '2->5'],
      doneNodes: [0, 1, 2],
      status: 'running',
    },
    {
      title: isEnglish ? 'Step 4: Pop Node 3 & Enqueue Node 4' : 'Step 4: 弹出节点 3，节点 4 入度归零入队',
      desc: isEnglish
        ? 'Pop 3, Order = [0, 1, 2, 3]. Successor 4 indegree drops 1->0 (All prerequisites satisfied, Enqueued!). Queue = [5, 4].'
        : '弹出节点 3 写入拓扑序。遍历后继 4：indegree[4] 从 1 降为 0（前置先修课全部修完，入队！）。队列变为 [5, 4]。',
      indegrees: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 },
      queue: [5, 4],
      order: [0, 1, 2, 3],
      activeNode: 3,
      activeEdges: ['3->4'],
      doneNodes: [0, 1, 2, 3],
      status: 'running',
    },
    {
      title: isEnglish ? 'Step 5: Pop Node 5' : 'Step 5: 弹出节点 5',
      desc: isEnglish
        ? 'Pop 5, Order = [0, 1, 2, 3, 5]. Node 5 has no outgoing edges. Queue = [4].'
        : '弹出节点 5 写入拓扑序。节点 5 无后继出边。队列剩下 [4]。',
      indegrees: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 },
      queue: [4],
      order: [0, 1, 2, 3, 5],
      activeNode: 5,
      activeEdges: [],
      doneNodes: [0, 1, 2, 3, 5],
      status: 'running',
    },
    {
      title: isEnglish ? 'Step 6: Pop Node 4' : 'Step 6: 弹出节点 4',
      desc: isEnglish
        ? 'Pop 4, Order = [0, 1, 2, 3, 5, 4]. Node 4 has no outgoing edges. Queue becomes empty.'
        : '弹出节点 4 写入拓扑序。节点 4 无后继出边。队列变空。',
      indegrees: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 },
      queue: [],
      order: [0, 1, 2, 3, 5, 4],
      activeNode: 4,
      activeEdges: [],
      doneNodes: [0, 1, 2, 3, 5, 4],
      status: 'running',
    },
    {
      title: isEnglish ? 'Step 7: Complete! Valid Topological Ordering' : 'Step 7: 拓扑排序顺利完成！判定为合法 DAG',
      desc: isEnglish
        ? 'Queue is empty and processed node count == 6 (|V|). Successfully produced valid topological order: [0, 1, 2, 3, 5, 4]!'
        : '队列为空且已处理节点总数等于 6（|V|）。成功生成合法选课拓扑序列：[0, 1, 2, 3, 5, 4]，判定图为无环 DAG！',
      indegrees: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 },
      queue: [],
      order: [0, 1, 2, 3, 5, 4],
      activeNode: null,
      activeEdges: [],
      doneNodes: [0, 1, 2, 3, 5, 4],
      status: 'success',
    },
  ];

  // -------------------------------------------------------------
  // Mode 2: Cycle Deadlock Data
  // -------------------------------------------------------------
  const cycleNodes = [
    { id: 0, name: '0: Task A', short: '0', x: 80, y: 130 },
    { id: 1, name: '1: Task B (In Cycle)', short: '1', x: 230, y: 70 },
    { id: 2, name: '2: Task C (In Cycle)', short: '2', x: 380, y: 70 },
    { id: 3, name: '3: Task D', short: '3', x: 380, y: 200 },
  ];
  const cycleEdges = [
    { from: 0, to: 1, id: '0->1' },
    { from: 1, to: 2, id: '1->2' },
    { from: 2, to: 1, id: '2->1', isCycle: true },
    { from: 1, to: 3, id: '1->3' },
  ];
  const cycleSteps = [
    {
      title: isEnglish ? 'Step 0: Initial Indegrees with Circular Dependency' : 'Step 0: 初始化入度表（含 1 <-> 2 互相循环依赖）',
      desc: isEnglish
        ? 'Notice cycle between 1 and 2. Only Node 0 has indegree=0. Queue = [0].'
        : '注意节点 1 和节点 2 之间存在互为先修的有向环。只有节点 0 入度为 0，初始队列 [0]。',
      indegrees: { 0: 0, 1: 2, 2: 1, 3: 1 },
      queue: [0],
      order: [],
      activeNode: null,
      activeEdges: [],
      doneNodes: [],
      cycleLockedNodes: [],
      status: 'init',
    },
    {
      title: isEnglish ? 'Step 1: Pop Node 0 & Decrement Node 1' : 'Step 1: 弹出节点 0，消减节点 1 入度',
      desc: isEnglish
        ? 'Pop 0, Order = [0]. Edge 0->1 relaxes indegree[1] from 2 to 1 (stays >0 due to cycle edge 2->1!). Queue becomes EMPTY.'
        : '弹出节点 0 写入拓扑序。后继节点 1 入度从 2 减为 1（仍 > 0，因为 2->1 的环边还在！）。队列变空！',
      indegrees: { 0: 0, 1: 1, 2: 1, 3: 1 },
      queue: [],
      order: [0],
      activeNode: 0,
      activeEdges: ['0->1'],
      doneNodes: [0],
      cycleLockedNodes: [],
      status: 'running',
    },
    {
      title: isEnglish ? 'Step 2: Premature Empty Queue -> Deadlock & Cycle Detected!' : 'Step 2: 队列提前排空！触发死锁，判定存在有向环！',
      desc: isEnglish
        ? 'Queue is empty, but processed count (1) < total nodes (4). Nodes 1, 2, 3 remain blocked forever with indegrees > 0. Graph is NOT a DAG!'
        : '队列为空但已处理节点数 1 < 总节点数 4！节点 1、2、3 残留入度 > 0，互相等待陷入死锁。判定：图中存在有向环，无法生成拓扑序！',
      indegrees: { 0: 0, 1: 1, 2: 1, 3: 1 },
      queue: [],
      order: [0],
      activeNode: null,
      activeEdges: ['1->2', '2->1'],
      doneNodes: [0],
      cycleLockedNodes: [1, 2, 3],
      status: 'cycle_error',
    },
  ];

  // -------------------------------------------------------------
  // Mode 3: Foreign Dictionary Word Pairs
  // -------------------------------------------------------------
  const alienPairs = [
    { first: 'hrn', second: 'hrf', diff: 'n -> f', id: 'pair-1' },
    { first: 'hrf', second: 'er', diff: 'h -> e', id: 'pair-2' },
    { first: 'er', second: 'enn', diff: 'r -> n', id: 'pair-3' },
    { first: 'enn', second: 'rfnn', diff: 'e -> r', id: 'pair-4' },
  ];
  const alienChain = [
    { char: 'h', inDeg: 0 },
    { char: 'e', inDeg: 1 },
    { char: 'r', inDeg: 1 },
    { char: 'n', inDeg: 1 },
    { char: 'f', inDeg: 1 },
  ];

  // Active steps selector
  const activeSteps = activeTab === 'dag' ? dagSteps : activeTab === 'cycle' ? cycleSteps : null;
  const maxStep = activeSteps ? activeSteps.length - 1 : 0;
  const currentStepData = activeSteps ? activeSteps[currentStep] : null;

  // Auto-play timer
  useEffect(() => {
    let timer;
    if (isPlaying && activeSteps) {
      timer = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= maxStep) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, speed);
    }
    return () => clearInterval(timer);
  }, [isPlaying, activeSteps, maxStep, speed]);

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setCurrentStep(0);
    setIsPlaying(false);
  };

  return (
    <section className="topo-interactive-root" aria-label={t('拓扑排序与 Kahn 算法交互式可视化', 'Topological Sort & Kahn Algorithm Interactive Visualizer')}>
      {/* 1. Header & Navigation Tabs */}
      <header className="topo-int-header">
        <div className="topo-int-title-group">
          <span className="topo-int-badge">
            <span className="topo-int-badge-dot" />
            {t('图论可视化实验室', 'Graph Theory Visual Lab')}
          </span>
          <h3>{t('Kahn 算法（入度 BFS）与环检测核心演练', "Kahn's Algorithm (Indegree BFS) & Cycle Detection")}</h3>
          <p>
            {t(
              '观察入度表动态消减、零入度节点就绪入队、以及遇到循环依赖时的死锁判定机制。',
              'Watch dynamic indegree reduction, zero-indegree queue activation, and cycle deadlock detection.'
            )}
          </p>
        </div>
        
        <div className="topo-mode-tabs" role="tablist">
          <button
            type="button"
            className={`topo-tab-btn ${activeTab === 'dag' ? 'is-active' : ''}`}
            onClick={() => handleTabChange('dag')}
          >
            🌿 {t('DAG 选课拓扑序 (6 节点)', 'DAG Course Schedule (6 Nodes)')}
          </button>
          <button
            type="button"
            className={`topo-tab-btn ${activeTab === 'cycle' ? 'is-active' : ''}`}
            onClick={() => handleTabChange('cycle')}
          >
            ⚠️ {t('有向环死锁检测 (4 节点)', 'Cycle Deadlock (4 Nodes)')}
          </button>
          <button
            type="button"
            className={`topo-tab-btn ${activeTab === 'alien' ? 'is-active' : ''}`}
            onClick={() => handleTabChange('alien')}
          >
            👽 {t('外星字典偏序抽取', 'Alien Dictionary')}
          </button>
        </div>
      </header>

      {/* 2. Interactive Stepper Toolbar (for DAG and Cycle modes) */}
      {activeTab !== 'alien' && (
        <div className="topo-int-controls">
          <div className="topo-step-indicator">
            <span className="topo-step-pill">
              {t('步骤', 'Step')} <b>{currentStep}</b> / {maxStep}
            </span>
            <span className="topo-step-title">{currentStepData?.title}</span>
          </div>

          <div className="topo-btn-group">
            <button
              type="button"
              className="topo-ctrl-btn"
              onClick={() => { setCurrentStep(0); setIsPlaying(false); }}
              title={t('重置', 'Reset')}
            >
              ⏮ {t('重置', 'Reset')}
            </button>
            <button
              type="button"
              className="topo-ctrl-btn"
              disabled={currentStep === 0}
              onClick={() => { setCurrentStep((s) => Math.max(0, s - 1)); setIsPlaying(false); }}
            >
              ◀ {t('上一步', 'Prev')}
            </button>
            <button
              type="button"
              className="topo-ctrl-btn topo-ctrl-primary"
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? `⏸ ${t('暂停', 'Pause')}` : `▶ ${t('自动播放', 'Auto Play')}`}
            </button>
            <button
              type="button"
              className="topo-ctrl-btn"
              disabled={currentStep >= maxStep}
              onClick={() => { setCurrentStep((s) => Math.min(maxStep, s + 1)); setIsPlaying(false); }}
            >
              {t('下一步', 'Next')} ▶
            </button>
          </div>
        </div>
      )}

      {/* 3. Main Stage: DAG & Cycle Modes */}
      {activeTab !== 'alien' && (
        <div className="topo-int-stage">
          {/* Top banner explaining the current step logic */}
          <div className={`topo-step-banner status-${currentStepData?.status}`}>
            <div className="topo-step-banner-icon">
              {currentStepData?.status === 'success' ? '🎉' : currentStepData?.status === 'cycle_error' ? '🚫' : '💡'}
            </div>
            <div className="topo-step-banner-text">
              <p>{currentStepData?.desc}</p>
            </div>
          </div>

          {/* SVG Graph Canvas */}
          <div className="topo-canvas-wrapper">
            <svg viewBox="0 0 550 280" className="topo-graph-svg" aria-label="Topological Graph Canvas">
              <defs>
                <marker id="topo-arrow" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#64748b" />
                </marker>
                <marker id="topo-arrow-active" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                  <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#3b82f6" />
                </marker>
                <marker id="topo-arrow-cycle" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                  <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#ef4444" />
                </marker>
                <filter id="glow-gold" x="-30%" y="-30%" width="160%" height="160%">
                  <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor="#f59e0b" floodOpacity="0.7" />
                </filter>
                <filter id="glow-blue" x="-30%" y="-30%" width="160%" height="160%">
                  <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor="#3b82f6" floodOpacity="0.7" />
                </filter>
                <filter id="glow-red" x="-30%" y="-30%" width="160%" height="160%">
                  <feDropShadow dx="0" dy="0" stdDeviation="5" floodColor="#ef4444" floodOpacity="0.8" />
                </filter>
              </defs>

              {/* Draw Edges */}
              {(activeTab === 'dag' ? dagEdges : cycleEdges).map((edge) => {
                const nodesList = activeTab === 'dag' ? dagNodes : cycleNodes;
                const src = nodesList.find((n) => n.id === edge.from);
                const dst = nodesList.find((n) => n.id === edge.to);
                if (!src || !dst) return null;

                const isActive = currentStepData?.activeEdges.includes(edge.id);
                const isCycleLocked = currentStepData?.cycleLockedNodes?.includes(edge.from) && currentStepData?.cycleLockedNodes?.includes(edge.to);
                const isDone = currentStepData?.doneNodes.includes(edge.from);

                // Curve path calculation
                let d = `M ${src.x} ${src.y} L ${dst.x} ${dst.y}`;
                if (edge.isCycle) {
                  // Arc curve for 2 -> 1 return edge
                  d = `M ${src.x} ${src.y + 15} Q ${(src.x + dst.x) / 2} ${src.y + 45} ${dst.x} ${dst.y + 15}`;
                }

                return (
                  <g key={edge.id} className="topo-edge-group">
                    <path
                      d={d}
                      className={`topo-edge-line ${isActive ? 'is-active' : ''} ${isCycleLocked ? 'is-cycle' : ''} ${isDone ? 'is-done' : ''}`}
                      markerEnd={isCycleLocked ? 'url(#topo-arrow-cycle)' : isActive ? 'url(#topo-arrow-active)' : 'url(#topo-arrow)'}
                    />
                    {isActive && (
                      <circle r="4" fill="#3b82f6">
                        <animateMotion dur="0.8s" repeatCount="indefinite" path={d} />
                      </circle>
                    )}
                  </g>
                );
              })}

              {/* Draw Nodes */}
              {(activeTab === 'dag' ? dagNodes : cycleNodes).map((node) => {
                const inDeg = currentStepData?.indegrees[node.id];
                const isInQueue = currentStepData?.queue.includes(node.id);
                const isDone = currentStepData?.doneNodes.includes(node.id);
                const isCurrentActive = currentStepData?.activeNode === node.id;
                const isCycleLocked = currentStepData?.cycleLockedNodes?.includes(node.id);

                let nodeClass = 'node-pending';
                let filter = undefined;
                if (isCycleLocked) {
                  nodeClass = 'node-cycle-locked';
                  filter = 'url(#glow-red)';
                } else if (isDone) {
                  nodeClass = 'node-done';
                } else if (isCurrentActive) {
                  nodeClass = 'node-active';
                  filter = 'url(#glow-blue)';
                } else if (isInQueue) {
                  nodeClass = 'node-in-queue';
                  filter = 'url(#glow-gold)';
                }

                return (
                  <g key={node.id} transform={`translate(${node.x}, ${node.y})`} className={`topo-node-group ${nodeClass}`} filter={filter}>
                    {/* Outer Circle */}
                    <circle r="24" className="topo-node-circle" />
                    {/* Node Text */}
                    <text textAnchor="middle" dy="-3" className="topo-node-name">
                      {node.short}
                    </text>
                    {/* Subtitle / Category */}
                    <text textAnchor="middle" dy="11" className="topo-node-sub">
                      {node.name.split(':')[1]?.trim() || ''}
                    </text>
                    {/* Indegree Badge */}
                    <g transform="translate(16, -16)">
                      <circle r="10" className={`topo-indegree-badge ${inDeg === 0 ? 'is-zero' : ''}`} />
                      <text textAnchor="middle" dy="3.5" className="topo-indegree-text">
                        {inDeg}
                      </text>
                    </g>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Real-time State Inspectors (3 Columns) */}
          <div className="topo-inspectors-grid">
            {/* Inspector 1: Indegree Table */}
            <div className="topo-inspector-card">
              <div className="topo-inspector-header">
                <span className="topo-insp-icon">📊</span>
                <h4>{t('实时入度表 indegree[]', 'Real-time Indegree Table')}</h4>
              </div>
              <div className="topo-indegree-chips">
                {(activeTab === 'dag' ? dagNodes : cycleNodes).map((node) => {
                  const deg = currentStepData?.indegrees[node.id];
                  const isDone = currentStepData?.doneNodes.includes(node.id);
                  const isLocked = currentStepData?.cycleLockedNodes?.includes(node.id);
                  return (
                    <div
                      key={node.id}
                      className={`topo-indegree-chip ${deg === 0 ? 'is-zero' : ''} ${isDone ? 'is-done' : ''} ${isLocked ? 'is-locked' : ''}`}
                    >
                      <span className="topo-chip-node">{node.short}</span>
                      <span className="topo-chip-val">{deg}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Inspector 2: FIFO Queue */}
            <div className="topo-inspector-card">
              <div className="topo-inspector-header">
                <span className="topo-insp-icon">📥</span>
                <h4>{t('就绪队列 Queue (FIFO)', 'Ready Queue (FIFO)')}</h4>
              </div>
              <div className="topo-queue-lane">
                {currentStepData?.queue.length === 0 ? (
                  <span className="topo-empty-tip">{t('【队列为空】', '[Queue Empty]')}</span>
                ) : (
                  currentStepData?.queue.map((nodeId, idx) => (
                    <span key={nodeId} className="topo-queue-box">
                      {idx === 0 && <small className="topo-front-tag">{t('队首', 'Front')}</small>}
                      <b>{nodeId}</b>
                    </span>
                  ))
                )}
              </div>
            </div>

            {/* Inspector 3: Topological Output Array */}
            <div className="topo-inspector-card">
              <div className="topo-inspector-header">
                <span className="topo-insp-icon">📋</span>
                <h4>{t('拓扑序列结果 Order[]', 'Topological Order Output')}</h4>
              </div>
              <div className="topo-order-lane">
                {currentStepData?.order.length === 0 ? (
                  <span className="topo-empty-tip">{t('【等待生成】', '[Pending...]')}</span>
                ) : (
                  currentStepData?.order.map((nodeId, idx) => (
                    <span key={nodeId} className="topo-order-box">
                      <small>#{idx + 1}</small>
                      <b>{nodeId}</b>
                    </span>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 4. Alien Dictionary Mode View */}
      {activeTab === 'alien' && (
        <div className="topo-alien-stage">
          <div className="topo-alien-explain">
            <h4>{t('外星文字典偏序抽取两步核心法', 'Alien Dictionary 2-Step Extraction')}</h4>
            <p>
              {t(
                '1. 只比较相邻单词对；2. 找到第一个不同字符，建立单向边 a -> b；3. 对所得字符 DAG 执行 Kahn 拓扑排序输出完整字母表！',
                '1. Compare only adjacent word pairs; 2. Find first differing character to form edge a -> b; 3. Run Kahn topological sort on the character DAG!'
              )}
            </p>
          </div>

          <div className="topo-alien-pairs-grid">
            {alienPairs.map((p) => (
              <div className="topo-alien-pair-card" key={p.id}>
                <div className="topo-alien-words">
                  <span className="topo-w-first">{p.first}</span>
                  <span className="topo-w-arrow">vs</span>
                  <span className="topo-w-second">{p.second}</span>
                </div>
                <div className="topo-alien-edge-badge">
                  <span>{t('提取偏序', 'Extracted Edge')}: <b>{p.diff}</b></span>
                </div>
              </div>
            ))}
          </div>

          <div className="topo-alien-chain-box">
            <span className="topo-chain-label">{t('字符依赖链', 'Character DAG Chain')}:</span>
            <div className="topo-chain-nodes">
              {alienChain.map((item, idx) => (
                <Fragment key={item.char}>
                  <div className="topo-alien-char-node">
                    <b>{item.char}</b>
                    <small>in:{item.inDeg}</small>
                  </div>
                  {idx < alienChain.length - 1 && <span className="topo-alien-arrow">➔</span>}
                </Fragment>
              ))}
            </div>
            <div className="topo-alien-result-box">
              {t('最终拓扑字典序', 'Final Topo Order')}: <code>hernf</code>
            </div>
          </div>
        </div>
      )}

      {/* 5. Memory & Mental Model Rhyme Footer */}
      <footer className="topo-memory-card">
        <div className="topo-memory-header">
          <span className="topo-mem-icon">🧠</span>
          <h4>{t('拓扑排序 Kahn 算法四步记忆心诀', "Kahn's Algorithm 4-Step Mental Flashcard")}</h4>
        </div>
        <div className="topo-rhymes-grid">
          <div className="topo-rhyme-item">
            <span className="topo-rhyme-num">1</span>
            <div>
              <b>{t('一数入度', 'Count Indegrees')}</b>
              <p>{t('建邻接表统计所有节点的初始入度 count。', 'Build adjacency list & tally initial in-degrees.')}</p>
            </div>
          </div>
          <div className="topo-rhyme-item">
            <span className="topo-rhyme-num">2</span>
            <div>
              <b>{t('二入零度', 'Enqueue Zeroes')}</b>
              <p>{t('所有入度为 0 的无前置节点作为源点全部压入队列。', 'Push all in-degree=0 nodes into the BFS queue.')}</p>
            </div>
          </div>
          <div className="topo-rhyme-item">
            <span className="topo-rhyme-num">3</span>
            <div>
              <b>{t('三砍后继', 'Relax & Decrement')}</b>
              <p>{t('出队节点入答案，后继入度减 1，新降为 0 者立即入队。', 'Pop node -> append to order -> decrement successors -> enqueue new 0s.')}</p>
            </div>
          </div>
          <div className="topo-rhyme-item">
            <span className="topo-rhyme-num">4</span>
            <div>
              <b>{t('四比数量', 'Compare Total Count')}</b>
              <p>{t('出队总数等于 |V| 则有拓扑序，小于 |V| 则必定有环！', 'Count == |V| means valid DAG; count < |V| means cycle detected!')}</p>
            </div>
          </div>
        </div>
      </footer>
    </section>
  );
}
"""

print("Code generated successfully!")
