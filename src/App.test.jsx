import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

describe('App', () => {
  beforeEach(() => {
    window.history.replaceState(null, '', '/');
    window.location.hash = '';
    sessionStorage.clear();
    vi.restoreAllMocks();
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const requestUrl = String(input);
      const english = requestUrl.includes('.en.md');
      const chineseContent = [
        '# 中文教程',
        '',
        '这是中文版本。',
        '',
        '```quiz',
        'title: Quick Check',
        'question: CUDA thread blocks are scheduled onto which hardware unit?',
        'answer: B',
        'A. Host compiler',
        'B. GPU SM',
        'C. Browser runtime',
        'explanation: Blocks are assigned to streaming multiprocessors.',
        '```',
        '',
        '```python',
        'def can_jump(nums):',
        '    if right - left + 1 == k and window == need:',
        '        return True',
        '```',
        '',
        'Inline math $QK^T$ and display math:',
        '',
        '$$',
        '\\sum_i x_i',
        '$$',
      ].join('\n');

      return {
        ok: true,
        text: async () =>
          requestUrl.includes('Business%20Algorithm%20TODO') || requestUrl.includes('Business Algorithm TODO')
            ? `${english ? '# Business algorithm system map' : '# 第一部分：系统总览与数据基础'}\n\n\`\`\`business-algorithm-map\n\`\`\``
            : english
              ? '# English tutorial\n\nThis is the English version.'
            : requestUrl.includes('SystemDesign03')
              ? '# System Design 03 · 数据库扩展\n\n分片与副本。'
            : requestUrl.includes('SystemDesign05')
              ? '# System Design 05 · 可靠性、复制与故障切换'
            : requestUrl.includes('SystemDesign06')
              ? '# System Design 06 · 消息队列'
            : requestUrl.includes('SystemDesign07')
              ? '# System Design 07 · 设计图片分享与 Home Feed'
            : requestUrl.includes('SystemDesign08')
              ? '# System Design 08 · 异步 LLM RL 平台\n\nSample Admission QPS ~60 /s。'
            : requestUrl.includes('SystemDesign09')
              ? '# System Design 09 · 一致性哈希\n\n节点变化时只迁移相邻区间。'
            : requestUrl.includes('SystemDesign10')
              ? '# System Design 10 · Flash Sale (秒杀)'
            : requestUrl.includes('SystemDesign01D')
              ? '# System Design 01D · Redis'
            : requestUrl.includes('CoreSkills09')
              ? '## 双源推进可行性：Interleaving String\n\n$$dp[i][j] = (dp[i-1][j] \\land s_1[i-1] == s_3[i+j-1]) \\lor (dp[i][j-1] \\land s_2[j-1] == s_3[i+j-1])$$\n'
            : chineseContent,
      };
    });
  });

  it('starts on a prominent home page with MLSYS navigation', () => {
    render(<App />);

    expect(screen.getByRole('heading', { name: /ML \/ LLM 技术复习笔记/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^mlsys/i })).toBeInTheDocument();
  });

  it('switches all homepage copy between Chinese and English', () => {
    render(<App />);

    expect(screen.getByRole('heading', { name: '笔记板块' })).toBeInTheDocument();
    expect(screen.getByText(/沿一次线上请求拆解召回/)).toBeInTheDocument();
    expect(screen.getByText('板块')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'English' }));

    expect(screen.getByRole('heading', { name: 'ML / LLM interview notes' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Browse the notes' })).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: /Business Algorithms/ })).toHaveLength(2);
    expect(screen.getByText(/Retrieval, ranking, list decisions/)).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'About the author' })).toBeInTheDocument();
    expect(screen.getByText(/I'm currently looking for new opportunities/)).toBeInTheDocument();
    expect(screen.getByText('Languages')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '中文' }));

    expect(screen.getByRole('heading', { name: /ML \/ LLM 技术复习笔记/ })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '关于作者' })).toBeInTheDocument();
  });

  it('shows author contact details in the About section', () => {
    render(<App />);

    expect(screen.getByRole('heading', { name: /关于作者/ })).toBeInTheDocument();
    expect(screen.getByText(/agent memory/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /githubgithub\.com\/currytang/i })).toHaveAttribute(
      'href',
      'https://github.com/CurryTang',
    );
    expect(screen.getByRole('link', { name: /linkedinzhikai chen/i })).toHaveAttribute(
      'href',
      'https://www.linkedin.com/in/zhikai-chen-435252129',
    );
    expect(screen.getByRole('link', { name: /emailchenzh85@msu\.edu/i })).toHaveAttribute(
      'href',
      'mailto:chenzh85@msu.edu',
    );
  });

  it('keeps the same tutorial selected while switching languages in place', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'MLSYS' }));
    fireEvent.click(await screen.findByRole('button', { name: /MLSYS1 · GPU 体系结构入门/i }));

    const initialHeading = await screen.findByRole('heading', {
      name: /mlsys1/i,
    });

    expect(initialHeading).toBeInTheDocument();
    expect(await screen.findByText('这是中文版本。')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /english/i }));

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /mlsys1/i })).toBeInTheDocument();
    });

    expect(screen.getByRole('button', { name: /english/i })).toHaveAttribute('aria-pressed', 'true');
    expect(await screen.findByText('This is the English version.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /MLSYS1 · GPU Architecture Basics/i })).toBeInTheDocument();
  });

  it('renders interactive multiple-choice practice blocks', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'MLSYS' }));

    expect(await screen.findByText('Quick Check')).toBeInTheDocument();
    expect(screen.getByText(/CUDA thread blocks/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Browser runtime/i }));

    expect(await screen.findByText(/再想一下/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /GPU SM/i }));

    expect(await screen.findByText(/回答正确/)).toBeInTheDocument();
    expect(screen.getByText(/streaming multiprocessors/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /收起/i }));

    expect(screen.queryByText(/CUDA thread blocks/)).not.toBeInTheDocument();
  });

  it('renders enhanced code blocks with language labels', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'MLSYS' }));

    expect(await screen.findByText('Python')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Copy' })).toBeInTheDocument();
    expect(screen.getByText('def')).toHaveClass('code-token', 'keyword');
  });

  it('does not treat Python equality operators as Obsidian highlights', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'MLSYS' }));

    const pythonFrame = (await screen.findByText('Python')).closest('.code-frame');
    expect(pythonFrame.querySelector('code')).toHaveTextContent(
      'if right - left + 1 == k and window == need:',
    );
    expect(pythonFrame.querySelector('mark')).toBeNull();
  });

  it('does not treat double equality operators in LaTeX math as Obsidian highlights', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 9 ·/i }));

    expect(await screen.findByRole('heading', { name: /双源推进可行性：Interleaving String/i })).toBeInTheDocument();

    const mathAnnotations = Array.from(document.querySelectorAll('annotation[encoding="application/x-tex"]')).map(
      (node) => node.textContent,
    );
    expect(mathAnnotations.some((tex) => tex.includes('s_1[i-1] == s_3[i+j-1]'))).toBe(true);
    expect(document.querySelector('mark')).toBeNull();
  });

  it('renders Markdown math through KaTeX without losing LaTeX commands', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'MLSYS' }));

    expect(await screen.findByText(/Inline math/)).toBeInTheDocument();

    const annotations = Array.from(document.querySelectorAll('annotation[encoding="application/x-tex"]')).map(
      (node) => node.textContent,
    );

    expect(annotations).toContain('QK^T');
    expect(annotations).toContain('\\sum_i x_i');
  });

  it('keeps the reader sidebar scoped to the current section', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'MLSYS' }));

    expect(await screen.findByRole('heading', { name: /MLSYS1/i })).toBeInTheDocument();

    const sidebar = document.querySelector('.notes-panel');
    expect(sidebar).not.toBeNull();
    expect(within(sidebar).getByRole('heading', { name: 'MLSYS' })).toBeInTheDocument();
    expect(within(sidebar).getByText('本板块共 18 篇笔记')).toBeInTheDocument();
    expect(within(sidebar).queryByText('LLM八股')).not.toBeInTheDocument();
    expect(within(sidebar).queryByText('LeetCode')).not.toBeInTheDocument();
  });

  it('opens the LeetCode section from the top navigation', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));

    expect(await screen.findByRole('heading', { name: /Core Skills 1/i })).toBeInTheDocument();
    expect(screen.getAllByText('CoreSkills01 Design Dynamic Array.md')).toHaveLength(2);
    expect(screen.getByText('本板块共 21 篇笔记')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Core Skills 15 · Two Pointers/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Core Skills 16 · Sliding Window/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Core Skills 17 · Stack & Monotonic Stack/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Core Skills 18 · Binary Search/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Core Skills 15 · Two Pointers/i }));

    expect(await screen.findByRole('heading', { name: /Two Pointers/i })).toBeInTheDocument();
    expect(screen.getAllByText('CoreSkills15 Two Pointers.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /Core Skills 16 · Sliding Window/i }));

    expect(await screen.findByRole('heading', { name: /中文教程/i })).toBeInTheDocument();
    expect(screen.getAllByText('CoreSkills16 Sliding Window.md')).toHaveLength(2);
  });

  it('renders the interactive 3Sum two-pointer walkthrough', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills15')
          ? '# Two Pointers\n\n```three-sum-demo\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 15 · Two Pointers/i }));

    expect(await screen.findByRole('region', { name: '3Sum 双指针演示' })).toBeInTheDocument();
    expect(screen.getByText('排序并初始化')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /跳到步骤 5: 固定 -1，命中第一组/i }));

    expect(screen.getByText('固定 -1，命中第一组')).toBeInTheDocument();
    expect(screen.getByText('[-1, -1, 2]')).toBeInTheDocument();
  });

  it('renders the standalone sliding window template visual', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills16')
          ? '# Sliding Window\n\n```sliding-window-demo\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 16 · Sliding Window/i }));

    const visual = await screen.findByRole('region', { name: '滑动窗口万能模板演示' });
    expect(within(visual).getByText('右扩：加入 A')).toBeInTheDocument();

    fireEvent.change(within(visual).getByRole('slider', { name: '选择滑动窗口演示步骤' }), {
      target: { value: '6' },
    });

    expect(within(visual).getByText('加入 A 后条件失效')).toBeInTheDocument();
    expect(within(visual).getByText('不合法')).toBeInTheDocument();
  });

  it('presents Permutation in String as the fixed-window branch', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills16')
          ? '# Sliding Window\n\n```sliding-window-patterns\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 16 · Sliding Window/i }));

    const atlas = await screen.findByRole('region', { name: '五道滑动窗口题模板对照' });
    fireEvent.click(within(atlas).getByRole('tab', { name: /LC 567.*Permutation in String/i }));

    expect(within(atlas).getByText('if：窗口长度 > |s1|')).toBeInTheDocument();
    expect(within(atlas).getByText('最多移出一个左端字符')).toBeInTheDocument();
    expect(within(atlas).getByText('窗口满 |s1| 时比较频次表')).toBeInTheDocument();
    expect(within(atlas).queryByText('3 · while 内')).not.toBeInTheDocument();
  });

  it('maps longest substring code to the sliding window template', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills16')
          ? '# Sliding Window\n\n```longest-substring-demo\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 16 · Sliding Window/i }));

    const visual = await screen.findByRole('region', { name: '最长无重复子串代码映射演示' });
    expect(within(visual).getByText('同一行骨架，逐项填入本题条件')).toBeInTheDocument();
    expect(within(visual).getByText('外层 loop right')).toBeInTheDocument();

    fireEvent.change(within(visual).getByRole('slider', { name: '选择最长无重复子串演示步骤' }), {
      target: { value: '6' },
    });

    expect(within(visual).getByText('right = 3：先加入第二个 a')).toBeInTheDocument();
    expect(within(visual).getByText('存在频次大于 1')).toBeInTheDocument();

    fireEvent.click(within(visual).getByRole('button', { name: '下一步' }));
    expect(within(visual).getByText('移除旧 a，left 从 0 变成 1')).toBeInTheDocument();
  });

  it('renders the unified monotonic stack walkthrough', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills17')
          ? '# Stack\n\n```monotonic-stack-demo\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 17 · Stack & Monotonic Stack/i }));

    const visual = await screen.findByRole('region', { name: '单调栈统一模板演示' });
    expect(within(visual).getByText('下标都在等待右侧第一个答案')).toBeInTheDocument();
    expect(within(visual).getByText('栈底 → 栈顶：单调不增')).toBeInTheDocument();

    fireEvent.change(within(visual).getByRole('slider', { name: '选择单调栈演示步骤' }), {
      target: { value: '6' },
    });

    expect(within(visual).getByText('2 > 1，弹出下标 1')).toBeInTheDocument();
    expect(within(visual).getByText('→ 2')).toBeInTheDocument();

    fireEvent.click(within(visual).getByRole('button', { name: '找右侧更小' }));
    expect(within(visual).getByText('栈底 → 栈顶：单调不减')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'English' }));
    const englishVisual = await screen.findByRole('region', {
      name: 'Unified monotonic-stack template walkthrough',
    });
    expect(englishVisual).toBeInTheDocument();
    expect(within(englishVisual).getByRole('button', { name: 'Next smaller' })).toBeInTheDocument();
  });

  it('renders the trapping rain water walkthrough and reaches six units', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills15')
          ? '# Two Pointers\n\n```rain-water-demo\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 15 · Two Pointers/i }));

    const visual = await screen.findByRole('region', { name: '接雨水双指针演示' });
    expect(within(visual).getByText(/较低的历史最高墙先结算/)).toBeInTheDocument();

    fireEvent.change(within(visual).getByRole('slider', { name: '选择接雨水演示步骤' }), {
      target: { value: '11' },
    });

    expect(visual.querySelector('.rain-water-total strong')).toHaveTextContent('6');
    expect(within(visual).getByText('重新播放')).toBeInTheDocument();
  });

  it('renders the backtracking decision tree, permutations, combination sum, dedup, and N-Queens visuals', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills20')
          ? [
            '# Backtracking',
            '',
            '```backtracking-patterns',
            '```',
            '',
            '```backtracking-tree-demo',
            '```',
            '',
            '```permutations-demo',
            '```',
            '',
            '```combination-sum-demo',
            '```',
            '',
            '```backtracking-dedup-demo',
            '```',
            '',
            '```n-queens-demo',
            '```',
          ].join('\n')
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 20 · Backtracking/i }));

    const atlas = await screen.findByRole('region', { name: '十道回溯题模板对照' });
    expect(within(atlas).getByText('backtrack(start)')).toBeInTheDocument();
    fireEvent.click(within(atlas).getByRole('tab', { name: /Permutations/ }));
    expect(within(atlas).getByText('排列型')).toBeInTheDocument();
    fireEvent.click(within(atlas).getByRole('tab', { name: /Generate Parentheses/ }));
    expect(within(atlas).getByText('约束构造型 / 前缀平衡')).toBeInTheDocument();

    const tree = screen.getByRole('region', { name: 'Subsets 决策树逐步演示' });
    fireEvent.change(within(tree).getByRole('slider', { name: '选择决策树演示步骤' }), {
      target: { value: '7' },
    });
    expect(within(tree).getByText('收答案：result 现在有 4 项')).toBeInTheDocument();
    expect(within(tree).getAllByText('[1, 2, 3]').length).toBeGreaterThan(0);
    expect(tree.querySelectorAll('.bt-results code')).toHaveLength(4);

    const perm = screen.getByRole('region', { name: 'Permutations 决策树逐步演示' });
    expect(perm.querySelectorAll('.pm-used-item')).toHaveLength(3);
    fireEvent.change(within(perm).getByRole('slider', { name: '选择全排列演示步骤' }), {
      target: { value: '7' },
    });
    expect(within(perm).getByText('叶子节点！收集全排列 [1, 2, 3]（第 1/6 个）')).toBeInTheDocument();

    const comb = screen.getByRole('region', { name: 'Combination Sum 决策树逐步演示' });
    expect(comb.querySelectorAll('.cs-budget-bar')).toHaveLength(1);
    fireEvent.change(within(comb).getByRole('slider', { name: '选择组合总和演示步骤' }), {
      target: { value: '7' },
    });
    expect(within(comb).getByText('命中目标！remain == 0，收答案 [2, 2, 3]（第 1/2 个）')).toBeInTheDocument();

    const dedup = screen.getByRole('region', { name: 'Subsets II 同层去重逐步演示' });
    fireEvent.change(within(dedup).getByRole('slider', { name: '选择去重演示步骤' }), {
      target: { value: '11' },
    });
    expect(dedup.querySelectorAll('.bt-node.cut')).toHaveLength(1);

    const queens = screen.getByRole('region', { name: '4 皇后回溯逐步演示' });
    fireEvent.change(within(queens).getByRole('slider', { name: '选择 N 皇后演示步骤' }), {
      target: { value: '34' },
    });
    expect(within(queens).getByText('[1, 3, 0, 2]')).toBeInTheDocument();
    expect(queens.querySelectorAll('.nq-cell.queen')).toHaveLength(1);

    fireEvent.click(screen.getByRole('button', { name: 'English' }));
    expect(await screen.findByRole('region', { name: 'Ten backtracking problems compared' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Step-through: the Permutations decision tree' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Step-through: the Combination Sum decision tree' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Step-through: backtracking on 4-Queens' })).toBeInTheDocument();
  });

  it('renders the greedy pattern atlas and interactive walkthroughs for jump game, gas station, and partition labels', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills10')
          ? [
            '# Greedy',
            '',
            '```kadane-demo',
            '```',
            '',
            '```greedy-patterns',
            '```',
            '',
            '```jump-game-demo',
            '```',
            '',
            '```gas-station-demo',
            '```',
            '',
            '```partition-labels-demo',
            '```',
          ].join('\n')
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 10 · Greedy Algorithms/i }));

    const kadane = await screen.findByRole('region', { name: 'Kadane 算法前缀动量与重置演示' });
    expect(within(kadane).getByText('Kadane 算法：正向利润累加与负前缀即时止损')).toBeInTheDocument();
    fireEvent.change(within(kadane).getByRole('slider', { name: '选择 Kadane 演示步骤' }), {
      target: { value: '6' },
    });
    expect(within(kadane).getByText(/达到全局峰值！更新 max_sum = 6/)).toBeInTheDocument();

    const atlas = screen.getByRole('region', { name: '八道贪心题全景对照' });
    expect(within(atlas).getByText('cur_sum = max(num, cur_sum + num)')).toBeInTheDocument();
    fireEvent.click(within(atlas).getByRole('tab', { name: /Jump Game II/ }));
    expect(within(atlas).getByText('隐式 BFS / 层次最远窗口贪心')).toBeInTheDocument();
    fireEvent.click(within(atlas).getByRole('tab', { name: /Gas Station/ }));
    expect(within(atlas).getByText('总净赤字校验 + 局部断点跳跃')).toBeInTheDocument();

    const jump = screen.getByRole('region', { name: '跳跃游戏贪心包络线演示' });
    expect(within(jump).getByText('Jump Game：维护 max_reach 消除回溯')).toBeInTheDocument();
    fireEvent.change(within(jump).getByRole('slider', { name: '选择跳跃游戏演示步骤' }), {
      target: { value: '1' },
    });
    expect(within(jump).getByText(/max_reach = max\(2, 1\+3\) = 4/)).toBeInTheDocument();

    const gas = screen.getByRole('region', { name: '加油站断点重置演示' });
    expect(within(gas).getByText('Gas Station：排除负累赘与候选点跳跃')).toBeInTheDocument();
    fireEvent.change(within(gas).getByRole('slider', { name: '选择加油站演示步骤' }), {
      target: { value: '3' },
    });
    expect(within(gas).getByText(/从起点 3 出发/)).toBeInTheDocument();

    const part = screen.getByRole('region', { name: '划分字母区间贪心切分演示' });
    expect(within(part).getByText(/Partition Labels/)).toBeInTheDocument();
    fireEvent.change(within(part).getByRole('slider', { name: '选择划分字母区间演示步骤' }), {
      target: { value: '2' },
    });
    expect(within(part).getByText(/\[9\]/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'English' }));
    expect(await screen.findByRole('region', { name: 'Kadane algorithm momentum and reset walkthrough' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Eight greedy problems compared' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Jump Game greedy envelope walkthrough' })).toBeInTheDocument();
  });

  it('opens the System Design section with the new overview notes', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'System Design' }));

    expect(await screen.findByRole('heading', { name: /System Design 0/i })).toBeInTheDocument();
    expect(screen.getByText('本板块共 12 篇笔记')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /System Design 00 · /i })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 01 · 无状态服务/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 01B · 虚拟化与容器/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 01C · Kubernetes/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 01D · Redis/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 02 · 数据库/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /System Design 03 · /i })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 04 · 存储/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /System Design 05 · /i })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 06 · 消息队列/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 07 · 图片分享与 Feed/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 08 · 异步 LLM RL/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 09 · 一致性哈希/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 10 · 秒杀/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 99 · 高频术语整合/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /System Design 02 · 数据库/i }));

    expect(await screen.findByRole('heading', { name: /数据库/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /System Design 04 · 存储/i }));

    expect(await screen.findByRole('heading', { name: /中文教程/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign04 Storage Systems.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 06 · 消息队列/i }));

    expect(await screen.findByRole('heading', { name: /消息队列/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign06 Async Messaging Systems.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 07 · 图片分享与 Feed/i }));

    expect(await screen.findByRole('heading', { name: /设计图片分享与 Home Feed/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign07 Photo Sharing Feed.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 08 · 异步 LLM RL/i }));

    expect(await screen.findByRole('heading', { name: /System Design 08 · 异步 LLM RL/ })).toBeInTheDocument();
    expect(screen.getByText(/Sample Admission QPS ~60 \/s/)).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign08 LLM Async RL Platform.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 09 · 一致性哈希/i }));

    expect(await screen.findByRole('heading', { name: /System Design 09 · 一致性哈希/ })).toBeInTheDocument();
    expect(screen.getByText(/节点变化时只迁移相邻区间/)).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign09 Consistent Hashing.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 10 · 秒杀/i }));

    expect(await screen.findByRole('heading', { name: /秒杀/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign10 Flash Sale.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 99 · 高频术语整合/i }));

    expect(await screen.findByRole('heading', { name: /高频术语整合/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign99 Glossary.md')).toHaveLength(2);
  });

  it('opens the business algorithms system map and switches architecture paths', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: '业务算法' }));

    expect(await screen.findByText('本板块共 20 篇笔记')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /第 2 章 · 数据、样本与特征流/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /第 15 章 · 在线实验与涨指标/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /第 18 章 · LLM 排序与生成式推荐/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /第 19 章 · RAG 与 Agentic Search/ })).toBeInTheDocument();
    expect(decodeURIComponent(window.location.hash)).toBe('#Business Algorithm TODO.md');
    const visual = await screen.findByRole('region', { name: '推荐与搜索业务算法系统地图' });
    expect(within(visual).getByText(/亿级候选，沿延迟预算逐层收窄/)).toBeInTheDocument();
    expect(within(visual).getByText('10⁸ → 3k')).toBeInTheDocument();

    fireEvent.click(within(visual).getByRole('button', { name: /端到端生成/ }));

    expect(within(visual).getByText(/把检索与排序目标并入一次序列生成/)).toBeInTheDocument();
    fireEvent.click(within(visual).getByRole('button', { name: /统一生成器/ }));
    expect(within(visual).getByText(/"端到端"范围因系统而异/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'English' }));

    const englishVisual = await screen.findByRole('region', { name: 'Recommendation and search algorithm system map' });
    expect(within(englishVisual).getByText('Traditional cascade')).toBeInTheDocument();
    expect(within(englishVisual).getByText(/Narrow hundreds of millions of candidates/)).toBeInTheDocument();
    expect(within(englishVisual).getAllByText('Multi-channel retrieval')).toHaveLength(2);
    expect(within(englishVisual).getByText('Exposure and interaction logs')).toBeInTheDocument();
    expect(within(englishVisual).queryByText('传统级联')).not.toBeInTheDocument();
  });

  it('renders quick coding problems inline instead of linking to a standalone appendix', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = decodeURIComponent(String(input));
      return {
        ok: true,
        text: async () => requestUrl.includes('BusinessAlgorithm02 Ranking')
          ? [
            '# 排序目标与离线评价',
            '',
            '### Quick Coding：NDCG@K',
            '',
            '实现 `ndcg_at_k(relevances, k)`。',
            '',
            '<details>',
            '<summary>参考答案</summary>',
            '',
            '```python',
            'def ndcg_at_k(relevances, k):',
            '    return 0.0',
            '```',
            '',
            '</details>',
          ].join('\n')
          : '# 业务算法',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: '业务算法' }));
    expect(screen.queryByRole('button', { name: /附录 · Quick Coding/ })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /第 9 章 · 排序目标与离线评价/ }));

    expect(await screen.findByRole('heading', { name: /Quick Coding：NDCG@K/ })).toBeInTheDocument();
    expect(screen.getByText('参考答案')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /QC05 NDCG@K/ })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'English' }));
    expect(screen.getByRole('button', {
      name: /Chapter 9 · Ranking Objectives and Offline Evaluation/,
    })).toBeInTheDocument();
    expect(screen.getByRole('heading', {
      name: /Chapter 9 · Ranking Objectives and Offline Evaluation/,
    })).toBeInTheDocument();
  });

  it('renders the message queue anatomy and redelivery walkthrough', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('SystemDesign06')
          ? '# 异步消息系统\n\n```message-queue-demo\n```'
          : '# System Design tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'System Design' }));
    fireEvent.click(screen.getByRole('button', { name: /System Design 06 · 消息队列/i }));

    const visual = await screen.findByRole('region', { name: '消息队列数据与投递生命周期演示' });
    expect(within(visual).getByText('Producer 构造应用消息')).toBeInTheDocument();
    expect(within(visual).getByText('消息还在 producer 内存中，broker 尚未接管')).toBeInTheDocument();

    fireEvent.change(within(visual).getByRole('slider', { name: '选择消息队列生命周期步骤' }), {
      target: { value: '5' },
    });

    expect(within(visual).getByText('Worker B 收到重投')).toBeInTheDocument();
    expect(within(visual).getAllByText(/rh_B2/).length).toBeGreaterThan(0);

    fireEvent.click(within(visual).getByRole('button', { name: '下一步' }));
    expect(within(visual).getByText('业务提交成功，再发送 ack')).toBeInTheDocument();
    expect(within(visual).getAllByText(/rh_B2 已确认/).length).toBeGreaterThan(0);
  });

  it('compares VM and container isolation boundaries', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('SystemDesign01B')
          ? '# 虚拟化与容器\n\n```virtualization-container-visual\n```'
          : '# System Design tutorial',
      };
    });

    render(<App />);
    fireEvent.click(screen.getByRole('button', { name: 'System Design' }));
    fireEvent.click(screen.getByRole('button', { name: /System Design 01B · 虚拟化与容器/i }));

    const visual = await screen.findByRole('region', { name: '虚拟机与容器隔离边界对比' });
    expect(within(visual).getByText('VM：每个 guest 有自己的 kernel')).toBeInTheDocument();
    expect(within(visual).getAllByText('Guest kernel')).toHaveLength(2);

    fireEvent.click(within(visual).getByRole('button', { name: 'Container' }));
    expect(within(visual).getByText('Container：多个进程共享 host kernel')).toBeInTheDocument();
    expect(within(visual).getByText('Shared host kernel')).toBeInTheDocument();
    expect(within(visual).getByText('隔离进程视图和资源，kernel 仍然共享。')).toBeInTheDocument();
  });

  it('animates Kubernetes object hierarchy and gang scheduling', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('SystemDesign01C')
          ? '# Kubernetes\n\n```k8s-hierarchy-visual\n```\n\n```k8s-gang-visual\n```\n\n```k8s-layered-arch-visual\n```'
          : '# System Design tutorial',
      };
    });

    render(<App />);
    fireEvent.click(screen.getByRole('button', { name: 'System Design' }));
    fireEvent.click(await screen.findByRole('button', { name: /System Design 01C · Kubernetes/i }));

    const hierarchy = await screen.findByRole('region', { name: 'Kubernetes 对象层级从容器到 Namespace' });
    expect(within(hierarchy).getByText('从容器一层层包到 Namespace')).toBeInTheDocument();
    expect(within(hierarchy).getByText(/中间永远是同一个进程/)).toBeInTheDocument();
    expect(within(hierarchy).getByText('train.py')).toBeInTheDocument();
    expect(within(hierarchy).getAllByText(/^Pod$/).length).toBeGreaterThan(0);
    expect(within(hierarchy).getAllByText(/^Namespace$/).length).toBeGreaterThan(0);

    fireEvent.click(within(hierarchy).getByRole('tab', { name: /Namespace/i }));
    expect(within(hierarchy).getByText('虚拟集群边界')).toBeInTheDocument();
    expect(within(hierarchy).getByText('多租户的第一刀，不是最后一道墙。')).toBeInTheDocument();

    const gang = screen.getByRole('region', { name: '逐 Pod 调度与 Gang Scheduling 对比' });
    expect(within(gang).getByText('默认调度：先占到的人先跑')).toBeInTheDocument();
    expect(within(gang).getByText('waiting forever')).toBeInTheDocument();

    fireEvent.click(within(gang).getByRole('button', { name: 'Gang' }));
    expect(within(gang).getByText('Gang：凑齐 4 张 GPU 才启动')).toBeInTheDocument();
    expect(within(gang).getAllByText('Queued')).toHaveLength(4);

    const layered = screen.getByRole('region', { name: '职责分层与拓扑分层' });
    expect(within(layered).getByText('职责分层：每层允许做什么')).toBeInTheDocument();
    fireEvent.click(within(layered).getByRole('tab', { name: /数据层/ }));
    expect(within(layered).getByText(/不处理用户 HTTP 业务规则/)).toBeInTheDocument();
    fireEvent.click(within(layered).getByRole('button', { name: '拓扑分层' }));
    expect(within(layered).getByText('拓扑分层：谁和谁一起死')).toBeInTheDocument();
    fireEvent.click(within(layered).getByRole('tab', { name: 'AZ / zone' }));
    expect(within(layered).getByText(/无状态副本 topologySpread/)).toBeInTheDocument();
  });

  it('redirects System Design 00 Overview to the first component note', async () => {
    window.history.replaceState(null, '', '/#SystemDesign00%20Overview.md');

    render(<App />);

    expect(await screen.findByRole('heading', { name: /无状态服务/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign01 Stateless Service.md')).toHaveLength(2);

    await waitFor(() => {
      expect(window.location.hash).toBe('#SystemDesign01%20Stateless%20Service.md');
    });
  });

  it('redirects renamed System Design note routes to the new chapter numbers', async () => {
    window.history.replaceState(null, '', '/#SystemDesign07%20Async%20Messaging%20Systems.md');

    render(<App />);

    expect(await screen.findByRole('heading', { name: /消息队列/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign06 Async Messaging Systems.md')).toHaveLength(2);

    await waitFor(() => {
      expect(window.location.hash).toBe('#SystemDesign06%20Async%20Messaging%20Systems.md');
    });
  });

  it('redirects legacy LeetCode note routes to the contiguous chapter numbers', async () => {
    window.history.replaceState(null, '', '/#CoreSkills33%20Backtracking.md');

    render(<App />);

    expect(await screen.findByRole('heading', { name: /Core Skills 20 · Backtracking/i })).toBeInTheDocument();
    expect(screen.getAllByText('CoreSkills20 Backtracking.md')).toHaveLength(2);

    await waitFor(() => {
      expect(window.location.hash).toBe('#CoreSkills20%20Backtracking.md');
    });
  });

  it('renders native HTML architecture diagrams for the System Design notes', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      let content = '# System Design tutorial';

      if (requestUrl.includes('SystemDesign01 Stateless') || requestUrl.includes('SystemDesign01%20Stateless')) {
        content = '# Overview\n\n```system-design-overview-visual\n```';
      } else if (requestUrl.includes('SystemDesign06')) {
        content = '# Async Messaging\n\n```async-messaging-architecture-visual\n```';
      } else if (requestUrl.includes('SystemDesign07')) {
        content = '# Photo Sharing\n\n```photo-sharing-architecture-visual\n```';
      } else if (requestUrl.includes('SystemDesign10')) {
        content = '# Flash Sale\n\n```flash-sale-architecture-visual\n```';
      }

      return { ok: true, text: async () => content };
    });

    render(<App />);
    fireEvent.click(screen.getByRole('button', { name: 'System Design' }));

    const overview = await screen.findByRole('region', { name: '系统设计基础架构图' });
    expect(within(overview).getByText('先跑通同步闭环，再按指标加组件')).toBeInTheDocument();
    fireEvent.click(within(overview).getByRole('button', { name: /Primary Store/i }));
    expect(within(overview).getByText('先明确 source of truth')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /System Design 07 · 图片分享与 Feed/i }));
    const photo = await screen.findByRole('region', { name: '图片分享系统架构图' });
    fireEvent.click(within(photo).getByRole('button', { name: '读取 Feed' }));
    expect(within(photo).getByText('先取 post_id，再批量补齐内容')).toBeInTheDocument();
    expect(within(photo).getByText('读取时校验')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /System Design 06 · 消息队列/i }));
    const asyncDiagram = await screen.findByRole('region', { name: '异步消息模式架构图' });
    fireEvent.click(within(asyncDiagram).getByRole('button', { name: 'Kafka groups' }));
    expect(within(asyncDiagram).getByText('系统是实现，group 决定语义')).toBeInTheDocument();
    expect(within(asyncDiagram).getByText('group: analytics')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /System Design 10 · 秒杀/i }));
    const flash = await screen.findByRole('region', { name: '秒杀系统架构图' });
    expect(within(flash).getByText(/202 不预占库存/)).toBeInTheDocument();
    fireEvent.click(within(flash).getByRole('button', { name: '查看活动' }));
    expect(within(flash).getByText('Sale Service')).toBeInTheDocument();
    fireEvent.click(within(flash).getByRole('button', { name: '查结果' }));
    expect(within(flash).getByText(/Redis 是读模型/)).toBeInTheDocument();
  });

  it('shows local-only draft notes in development mode', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: '草稿区' }));

    expect(await screen.findByRole('heading', { name: '草稿区' })).toBeInTheDocument();
    expect(screen.getByText('本板块共 2 篇笔记')).toBeInTheDocument();
    expect(screen.getAllByText(/LLM八股 Overview · JD 高频主题拆解/).length).toBeGreaterThan(0);
    expect(screen.getByRole('button', { name: /Quant 草稿 · 概率基础公式与记忆框架/i })).toBeInTheDocument();
    expect(await screen.findByRole('heading', { name: /Motivation/ })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /7\. RLVR & Agentic RL/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Quant 草稿 · 概率基础公式与记忆框架/i }));
    expect(await screen.findByRole('heading', { name: /先按题型选工具/ })).toBeInTheDocument();
    expect(
      Array.from(document.querySelectorAll('annotation[encoding="application/x-tex"]')).some((node) =>
        node.textContent?.includes('\\mathbb{E}[X]'),
      ),
    ).toBe(true);

    expect(screen.queryByRole('button', { name: /System Design 草稿 · 数据库扩展三件套/i })).not.toBeInTheDocument();
  });

  it('renders the high-dimensional integral visual and changes dimension', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('Quant06')
          ? '# 高维积分\n\n```high-dimensional-integral-demo\n```'
          : '# Quant tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'Quant' }));
    fireEvent.click(screen.getByRole('button', { name: /Quant 05 · 高维积分/i }));

    expect(await screen.findByRole('region', { name: '高维积分动态三维可视化' })).toBeInTheDocument();
    expect(screen.getByText('积分 = 曲面的平均高度')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'n → ∞ 云团' }));
    expect(screen.getByText('n = 2')).toBeInTheDocument();

    fireEvent.change(screen.getByRole('slider', { name: '选择积分维度' }), { target: { value: '4' } });
    expect(screen.getByText('n = 32')).toBeInTheDocument();
    expect(screen.getByText('目标：2/3 ≈ 0.6667')).toBeInTheDocument();
  });

  it('steps through the prefix-minimum grouping visual', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('Quant01')
          ? '# 期望与计数\n\n```record-minimum-demo\n```'
          : '# Quant tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'Quant' }));
    fireEvent.click(screen.getByRole('button', { name: /Quant 01 · 期望/i }));

    expect(await screen.findByRole('region', { name: '前缀最小值与最终队伍可视化' })).toBeInTheDocument();
    expect(screen.getByText('前缀最小值')).toBeInTheDocument();
    expect(screen.queryByText(/^Prefix minimum$/i)).not.toBeInTheDocument();
    expect(screen.getByText('位置 3 / 7')).toBeInTheDocument();
    expect(screen.getByText('v3 = 6 > 4，最终会追上第 2 位领队。')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '下一步 →' }));

    expect(screen.getByText('位置 4 / 7')).toBeInTheDocument();
    expect(screen.getByText('v4 = 2 < 4，刷新前缀最小值，成为新领队。')).toBeInTheDocument();
    expect(screen.getByText('3 支队伍')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'English' }));

    expect(await screen.findByRole('region', { name: 'Prefix minimum and final groups visualization' })).toBeInTheDocument();
    expect(screen.getByText('Prefix minimum')).toBeInTheDocument();
    expect(screen.getByText('Position 3 / 7')).toBeInTheDocument();
    expect(screen.getByText('v3 = 6 > 4, so it eventually catches leader 2.')).toBeInTheDocument();
    expect(screen.queryByText('行进方向')).not.toBeInTheDocument();
  });

  it('opens the recursion chapter for the absent-minded passenger problem', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('Quant01')
          ? '# Quant 01 · 期望、计数与递推\n\n## 健忘乘客\n\n答案是 $1/2$。'
          : '# Quant tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'Quant' }));

    expect(screen.getByText(/本板块共 \d+ 篇笔记/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Quant 01 · 期望/i }));

    expect(await screen.findByRole('heading', { name: /健忘乘客/i })).toBeInTheDocument();
    expect(screen.getAllByText('Quant01 Expectation Counting Multinomial.md')).toHaveLength(2);
  });

  it('opens Effective Modern C++ 7 for C++17 and C++20 core features', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('QuantDevEMC07')
          ? '# Effective Modern C++ 7 · C++17 与 C++20 核心新特性深度解构\n\nif constexpr 与 Concepts。'
          : '# Quant tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'Quant' }));
    fireEvent.click(screen.getByRole('button', { name: /Effective Modern C\+\+ 7/i }));

    expect(await screen.findByRole('heading', { name: /Effective Modern C\+\+ 7 · C\+\+17 与 C\+\+20 核心新特性深度解构/i })).toBeInTheDocument();
    expect(screen.getByText('if constexpr 与 Concepts。')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'English' }));
    expect(screen.getByRole('button', { name: /Effective Modern C\+\+ 7 · C\+\+17 & C\+\+20 Core Modern Features/i })).toBeInTheDocument();
  });

  it('opens Quant 11 for Martingales, Wald Equations, and Optimal Stopping and renders interactive visualizer', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('Quant11')
          ? '# Quant 07 · 鞅、停时与下注\n\n```martingale-rw-demo\n```\n\nWald 一阶与二阶等式。'
          : '# Quant tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'Quant' }));
    fireEvent.click(screen.getByRole('button', { name: /Quant 07 · 鞅/i }));

    expect(await screen.findByRole('heading', { name: /Quant 07 · 鞅、停时与下注/i })).toBeInTheDocument();
    expect(await screen.findByText('Wald 等式、1D 随机游走与最优决策')).toBeInTheDocument();
    expect(await screen.findByText('胜率 P(到达 +a)')).toBeInTheDocument();
    expect(screen.getByText('期望停止时间 E[T]')).toBeInTheDocument();

    // Switch to Secretary Problem tab
    fireEvent.click(screen.getByRole('tab', { name: /秘书问题 37% 法则/i }));
    expect(await screen.findByText(/候选人总人数 n/i)).toBeInTheDocument();
    expect(screen.getByText(/1\/e ≈ 36\.79%/)).toBeInTheDocument();

    // Switch to Pattern Waiting & Li's Martingale tab
    fireEvent.click(screen.getByRole('tab', { name: /模式等待与赌场鞅/i }));
    expect(await screen.findByText(/HTTH vs HTHT/)).toBeInTheDocument();
    expect(screen.getByText(/E\[T_A\] = 18/)).toBeInTheDocument();
    expect(screen.getByText(/E\[T_B\] = 20/)).toBeInTheDocument();
  });

  it('opens Quant 12 Brownian motion and stochastic calculus note and renders interactive components', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = decodeURIComponent(String(input));
      return {
        ok: true,
        text: async () => requestUrl.includes('Quant12')
          ? '# Quant 08 · 布朗运动、伊藤公式与测度变换\n\n```brownian-motion-demo\n```\n\n```two-d-walk-demo\n```\n\n```ito-geometry-demo\n```\n\n```reflection-principle-demo\n```\n\n```delta-hedging-demo\n```'
          : '# Quant tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'Quant' }));
    fireEvent.click(screen.getByRole('button', { name: /Quant 08 · 布朗运动/i }));

    expect(await screen.findByRole('heading', { name: /Quant 08 · 布朗运动、伊藤公式与测度变换/i })).toBeInTheDocument();
    expect(await screen.findByLabelText('布朗运动轨道与二次变差演示')).toBeInTheDocument();
    expect(screen.getByLabelText('2D 随机游走与布朗运动极限演示')).toBeInTheDocument();
    expect(screen.getByLabelText('伊藤几何与斯特拉托诺维奇积分对比演示')).toBeInTheDocument();
    expect(screen.getByLabelText('停时与反射原理演示')).toBeInTheDocument();
    expect(screen.getByLabelText('期权 Delta 对冲与 Gamma 损益演示')).toBeInTheDocument();
  });

  it('opens Quant 13 Game Theory note and renders the game theory simulator', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = decodeURIComponent(String(input));
      return {
        ok: true,
        text: async () => requestUrl.includes('Quant13')
          ? '# Quant 09 · 博弈论\n\n```game-theory-interactive-demo\n```'
          : '# Quant tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'Quant' }));
    fireEvent.click(screen.getByRole('button', { name: /Quant 09 · 博弈论/i }));

    expect(await screen.findByRole('heading', { name: /Quant 09 · 博弈论/i })).toBeInTheDocument();
    expect(await screen.findByLabelText('博弈论与策略性决策演示')).toBeInTheDocument();
    expect(screen.getByText(/博弈论模拟器/)).toBeInTheDocument();

    // Switch tab to Truel
    fireEvent.click(screen.getByRole('button', { name: '三方决斗' }));
    expect(screen.getByText(/枪手 A 最终胜率/)).toBeInTheDocument();

    // Switch tab to Auctions
    fireEvent.click(screen.getByRole('button', { name: '拍卖与胜者诅咒' }));
    expect(screen.getByText(/一阶密封出价 FPA/)).toBeInTheDocument();
  });

  it('opens Quant 14 Financial Markets and Derivatives note and verifies content and language toggle', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = decodeURIComponent(String(input));
      return {
        ok: true,
        text: async () => {
          if (requestUrl.includes('Quant14') && requestUrl.endsWith('.en.md')) {
            return '# Quant 10 · Markets, assets, and portfolios\n\nEnglish content for Quant 10.';
          }
          if (requestUrl.includes('Quant14')) {
            return '# Quant 10 · 市场、资产与组合\n\n中文内容 for Quant 10.';
          }
          return '# Default note';
        },
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'Quant' }));
    fireEvent.click(screen.getByRole('button', { name: /Quant 10 · 市场/i }));

    expect(await screen.findByRole('heading', { name: /Quant 10 · 市场、资产与组合/i })).toBeInTheDocument();

    // Toggle language to English
    fireEvent.click(screen.getByRole('button', { name: 'English' }));
    expect(await screen.findByRole('heading', { name: /Quant 10 · Markets, assets, and portfolios/i })).toBeInTheDocument();
  });

  it('redirects absorbed Quant notes to the surviving chapter', async () => {
    window.history.replaceState(null, '', '/#Quant15%20Equivalent%20Martingale%20Measure%20Girsanov%20and%20FTAP.md');

    render(<App />);

    expect(await screen.findByRole('heading', { name: /布朗运动/ })).toBeInTheDocument();
    expect(screen.getAllByText('Quant12 Brownian Motion Ito Calculus Stopping Times and Options.md')).toHaveLength(2);

    await waitFor(() => {
      expect(window.location.hash).toBe('#Quant12%20Brownian%20Motion%20Ito%20Calculus%20Stopping%20Times%20and%20Options.md');
    });
  });

  it('opens Quant 11 Linear Regression note and renders FWL interactive geometry visualizer', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = decodeURIComponent(String(input));
      return {
        ok: true,
        text: async () => {
          if (requestUrl.includes('Quant16') && requestUrl.endsWith('.en.md')) {
            return '# Quant 11 · Linear regression and kernel smoothing\n\n```fwl-geometry-demo\n```\n\n```anova-variance-demo\n```\n\n```nadaraya-watson-demo\n```\n\n```local-linear-carpentry-demo\n```';
          }
          if (requestUrl.includes('Quant16')) {
            return '# Quant 11 · 线性回归与核平滑\n\n```fwl-geometry-demo\n```\n\n```anova-variance-demo\n```\n\n```nadaraya-watson-demo\n```\n\n```local-linear-carpentry-demo\n```';
          }
          return '# Default note';
        },
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'Quant' }));
    fireEvent.click(screen.getByRole('button', { name: /Quant 11 · 线性回归/i }));

    expect(await screen.findByRole('heading', { name: /Quant 11 · 线性回归与核平滑/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/FWL 定理几何投影与两阶段残差回归演示/i)).toBeInTheDocument();
    expect(screen.getByText(/Frisch–Waugh–Lovell \(FWL\) 几何投影/i)).toBeInTheDocument();

    // Verify ANOVA demo rendering
    expect(screen.getByLabelText(/ANOVA 方差分解与 R² 几何投影演示/i)).toBeInTheDocument();
    expect(screen.getByText(/方差分解 \(TSS = ESS \+ RSS\) 与判定系数 R² = cos²\(θ\)/i)).toBeInTheDocument();

    // Switch ANOVA tab to 2D scatter
    fireEvent.click(screen.getByRole('button', { name: /2D 样本散点与平方和分解/i }));
    expect(screen.getByText(/点击任意样本点观察该点/i)).toBeInTheDocument();

    // Verify Nadaraya-Watson demo rendering
    expect(screen.getByText(/Nadaraya–Watson 核回归：连续距离权重的“软 KNN”/i)).toBeInTheDocument();
    expect(screen.getByText(/有效样本容量 N_eff/i)).toBeInTheDocument();

    // Switch Nadaraya-Watson tab to Comparison
    fireEvent.click(screen.getByRole('button', { name: /平滑 vs 阶梯对比/i }));
    expect(screen.getByText(/硬截断 3-KNN 阶梯/i)).toBeInTheDocument();

    // Switch Kernel to Epanechnikov
    fireEvent.click(screen.getByRole('button', { name: /埃帕内奇尼科夫核 \(Epanechnikov\)/i }));

    // Verify Local Linear Carpentry demo rendering
    expect(screen.getByLabelText(/局部线性回归与自动核修缮交互实验室/i)).toBeInTheDocument();
    expect(screen.getByText(/局部线性回归与“自动核修缮”：消除边界偏差与自由度调控/i)).toBeInTheDocument();
    expect(screen.getAllByText(/有效自由度 df = tr\(S\)/i).length).toBeGreaterThanOrEqual(1);

    // Switch Carpentry tab to df & LOOCV
    fireEvent.click(screen.getByRole('button', { name: /有效自由度 tr\(S\) 与 LOOCV/i }));
    expect(screen.getByText(/样本杠杆值与自我影响力 S_ii/i)).toBeInTheDocument();

    // Verify interaction with FWL step button
    fireEvent.click(screen.getByRole('button', { name: /步骤 2: X₂ 正交化得 X̃₂/i }));
    expect(screen.getByText(/FWL 核心正交化/i)).toBeInTheDocument();

    // Toggle language to English
    fireEvent.click(screen.getByRole('button', { name: 'English' }));
    expect(await screen.findByRole('heading', { name: /Quant 11 · Linear regression/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/FWL Theorem Geometry & Two-Stage Regression Demo/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/ANOVA Variance Decomposition & R² Geometry Demo/i)).toBeInTheDocument();
    expect(screen.getByText(/Nadaraya–Watson Kernel: Continuous Soft Distance-Weighted KNN/i)).toBeInTheDocument();
    expect(screen.getByText(/Local Linear Regression & Kernel Carpentry: Boundary Bias & Effective df/i)).toBeInTheDocument();
  });

  it('renders the Palindromic Substrings 2D DP matrix visual walkthrough and steps through states', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills09')
          ? '# Dynamic Programming\n\n```palindrome-dp-demo\n```'
          : '# Default note',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 9 ·/i }));

    const pdp = await screen.findByRole('region', { name: '回文子串 2D DP 状态转移演示' });
    expect(within(pdp).getByText(/Palindromic Substrings：二维 DP 状态表填表与西南角依赖可视化/i)).toBeInTheDocument();

    // Verify presence of preset buttons and table
    expect(within(pdp).getByRole('button', { name: /s = "ababa"/i })).toBeInTheDocument();
    expect(within(pdp).getByRole('button', { name: /s = "babad"/i })).toBeInTheDocument();

    // Step forward
    fireEvent.click(within(pdp).getByRole('button', { name: /下一步/i }));

    // Verify step index increment and state change
    expect(within(pdp).getByText(/单个字符自身必为回文/i)).toBeInTheDocument();

    // Switch preset to "babad"
    fireEvent.click(within(pdp).getByRole('button', { name: /s = "babad"/i }));
    expect(within(pdp).getByText(/初始化 2D DP 表/i)).toBeInTheDocument();
  });

  it('renders the Coin Change Complete Knapsack visual walkthrough and steps through states', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills09')
          ? '# Dynamic Programming\n\n```coin-change-demo\n```'
          : '# Default note',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 9 ·/i }));

    const ccv = await screen.findByRole('region', { name: '零钱兑换 DP 状态转移演示' });
    expect(within(ccv).getByText(/Coin Change：零钱兑换最少枚数状态转移与回溯可视化/i)).toBeInTheDocument();

    // Verify presence of preset chips
    expect(within(ccv).getByRole('button', { name: /coins = \[1, 2, 5\]/i })).toBeInTheDocument();
    expect(within(ccv).getByRole('button', { name: /coins = \[1, 3, 4\]/i })).toBeInTheDocument();

    // Step forward via slider
    const slider = within(ccv).getByRole('slider', { name: '选择零钱兑换 DP 演示步骤' });
    fireEvent.change(slider, { target: { value: '1' } });

    // Verify step description update
    expect(within(ccv).getByText(/发现更优解/i)).toBeInTheDocument();

    // Switch view mode to 2D Grid
    fireEvent.click(within(ccv).getByRole('button', { name: /二维完全背包决策表格/i }));
    expect(within(ccv).getByText(/二维完全背包决策表格 dp\[coin_idx\]\[amount\]/i)).toBeInTheDocument();

    // Switch preset to greedy trap [1, 3, 4]
    fireEvent.click(within(ccv).getByRole('button', { name: /coins = \[1, 3, 4\]/i }));
    expect(within(ccv).getByText(/初始化状态/i)).toBeInTheDocument();
  });

  it('renders the Partition Equal Subset Sum 0/1 Knapsack visual and switches modes', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills09')
          ? '# Dynamic Programming\n\n```subset-sum-demo\n```'
          : '# Default note',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 9 ·/i }));

    const pssv = await screen.findByRole('region', { name: '分割等和子集 0/1 背包状态转移演示' });
    expect(within(pssv).getByText(/Partition Equal Subset Sum：分割等和子集可达性与决策分支可视化/i)).toBeInTheDocument();

    // Verify preset buttons exist
    expect(within(pssv).getByRole('button', { name: /nums = \[1, 5, 11, 5\]/i })).toBeInTheDocument();
    expect(within(pssv).getByRole('button', { name: /nums = \[1, 2, 3, 5\]/i })).toBeInTheDocument();

    // Step forward
    fireEvent.click(within(pssv).getByRole('button', { name: /下一步/i }));

    // Verify step description
    expect(within(pssv).getByText(/当前物品 nums\[0\] = 1/i)).toBeInTheDocument();

    // Switch view modes
    fireEvent.click(within(pssv).getByRole('button', { name: /2D 状态表格与转移依赖/i }));
    expect(within(pssv).getByText(/2D 状态表格演化/i)).toBeInTheDocument();

    fireEvent.click(within(pssv).getByRole('button', { name: /可达和集合与二叉决策树/i }));
    expect(within(pssv).getByText(/可达和集合演化/i)).toBeInTheDocument();

    // Switch preset to Odd Sum
    fireEvent.click(within(pssv).getByRole('button', { name: /nums = \[1, 2, 3, 5\]/i }));
    expect(within(pssv).getByText(/总和 total = 11 为奇数/i)).toBeInTheDocument();
  });

  it('opens the ML Coding section and displays ML Coding 00 note', async () => {
    render(<App />);

    const nav = screen.getByRole('navigation', { name: '主导航' });
    fireEvent.click(within(nav).getByRole('button', { name: /ML Coding/i }));
    expect(screen.getByRole('button', { name: /ML Coding 00 ·/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /ML Coding 00 ·/i }));
    expect(await screen.findByRole('heading', { name: /ML Coding 00 · ML 基础：数据预处理、数据泄露与经典损失函数/i })).toBeInTheDocument();

    // Switch to English and verify translation
    fireEvent.click(screen.getByRole('button', { name: 'English' }));
    expect(await screen.findByRole('heading', { name: /ML Coding 00 · ML Basics: Data Preprocessing, Data Leakage & Loss Functions/i })).toBeInTheDocument();
  });

  it('routes directly to ML Coding 00 via URL hash', async () => {
    window.location.hash = '#MLCoding00%20ML%20Basics%20Data%20Preprocessing%20Loss%20Functions.md';
    render(<App />);

    expect(await screen.findByRole('heading', { name: /ML Coding 00 · ML 基础：数据预处理、数据泄露与经典损失函数/i })).toBeInTheDocument();
  });

  it('renders the interactive ML metrics lab in ML Coding 00 note and switches tabs', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = decodeURIComponent(String(input));
      return {
        ok: true,
        text: async () => {
          if (requestUrl.includes('MLCoding00') && requestUrl.endsWith('.en.md')) {
            return '# ML Coding 00 · ML Basics: Data Preprocessing, Data Leakage & Loss Functions\n\n```ml-metrics-demo\n```';
          }
          if (requestUrl.includes('MLCoding00')) {
            return '# ML Coding 00 · ML 基础：数据预处理、数据泄露与经典损失函数\n\n```ml-metrics-demo\n```';
          }
          return '# Default note';
        },
      };
    });

    window.location.hash = '#MLCoding00%20ML%20Basics%20Data%20Preprocessing%20Loss%20Functions.md';
    render(<App />);

    expect(await screen.findByRole('heading', { name: /混淆矩阵、ROC\/PR 双曲线与校准业务代价全景/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /双曲线与混淆矩阵/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /概率校准 \(ECE \/ Brier\)/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Top-k 与业务损失曲面/i })).toBeInTheDocument();

    // Verify presence of confusion matrix & metrics
    expect(screen.getByText(/TP \(真正例\)/i)).toBeInTheDocument();
    expect(screen.getAllByText(/ROC-AUC/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/PR-AUC \(AP\)/i).length).toBeGreaterThan(0);

    // Switch to Calibration tab
    fireEvent.click(screen.getByRole('button', { name: /概率校准 \(ECE \/ Brier\)/i }));
    expect(screen.getByText(/校准温度系数/i)).toBeInTheDocument();
    expect(screen.getByText(/Reliability Diagram/i)).toBeInTheDocument();
    expect(screen.getByText(/ECE =/i)).toBeInTheDocument();

    // Switch to Business Cost tab
    fireEvent.click(screen.getByRole('button', { name: /Top-k 与业务损失曲面/i }));
    expect(screen.getByText(/Top-k 审核容量/i)).toBeInTheDocument();
    expect(screen.getByText(/当前预期损失/i)).toBeInTheDocument();
    expect(screen.getByText(/全局最优 k\*/i)).toBeInTheDocument();

    // Toggle language to English
    fireEvent.click(screen.getByRole('button', { name: 'English' }));
    expect(await screen.findByRole('heading', { name: /Confusion Matrix, Dual ROC\/PR Curves & Cost Frontier/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Top-k & Business Cost/i })).toBeInTheDocument();
  });

  it('routes directly to ML Coding 00B and ML Coding 01B via URL hash and renders them properly', async () => {
    // 1. Test MLCoding00B
    window.location.hash = '#MLCoding00B%20LLM%20Basics%20Decoder%20Only%20Precision%20Alignment%20Distillation.md';
    const { unmount } = render(<App />);

    expect(await screen.findByRole('heading', { name: /ML Coding 00B · LLM 基础：Decoder-Only 架构胜出原理解析/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'English' }));
    expect(await screen.findByRole('heading', { name: /ML Coding 00B · LLM Basics: Why Decoder-Only Won/i })).toBeInTheDocument();

    unmount();

    // 2. Test MLCoding01B
    window.location.hash = '#MLCoding01B%20Transformer%20Architecture%20Variants%20Attention%20FLOPs%20KV%20Cache.md';
    render(<App />);

    expect(await screen.findByRole('heading', { name: /ML Coding 01B · Transformer 架构变体：MHA\/MQA\/GQA/i })).toBeInTheDocument();
  });

  it('renders the Anisotropy Cone visual and switches modes', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('MLCoding00B')
          ? '# LLM Basics\n\n```anisotropy-cone-demo\n```'
          : '# Default note',
      };
    });

    window.location.hash = '#MLCoding00B%20LLM%20Basics%20Decoder%20Only%20Precision%20Alignment%20Distillation.md';
    render(<App />);

    const visual = await screen.findByRole('region', { name: /各向异性圆锥效应交互实验室/i });
    expect(visual).toBeInTheDocument();
    expect(within(visual).getByText(/各向异性圆锥效应 vs 各向同性超球面交互实验室/i)).toBeInTheDocument();

    // Verify initial Anisotropic mode
    expect(within(visual).getByText(/圆锥坍塌态/i)).toBeInTheDocument();

    // Switch to Isotropic mode
    fireEvent.click(within(visual).getByRole('button', { name: /2\. 对比学习后/i }));
    expect(within(visual).getByText(/均匀各向同性/i)).toBeInTheDocument();

    // Switch to Centering baseline mode
    fireEvent.click(within(visual).getByRole('button', { name: /3\. 去均值白化基线/i }));
    expect(within(visual).getByText(/去中心化过渡态/i)).toBeInTheDocument();
  });

  it('routes directly to ML Coding 06B (RLHF) and ML Coding 06C (RLVR / GRPO) via URL hash', async () => {
    // 1. Test MLCoding06B
    window.location.hash = '#MLCoding06B%20RLHF%20Preference%20Alignment%20PPO%20DPO.md';
    const { unmount } = render(<App />);

    expect(await screen.findByRole('heading', { name: /ML Coding 06B · RLHF 与偏好对齐全景/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'English' }));
    expect(await screen.findByRole('heading', { name: /ML Coding 06B · RLHF & Preference Alignment/i })).toBeInTheDocument();

    unmount();

    // 2. Test MLCoding06C
    window.location.hash = '#MLCoding06C%20RLVR%20Reasoning%20GRPO%20Agentic%20RL.md';
    render(<App />);

    expect(await screen.findByRole('heading', { name: /ML Coding 06C · RLVR、推理大模型与 Agentic RL/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'English' }));
    expect(await screen.findByRole('heading', { name: /ML Coding 06C · RLVR, Reasoning Models & Agentic RL/i })).toBeInTheDocument();
  });



  it('routes directly to unified MLCoding07 Industrial ML Systems note', async () => {
    window.location.hash = '#MLCoding07%20Industrial%20Machine%20Learning%20System%20RecSys%20Reranking%20ABTesting.md';
    render(<App />);

    expect(await screen.findByRole('heading', { name: /ML Coding 07 · 工业级机器学习体系/i })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'English' }));
    expect(await screen.findByRole('heading', { name: /ML Coding 07 · Industrial ML Systems/i })).toBeInTheDocument();
  });

  it('resolves relative note links inside markdown and routes to BusinessAlgorithm02C', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      if (requestUrl.includes('MLCoding07')) {
        return {
          ok: true,
          text: async () => [
            '# ML Coding 07',
            '',
            '[第 11 章 · 特征交叉、粗排与个性化](notes/BusinessAlgorithm/BusinessAlgorithm02C%20Feature%20Interaction.md)',
          ].join('\n'),
        };
      }
      if (requestUrl.includes('BusinessAlgorithm02C')) {
        return {
          ok: true,
          text: async () => '# 第 11 章 · 特征交叉、粗排与个性化\n\n欢迎来到第 11 章。',
        };
      }
      return { ok: true, text: async () => '# Default' };
    });

    window.location.hash = '#MLCoding07%20Industrial%20Machine%20Learning%20System%20RecSys%20Reranking%20ABTesting.md';
    render(<App />);

    const link = await screen.findByRole('link', { name: /第 11 章 · 特征交叉、粗排与个性化/i });
    expect(link).toHaveAttribute('href', '#BusinessAlgorithm02C%20Feature%20Interaction.md');

    fireEvent.click(link);
    expect(await screen.findByRole('heading', { name: /第 11 章 · 特征交叉、粗排与个性化/i })).toBeInTheDocument();
  });

  it('routes to BusinessAlgorithm02C via relative path in hash and with heading', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      if (requestUrl.includes('BusinessAlgorithm02C')) {
        return {
          ok: true,
          text: async () => '# 第 11 章 · 特征交叉、粗排与个性化\n\n## 核心原理\n\n内容详情。',
        };
      }
      return { ok: true, text: async () => '# Default' };
    });

    window.location.hash = '#notes/BusinessAlgorithm/BusinessAlgorithm02C%20Feature%20Interaction.md#核心原理';
    render(<App />);

    expect(await screen.findByRole('heading', { name: /第 11 章 · 特征交叉、粗排与个性化/i })).toBeInTheDocument();
  });

  it('renders the interactive topological sort visualizer with Kahn BFS and cycle detection', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills07')
          ? '# Graph Design\n\n```topo-demo\n```'
          : '# Default Tutorial',
      };
    });

    window.location.hash = '#CoreSkills07%20Design%20Graph.md';
    render(<App />);

    const visual = await screen.findByRole('region', { name: /拓扑排序与 Kahn 算法交互式可视化/i });
    expect(visual).toBeInTheDocument();
    expect(within(visual).getByText(/Kahn 算法（入度 BFS）与环检测核心演练/i)).toBeInTheDocument();

    // Step through DAG mode
    const nextBtn = within(visual).getByRole('button', { name: /下一步/i });
    fireEvent.click(nextBtn);
    expect(within(visual).getByText(/Step 1:/i)).toBeInTheDocument();

    // Switch to Cycle detection mode
    const cycleTab = within(visual).getByRole('button', { name: /有向环死锁检测/i });
    fireEvent.click(cycleTab);
    expect(within(visual).getByText(/含 1 <-> 2 互相循环依赖/i)).toBeInTheDocument();

    // Switch to Alien Dictionary mode
    const alienTab = within(visual).getByRole('button', { name: /外星字典偏序抽取/i });
    fireEvent.click(alienTab);
    expect(within(visual).getByText(/外星文字典偏序抽取两步核心法/i)).toBeInTheDocument();
  });

  it('routes to Review 01 Core Fundamentals flashcards and expands a card', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('Review01')
          ? `# 复习卡片：常考基础题\n\n<details class="review-card">\n<summary class="review-card-summary">\n  <span class="review-card-title">归并排序 (Merge Sort)</span>\n</summary>\n<div class="review-card-content">\n\n分治排序：切分、递归、双指针合并。\n\n</div>\n</details>\n\n<details class="review-card">\n<summary class="review-card-summary">\n  <span class="review-card-title">快速排序 (Quick Sort)</span>\n</summary>\n</details>\n\n<details class="review-card">\n<summary class="review-card-summary">\n  <span class="review-card-title">动态数组实现 (Dynamic Array)</span>\n</summary>\n</details>`
          : '# Default Tutorial',
      };
    });

    window.location.hash = '#Review01%20Common%20Fundamentals.md';
    render(<App />);

    expect(await screen.findByRole('heading', { name: /复习卡片：常考基础题/i })).toBeInTheDocument();
    const mergeSortHeader = screen.getByText('归并排序 (Merge Sort)');
    expect(mergeSortHeader).toBeInTheDocument();
    expect(screen.getByText('快速排序 (Quick Sort)')).toBeInTheDocument();
    expect(screen.getByText('动态数组实现 (Dynamic Array)')).toBeInTheDocument();

    const detailsElem = mergeSortHeader.closest('details');
    expect(detailsElem).not.toHaveAttribute('open');

    fireEvent.click(mergeSortHeader);
    expect(await screen.findByText(/分治排序：切分、递归、双指针合并/i)).toBeInTheDocument();
  });

  it('restores scroll position from sessionStorage when loading a note', async () => {
    const scrollToSpy = vi.fn();
    window.scrollTo = scrollToSpy;
    sessionStorage.setItem('note_scroll_Quant01 Expectation Counting Multinomial.md', '1250');

    window.location.hash = '#Quant01%20Expectation%20Counting%20Multinomial.md';
    render(<App />);

    expect(await screen.findByRole('heading', { name: /期望/i })).toBeInTheDocument();

    await waitFor(() => {
      expect(scrollToSpy).toHaveBeenCalledWith({ top: 1250, behavior: 'instant' });
    });
  });

  it('renders the CART 2D partition visualizer, steps through splits, and inspects points', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('MLCoding09')
          ? '# ML Coding 09\n\n```cart-partition-demo\n```'
          : '# Default Tutorial',
      };
    });

    window.location.hash = '#MLCoding09%20Data%20Science%20Statistical%20Testing%20Distribution%20Drift%20C2ST.md';
    render(<App />);

    const cartSection = await screen.findByRole('region', { name: /CART 递归二叉切分/i });
    expect(cartSection).toBeInTheDocument();
    expect(within(cartSection).getByText(/步骤 0: 原始数据/i)).toBeInTheDocument();
    expect(within(cartSection).getByText(/3810.0/)).toBeInTheDocument();

    // Click Split 1
    fireEvent.click(within(cartSection).getByRole('button', { name: /切分 1: X₁ ≤ 5.0/i }));
    expect(within(cartSection).getByText(/73.7%/i)).toBeInTheDocument();
    expect(within(cartSection).getByText(/\+2809.0/i)).toBeInTheDocument();

    // Click Split 4 (Full R1~R5)
    fireEvent.click(within(cartSection).getByRole('button', { name: /切分 4: X₁ ≤ 7.5/i }));
    expect(within(cartSection).getByText(/99.1%/i)).toBeInTheDocument();
    expect(within(cartSection).getByText(/5 个区域/i)).toBeInTheDocument();
  });
});





