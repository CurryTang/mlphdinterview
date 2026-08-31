import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

describe('App', () => {
  beforeEach(() => {
    window.history.replaceState(null, '', '/');
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
            : requestUrl.includes('SystemDesign05')
              ? '# System Design 05 · 可靠性、复制与故障切换'
            : requestUrl.includes('SystemDesign06')
              ? '# System Design 06 · 异步处理、消息系统与 Event Bus'
            : requestUrl.includes('SystemDesign07')
              ? '# System Design 07 · 设计图片分享与 Home Feed'
            : requestUrl.includes('SystemDesign08')
              ? '# System Design 08 · 异步 LLM RL 训练平台\n\n这个例子只有约 60 sample admission QPS。'
            : requestUrl.includes('SystemDesign09')
              ? '# System Design 09 · 一致性哈希\n\n节点变化时只迁移相邻区间。'
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
    expect(screen.getByText('本板块共 22 篇笔记')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Core Skills 17 · Two Pointers/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Core Skills 18 · Sliding Window/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Core Skills 19 · Stack & Monotonic Stack/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Core Skills 20 · Binary Search/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Core Skills 17 · Two Pointers/i }));

    expect(await screen.findByRole('heading', { name: /Two Pointers/i })).toBeInTheDocument();
    expect(screen.getAllByText('CoreSkills17 Two Pointers.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /Core Skills 18 · Sliding Window/i }));

    expect(await screen.findByRole('heading', { name: /中文教程/i })).toBeInTheDocument();
    expect(screen.getAllByText('CoreSkills18 Sliding Window.md')).toHaveLength(2);
  });

  it('renders the interactive 3Sum two-pointer walkthrough', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills17')
          ? '# Two Pointers\n\n```three-sum-demo\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 17 · Two Pointers/i }));

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
        text: async () => requestUrl.includes('CoreSkills18')
          ? '# Sliding Window\n\n```sliding-window-demo\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 18 · Sliding Window/i }));

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
        text: async () => requestUrl.includes('CoreSkills18')
          ? '# Sliding Window\n\n```sliding-window-patterns\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 18 · Sliding Window/i }));

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
        text: async () => requestUrl.includes('CoreSkills18')
          ? '# Sliding Window\n\n```longest-substring-demo\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 18 · Sliding Window/i }));

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
        text: async () => requestUrl.includes('CoreSkills19')
          ? '# Stack\n\n```monotonic-stack-demo\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 19 · Stack & Monotonic Stack/i }));

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
        text: async () => requestUrl.includes('CoreSkills17')
          ? '# Two Pointers\n\n```rain-water-demo\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 17 · Two Pointers/i }));

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
        text: async () => requestUrl.includes('CoreSkills22')
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
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 22 · Backtracking/i }));

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
        text: async () => requestUrl.includes('CoreSkills12')
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
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 12 · Greedy Algorithms/i }));

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
    expect(screen.getByRole('button', { name: /System Design 01 · 无状态设计范式/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 01B · 虚拟化与容器/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 02 · 数据库基本范式/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 03 · 数据库扩展三件套/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 04 · 存储系统/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 05 · 可靠性与复制/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 06 · 异步消息系统/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 07 · 图片分享与 Feed/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 08 · 异步 LLM RL 训练平台/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 09 · 一致性哈希/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 99 · 高频术语整合/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /System Design 02 · 数据库基本范式/i }));

    expect(await screen.findByRole('heading', { name: /数据库基本范式/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /System Design 03 · 数据库扩展三件套/i }));

    expect(await screen.findByRole('heading', { name: /Feature Store 分片的代价/ })).toBeInTheDocument();
    expect(screen.getByText(/Push \/ active update/)).toBeInTheDocument();
    expect(screen.getByText('Database Scaling Check 1')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /System Design 04 · 存储系统/i }));

    expect(await screen.findByRole('heading', { name: /中文教程/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign04 Storage Systems.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 05 · 可靠性与复制/i }));

    expect(await screen.findByRole('heading', { name: /可靠性、复制与故障切换/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign05 Reliability Replication.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 06 · 异步消息系统/i }));

    expect(await screen.findByRole('heading', { name: /异步处理、消息系统与 Event Bus/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign06 Async Messaging Systems.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 07 · 图片分享与 Feed/i }));

    expect(await screen.findByRole('heading', { name: /设计图片分享与 Home Feed/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign07 Photo Sharing Feed.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 08 · 异步 LLM RL 训练平台/i }));

    expect(await screen.findByRole('heading', { name: /System Design 08 · 异步 LLM RL 训练平台/ })).toBeInTheDocument();
    expect(screen.getByText(/60 sample admission QPS/)).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign08 LLM Async RL Platform.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 09 · 一致性哈希/i }));

    expect(await screen.findByRole('heading', { name: /System Design 09 · 一致性哈希/ })).toBeInTheDocument();
    expect(screen.getByText(/节点变化时只迁移相邻区间/)).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign09 Consistent Hashing.md')).toHaveLength(2);

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
    fireEvent.click(screen.getByRole('button', { name: /System Design 06 · 异步消息系统/i }));

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

  it('redirects renamed System Design note routes to the new chapter numbers', async () => {
    window.history.replaceState(null, '', '/#SystemDesign07%20Async%20Messaging%20Systems.md');

    render(<App />);

    expect(await screen.findByRole('heading', { name: /异步处理、消息系统与 Event Bus/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign06 Async Messaging Systems.md')).toHaveLength(2);

    await waitFor(() => {
      expect(window.location.hash).toBe('#SystemDesign06%20Async%20Messaging%20Systems.md');
    });
  });

  it('redirects legacy LeetCode note routes to the contiguous chapter numbers', async () => {
    window.history.replaceState(null, '', '/#CoreSkills33%20Backtracking.md');

    render(<App />);

    expect(await screen.findByRole('heading', { name: /Core Skills 22 · Backtracking/i })).toBeInTheDocument();
    expect(screen.getAllByText('CoreSkills22 Backtracking.md')).toHaveLength(2);

    await waitFor(() => {
      expect(window.location.hash).toBe('#CoreSkills22%20Backtracking.md');
    });
  });

  it('renders native HTML architecture diagrams for the System Design notes', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      let content = '# System Design tutorial';

      if (requestUrl.includes('SystemDesign00')) {
        content = '# Overview\n\n```system-design-overview-visual\n```';
      } else if (requestUrl.includes('SystemDesign06')) {
        content = '# Async Messaging\n\n```async-messaging-architecture-visual\n```';
      } else if (requestUrl.includes('SystemDesign07')) {
        content = '# Photo Sharing\n\n```photo-sharing-architecture-visual\n```';
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

    fireEvent.click(screen.getByRole('button', { name: /System Design 06 · 异步消息系统/i }));
    const asyncDiagram = await screen.findByRole('region', { name: '异步消息模式架构图' });
    fireEvent.click(within(asyncDiagram).getByRole('button', { name: 'Kafka groups' }));
    expect(within(asyncDiagram).getByText('系统是实现，group 决定语义')).toBeInTheDocument();
    expect(within(asyncDiagram).getByText('group: analytics')).toBeInTheDocument();
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
    fireEvent.click(screen.getByRole('button', { name: /Quant 6 · 高维积分/i }));

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
    fireEvent.click(screen.getByRole('button', { name: /Quant 1 · 期望与计数/i }));

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
        text: async () => requestUrl.includes('Quant07')
          ? '# 递推法：健忘乘客登机\n\n答案是 $1/2$。'
          : '# Quant tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'Quant' }));

    expect(screen.getByText(/本板块共 \d+ 篇笔记/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Quant 7 · 递推法/i }));

    expect(await screen.findByRole('heading', { name: /递推法：健忘乘客登机/i })).toBeInTheDocument();
    expect(screen.getAllByText('Quant07 Recursion Absent-Minded Passenger.md')).toHaveLength(2);
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
          ? '# Quant 11 · 鞅、停时与随机游走\n\n```martingale-rw-demo\n```\n\nWald 一阶与二阶等式。'
          : '# Quant tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'Quant' }));
    fireEvent.click(screen.getByRole('button', { name: /Quant 11 · 鞅、停时与随机游走/i }));

    expect(await screen.findByRole('heading', { name: /Quant 11 · 鞅、停时与随机游走/i })).toBeInTheDocument();
    expect(screen.getByText('Wald 等式、1D 随机游走与最优决策')).toBeInTheDocument();
    expect(screen.getByText('胜率 P(到达 +a)')).toBeInTheDocument();
    expect(screen.getByText('期望停止时间 E[T]')).toBeInTheDocument();

    // Switch to Secretary Problem tab
    fireEvent.click(screen.getByRole('tab', { name: /秘书问题 37% 法则/i }));
    expect(screen.getByText(/候选人总人数 n/i)).toBeInTheDocument();
    expect(screen.getByText(/1\/e ≈ 36\.79%/)).toBeInTheDocument();

    // Switch to Pattern Waiting & Li's Martingale tab
    fireEvent.click(screen.getByRole('tab', { name: /模式等待与赌场鞅/i }));
    expect(screen.getByText(/HTTH vs HTHT/)).toBeInTheDocument();
    expect(screen.getByText(/E\[T_A\] = 18/)).toBeInTheDocument();
    expect(screen.getByText(/E\[T_B\] = 20/)).toBeInTheDocument();
  });

  it('opens Quant 12 Brownian motion and stochastic calculus note and renders interactive components', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('Quant12')
          ? '# Quant 12 · 布朗运动、伊藤微积分、停时与期权交易应用\n\n```brownian-motion-demo\n```\n\n```two-d-walk-demo\n```\n\n```ito-geometry-demo\n```\n\n```reflection-principle-demo\n```\n\n```delta-hedging-demo\n```'
          : '# Quant tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'Quant' }));
    fireEvent.click(screen.getByRole('button', { name: /Quant 12 · 布朗运动/i }));

    expect(await screen.findByRole('heading', { name: /Quant 12 · 布朗运动、伊藤微积分、停时与期权交易应用/i })).toBeInTheDocument();
    expect(screen.getByLabelText('布朗运动轨道与二次变差演示')).toBeInTheDocument();
    expect(screen.getByLabelText('2D 随机游走与布朗运动极限演示')).toBeInTheDocument();
    expect(screen.getByLabelText('伊藤几何与斯特拉托诺维奇积分对比演示')).toBeInTheDocument();
    expect(screen.getByLabelText('停时与反射原理演示')).toBeInTheDocument();
    expect(screen.getByLabelText('期权 Delta 对冲与 Gamma 损益演示')).toBeInTheDocument();
  });

  it('opens Quant 13 Game Theory note and renders the game theory simulator', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('Quant13')
          ? '# Quant 13 · 博弈论与策略性决策：纳什均衡、逆向归纳与华尔街量化经典\n\n```game-theory-interactive-demo\n```'
          : '# Quant tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'Quant' }));
    fireEvent.click(screen.getByRole('button', { name: /Quant 13 · 博弈论/i }));

    expect(await screen.findByRole('heading', { name: /Quant 13 · 博弈论与策略性决策/i })).toBeInTheDocument();
    expect(screen.getByLabelText('博弈论与策略性决策演示')).toBeInTheDocument();
    expect(screen.getByText(/量化博弈论与经典策略交互模拟器/)).toBeInTheDocument();

    // Switch tab to Truel
    fireEvent.click(screen.getByRole('button', { name: '三方决斗' }));
    expect(screen.getByText(/枪手 A 最终胜率/)).toBeInTheDocument();

    // Switch tab to Auctions
    fireEvent.click(screen.getByRole('button', { name: '拍卖与胜者诅咒' }));
    expect(screen.getByText(/一阶密封出价 FPA/)).toBeInTheDocument();
  });

  it('renders the Palindromic Substrings 2D DP matrix visual walkthrough and steps through states', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills10')
          ? '# Dynamic Programming\n\n```palindrome-dp-demo\n```'
          : '# Default note',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 10 ·/i }));

    const pdp = await screen.findByRole('region', { name: '回文子串 2D DP 状态转移演示' });
    expect(within(pdp).getByText(/Palindromic Substrings：二维 DP 状态表填表与西南角依赖可视化/i)).toBeInTheDocument();

    // Verify presence of preset buttons and table
    expect(within(pdp).getByRole('button', { name: /s = "ababa"/i })).toBeInTheDocument();
    expect(within(pdp).getByRole('button', { name: /s = "babad"/i })).toBeInTheDocument();

    // Step forward via slider
    const slider = within(pdp).getByRole('slider', { name: '选择回文子串 DP 演示步骤' });
    fireEvent.change(slider, { target: { value: '1' } });

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
        text: async () => requestUrl.includes('CoreSkills10')
          ? '# Dynamic Programming\n\n```coin-change-demo\n```'
          : '# Default note',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 10 ·/i }));

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
        text: async () => requestUrl.includes('CoreSkills10')
          ? '# Dynamic Programming\n\n```subset-sum-demo\n```'
          : '# Default note',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 10 ·/i }));

    const pssv = await screen.findByRole('region', { name: '分割等和子集 0/1 背包状态转移演示' });
    expect(within(pssv).getByText(/Partition Equal Subset Sum：分割等和子集可达性与决策分支可视化/i)).toBeInTheDocument();

    // Verify preset buttons exist
    expect(within(pssv).getByRole('button', { name: /nums = \[1, 5, 11, 5\]/i })).toBeInTheDocument();
    expect(within(pssv).getByRole('button', { name: /nums = \[1, 2, 3, 5\]/i })).toBeInTheDocument();

    // Step forward using slider
    const slider = within(pssv).getByRole('slider', { name: '选择分割等和子集 DP 演示步骤' });
    fireEvent.change(slider, { target: { value: '1' } });

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

  it('routes directly to ML Coding 00B and ML Coding 01B via URL hash and renders them properly', async () => {
    // 1. Test MLCoding00B
    window.location.hash = '#MLCoding00B%20LLM%20Basics%20Decoder%20Only%20Precision%20Alignment%20Distillation.md';
    const { unmount } = render(<App />);

    expect(await screen.findByRole('heading', { name: /ML Coding 00B · LLM 基础：Decoder-Only 为何胜出/i })).toBeInTheDocument();

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
});



