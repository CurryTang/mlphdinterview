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
        '    return True',
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
          english
            ? '# English tutorial\n\nThis is the English version.'
            : chineseContent,
      };
    });
  });

  it('starts on a prominent home page with MLSYS navigation', () => {
    render(<App />);

    expect(screen.getByRole('heading', { name: /ML \/ LLM 技术复习笔记/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /start mlsys/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^mlsys/i })).toBeInTheDocument();
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

    fireEvent.click(screen.getByRole('button', { name: /start mlsys/i }));
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
  });

  it('renders interactive multiple-choice practice blocks', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: /start mlsys/i }));

    expect(await screen.findByText('Quick Check')).toBeInTheDocument();
    expect(screen.getByText(/CUDA thread blocks/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Browser runtime/i }));

    expect(await screen.findByText(/Not quite/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /GPU SM/i }));

    expect(await screen.findByText(/Correct/)).toBeInTheDocument();
    expect(screen.getByText(/streaming multiprocessors/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /hide/i }));

    expect(screen.queryByText(/CUDA thread blocks/)).not.toBeInTheDocument();
  });

  it('renders enhanced code blocks with language labels', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: /start mlsys/i }));

    expect(await screen.findByText('Python')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Copy' })).toBeInTheDocument();
    expect(screen.getByText('def')).toHaveClass('code-token', 'keyword');
  });

  it('renders Markdown math through KaTeX without losing LaTeX commands', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: /start mlsys/i }));

    expect(await screen.findByText(/Inline math/)).toBeInTheDocument();

    const annotations = Array.from(document.querySelectorAll('annotation[encoding="application/x-tex"]')).map(
      (node) => node.textContent,
    );

    expect(annotations).toContain('QK^T');
    expect(annotations).toContain('\\sum_i x_i');
  });

  it('keeps the reader sidebar scoped to the current section', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: /start mlsys/i }));

    expect(await screen.findByRole('heading', { name: /MLSYS1/i })).toBeInTheDocument();

    const sidebar = document.querySelector('.notes-panel');
    expect(sidebar).not.toBeNull();
    expect(within(sidebar).getByRole('heading', { name: 'MLSYS' })).toBeInTheDocument();
    expect(within(sidebar).getByText('18 notes in this section')).toBeInTheDocument();
    expect(within(sidebar).queryByText('LLM八股')).not.toBeInTheDocument();
    expect(within(sidebar).queryByText('LeetCode')).not.toBeInTheDocument();
  });

  it('opens the LeetCode section from the top navigation', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));

    expect(await screen.findByRole('heading', { name: /Core Skills 1/i })).toBeInTheDocument();
    expect(screen.getAllByText('CoreSkills01 Design Dynamic Array.md')).toHaveLength(2);
    expect(screen.getByText('29 notes in this section')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Core Skills 28 · Two Pointers/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Core Skills 29 · Sliding Window/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Core Skills 28 · Two Pointers/i }));

    expect(await screen.findByRole('heading', { name: /Two Pointers/i })).toBeInTheDocument();
    expect(screen.getAllByText('CoreSkills28 Two Pointers.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /Core Skills 29 · Sliding Window/i }));

    expect(await screen.findByRole('heading', { name: /中文教程/i })).toBeInTheDocument();
    expect(screen.getAllByText('CoreSkills29 Sliding Window.md')).toHaveLength(2);
  });

  it('renders the interactive 3Sum two-pointer walkthrough', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills28')
          ? '# Two Pointers\n\n```three-sum-demo\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 28 · Two Pointers/i }));

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
        text: async () => requestUrl.includes('CoreSkills29')
          ? '# Sliding Window\n\n```sliding-window-demo\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 29 · Sliding Window/i }));

    const visual = await screen.findByRole('region', { name: '滑动窗口万能模板演示' });
    expect(within(visual).getByText('右扩：加入 A')).toBeInTheDocument();

    fireEvent.change(within(visual).getByRole('slider', { name: '选择滑动窗口演示步骤' }), {
      target: { value: '6' },
    });

    expect(within(visual).getByText('加入 A 后条件失效')).toBeInTheDocument();
    expect(within(visual).getByText('不合法')).toBeInTheDocument();
  });

  it('maps longest substring code to the sliding window template', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills29')
          ? '# Sliding Window\n\n```longest-substring-demo\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 29 · Sliding Window/i }));

    const visual = await screen.findByRole('region', { name: '最长无重复子串代码映射演示' });
    expect(within(visual).getByText('同一行骨架，逐项填入本题条件')).toBeInTheDocument();
    expect(within(visual).getByText('外层 loop right')).toBeInTheDocument();

    fireEvent.change(within(visual).getByRole('slider', { name: '选择最长无重复子串演示步骤' }), {
      target: { value: '6' },
    });

    expect(within(visual).getByText('right = 3：候选 a 重复')).toBeInTheDocument();
    expect(within(visual).getByText('候选字符已经存在')).toBeInTheDocument();

    fireEvent.click(within(visual).getByRole('button', { name: '下一步' }));
    expect(within(visual).getByText('移除旧 a，left 从 0 变成 1')).toBeInTheDocument();
  });

  it('renders the trapping rain water walkthrough and reaches six units', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills28')
          ? '# Two Pointers\n\n```rain-water-demo\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 28 · Two Pointers/i }));

    const visual = await screen.findByRole('region', { name: '接雨水双指针演示' });
    expect(within(visual).getByText(/较低的历史最高墙先结算/)).toBeInTheDocument();

    fireEvent.change(within(visual).getByRole('slider', { name: '选择接雨水演示步骤' }), {
      target: { value: '11' },
    });

    expect(visual.querySelector('.rain-water-total strong')).toHaveTextContent('6');
    expect(within(visual).getByText('重新播放')).toBeInTheDocument();
  });

  it('opens the System Design section with the new overview notes', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'System Design' }));

    expect(await screen.findByRole('heading', { name: /System Design 0/i })).toBeInTheDocument();
    expect(screen.getByText('8 notes in this section')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 1 · 无状态设计范式/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 2 · 数据库基本范式/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 3 · 数据库扩展三件套/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 4 · 存储系统/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 5 · 设计题基本流程/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 6 · 图片分享与 Feed/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 99 · 高频术语整合/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /System Design 2 · 数据库基本范式/i }));

    expect(await screen.findByRole('heading', { name: /数据库基本范式/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /System Design 3 · 数据库扩展三件套/i }));

    expect(await screen.findByRole('heading', { name: /Feature Store 分片的代价/ })).toBeInTheDocument();
    expect(screen.getByText(/Push \/ active update/)).toBeInTheDocument();
    expect(screen.getByText('Database Scaling Check 1')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /System Design 4 · 存储系统/i }));

    expect(await screen.findByRole('heading', { name: /中文教程/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign04 Storage Systems.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 5 · 设计题基本流程/i }));

    expect(await screen.findByRole('heading', { name: /设计题基本流程/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign05 Interview Flow.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 6 · 图片分享与 Feed/i }));

    expect(await screen.findByRole('heading', { name: /图片分享与 Feed/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign06 Photo Sharing Feed.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 99 · 高频术语整合/i }));

    expect(await screen.findByRole('heading', { name: /高频术语整合/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign99 Glossary.md')).toHaveLength(2);
  });

  it('shows local-only draft notes in development mode', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: '草稿区' }));

    expect(await screen.findByRole('heading', { name: '草稿区' })).toBeInTheDocument();
    expect(screen.getByText('2 notes in this section')).toBeInTheDocument();
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

    expect(screen.getByText('7 notes in this section')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Quant 7 · 递推法/i }));

    expect(await screen.findByRole('heading', { name: /递推法：健忘乘客登机/i })).toBeInTheDocument();
    expect(screen.getAllByText('Quant07 Recursion Absent-Minded Passenger.md')).toHaveLength(2);
  });
});
