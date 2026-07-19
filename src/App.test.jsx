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
          english
            ? '# English tutorial\n\nThis is the English version.'
            : requestUrl.includes('SystemDesign05')
              ? '# System Design 05 · 可靠性、复制与故障切换'
            : requestUrl.includes('SystemDesign06')
              ? '# System Design 06 · 异步处理、消息系统与 Event Bus'
            : requestUrl.includes('SystemDesign07')
              ? '# System Design 07 · 设计图片分享与 Home Feed'
            : requestUrl.includes('SystemDesign08')
              ? '# System Design 08 · 异步 LLM RL 训练平台\n\n这个例子只有约 60 sample admission QPS。'
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

  it('does not treat Python equality operators as Obsidian highlights', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: /start mlsys/i }));

    const pythonFrame = (await screen.findByText('Python')).closest('.code-frame');
    expect(pythonFrame.querySelector('code')).toHaveTextContent(
      'if right - left + 1 == k and window == need:',
    );
    expect(pythonFrame.querySelector('mark')).toBeNull();
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

  it('presents Permutation in String as the fixed-window branch', async () => {
    globalThis.fetch.mockImplementation(async (input) => {
      const requestUrl = String(input);
      return {
        ok: true,
        text: async () => requestUrl.includes('CoreSkills29')
          ? '# Sliding Window\n\n```sliding-window-patterns\n```'
          : '# LeetCode tutorial',
      };
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));
    fireEvent.click(screen.getByRole('button', { name: /Core Skills 29 · Sliding Window/i }));

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

    expect(within(visual).getByText('right = 3：先加入第二个 a')).toBeInTheDocument();
    expect(within(visual).getByText('存在频次大于 1')).toBeInTheDocument();

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
    expect(screen.getByText('11 notes in this section')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 01 · 无状态设计范式/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 01B · 虚拟化与容器/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 02 · 数据库基本范式/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 03 · 数据库扩展三件套/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 04 · 存储系统/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 05 · 可靠性与复制/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 06 · 异步消息系统/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 07 · 图片分享与 Feed/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 08 · 异步 LLM RL 训练平台/i })).toBeInTheDocument();
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

    fireEvent.click(screen.getByRole('button', { name: /System Design 99 · 高频术语整合/i }));

    expect(await screen.findByRole('heading', { name: /高频术语整合/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign99 Glossary.md')).toHaveLength(2);
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
    expect(within(photo).getByText('READ-TIME GUARDS')).toBeInTheDocument();

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
    expect(screen.getByText('位置 3 / 7')).toBeInTheDocument();
    expect(screen.getByText('v3 = 6 > 4，最终会追上第 2 位领队。')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '下一步 →' }));

    expect(screen.getByText('位置 4 / 7')).toBeInTheDocument();
    expect(screen.getByText('v4 = 2 < 4，刷新前缀最小值，成为新领队。')).toBeInTheDocument();
    expect(screen.getByText('3 支队伍')).toBeInTheDocument();
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
