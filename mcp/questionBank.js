import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { getNotesIndex, getNoteContent } from './knowledge.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..');
const CANDIDATE_QUESTION_FILES = [
  path.join(PROJECT_ROOT, 'interview_questions_index.txt'),
  path.join(PROJECT_ROOT, 'bytedance_interview_questions.txt'),
];

function resolveQuestionBankFile() {
  for (const file of CANDIDATE_QUESTION_FILES) {
    if (fs.existsSync(file)) return file;
  }
  return null;
}

let cachedByteDanceQuestions = null;
let cachedQuizzes = null;

/**
 * Parse all 65 ByteDance interview questions via TSV block and text block.
 */
export function getByteDanceQuestions() {
  if (cachedByteDanceQuestions) return cachedByteDanceQuestions;

  const questionFile = resolveQuestionBankFile();
  if (!questionFile) {
    return [];
  }

  const content = fs.readFileSync(questionFile, 'utf-8');
  const questions = [];

  const tsvHeader = '序号\t所属篇章\t题目中文名';
  const tsvIndex = content.indexOf(tsvHeader);

  if (tsvIndex >= 0) {
    const tsvContent = content.slice(tsvIndex);
    const rows = tsvContent.split('\n').filter((r) => r.trim() && /^\d{2}\t/.test(r));

    for (const row of rows) {
      const cols = row.split('\t').map((c) => c.trim());
      if (cols.length >= 10) {
        const [numStr, chapter, titleZh, titleEn, typeTag, keyPoints, timeComp, spaceComp, spec, codePath] = cols;
        const num = parseInt(numStr, 10);

        let section = '一、字节工业界手撕独家真题';
        let category = 'Leetcode';
        if (num >= 17 && num <= 55) {
          section = '二、字节核心算法手撕高频真题 (LeetCode)';
          category = 'Leetcode';
        } else if (num >= 56) {
          section = '三、字节机器学习系统工程、手算推导与算法手撕';
          category = 'MLCoding';
        }

        questions.push({
          id: `bytedance-${numStr}`,
          num,
          chapter,
          titleZh,
          titleEn,
          section,
          category,
          typeTag,
          complexity: `时间 ${timeComp} | 空间 ${spaceComp}`,
          timeComplexity: timeComp,
          spaceComplexity: spaceComp,
          keyPoints,
          spec,
          codePath,
        });
      }
    }
  }

  cachedByteDanceQuestions = questions;
  return questions;
}

/**
 * High-yield curated practice quizzes across MLSys, System Design, Quant, and MLCoding.
 */
const CURATED_QUIZZES = [
  {
    id: 'quiz-sse-nginx',
    title: 'SSE 流式传输代理穿透',
    question: '在大模型 (如 ChatGPT) 流式输出架构中，Nginx 默认会缓冲上游响应导致打字机效果失效。服务端应下发哪一个专属 HTTP 响应头通知 Nginx 立即 bypass 缓冲？',
    options: [
      { id: 'A', label: 'A', text: 'Cache-Control: private' },
      { id: 'B', label: 'B', text: 'X-Accel-Buffering: no' },
      { id: 'C', label: 'C', text: 'Connection: keep-alive' },
      { id: 'D', label: 'D', text: 'Transfer-Encoding: chunked' },
    ],
    answer: 'B',
    explanation: 'Nginx 识别 X-Accel-Buffering: no 响应头后会立即关闭 proxy_buffering，将收到的每个 chunk 立即下发给客户端，避免缓冲区积攒满 4KB/16KB 才刷盘的问题。',
    category: 'SystemDesign',
    sourceNote: 'notes/SystemDesign/SystemDesignWiki SSE.md',
  },
  {
    id: 'quiz-attention-scaling',
    title: '缩放点积注意力 Softmax 梯度稳定性',
    question: '在 Transformer 的 Scaled Dot-Product Attention 计算中，为什么计算 QK^T 后必须除以根号 d_k (sqrt(d_k))？',
    options: [
      { id: 'A', label: 'A', text: '加速矩阵乘法硬件运算吞吐' },
      { id: 'B', label: 'B', text: '防止点积结果方差随维度增大而线性膨胀，导致 Softmax 进入饱和区出现极小梯度 (梯度消失)' },
      { id: 'C', label: 'C', text: '实现注意力矩阵各行元素绝对值求和为 1 的归一化' },
      { id: 'D', label: 'D', text: '消除多头注意力不同 head 之间的秩塌陷' },
    ],
    answer: 'B',
    explanation: '假设 Q 与 K 各分量为独立同分布的均值 0、方差 1 变量，点积结果的均值为 0，方差为 d_k。当 d_k 较大时点积绝对值极大，Softmax 函数输出趋于 one-hot 分布，梯度趋向于 0。除以 sqrt(d_k) 将方差拉回 1，保持稳定的反向传播梯度。',
    category: 'MLCoding',
    sourceNote: 'notes/MLCoding/MLCoding00 ML Basics Data Preprocessing Loss Functions.md',
  },
  {
    id: 'quiz-absent-passenger',
    title: '丢三落四乘客登机问题概率',
    question: '100 位乘客登机，第 1 位乘客丢失登机牌随机乱坐。其后每位乘客若原座位空着则坐原位，否则随机挑座位。求第 100 位乘客坐到自己原座位的概率是多少？',
    options: [
      { id: 'A', label: 'A', text: '1/100' },
      { id: 'B', label: 'B', text: '1/e (约 36.8%)' },
      { id: 'C', label: 'C', text: '1/2 (50%)' },
      { id: 'D', label: 'D', text: '99/100' },
    ],
    answer: 'C',
    explanation: '无论中间发生何种置换，只要第 1 位乘客的座位被坐，链式错位立即终止；只要第 100 位乘客的座位被坐，结局立即确定。在两者之间第 1 位乘客座位与第 100 位乘客座位被挑中的概率永远对称等可能，因此概率恒为 1/2。',
    category: 'quant',
    sourceNote: 'notes/quant/Quant01 Expectation Counting Multinomial.md',
  },
  {
    id: 'quiz-pagedattention-fragmentation',
    title: 'PagedAttention 核心解决痛点',
    question: 'vLLM 提出的 PagedAttention 机制相比传统连续显存分配 (Contiguous Allocation) 最核心的优化是什么？',
    options: [
      { id: 'A', label: 'A', text: '将 Attention 计算复杂度由 O(N^2) 降至 O(N)' },
      { id: 'B', label: 'B', text: '借鉴操作系统虚拟内存页表思想，按物理 Block 动态分配 KV Cache，消除显存内外部碎片并将内存浪费从 60%-80% 降至 <4%' },
      { id: 'C', label: 'C', text: '将模型权重从 FP16 量化为 INT4' },
      { id: 'D', label: 'D', text: '消除 GPU 间通信带宽瓶颈' },
    ],
    answer: 'B',
    explanation: '传统推理系统必须为每个请求提前按 max_model_len 分配连续显存槽位，产生巨大内部碎片与保留内存。PagedAttention 允许 KV Cache 物理块不连续分布并通过 Logical-to-Physical Block Table 映射，极大节省显存。',
    category: 'Mlsys',
    sourceNote: 'notes/Mlsys/MLSYS06 LLM Inference Engine Architecture nano-vllm.md',
  },
  {
    id: 'quiz-kafka-zero-copy',
    title: 'Kafka 极速吞吐零拷贝系统调用',
    question: 'Kafka 消费端在将磁盘消息发送到网络网卡 Socket 时，利用了 Linux 内核的哪一个系统调用实现零拷贝 (Zero-Copy)？',
    options: [
      { id: 'A', label: 'A', text: 'mmap()' },
      { id: 'B', label: 'B', text: 'sendfile()' },
      { id: 'C', label: 'C', text: 'epoll_wait()' },
      { id: 'D', label: 'D', text: 'splice()' },
    ],
    answer: 'B',
    explanation: 'Kafka 调用 Linux sendfile()，数据直接从 OS Page Cache 通过 DMA 拷贝到网卡缓冲区，完全不经过 JVM 用户空间，将上下文切换从 4 次降至 2 次，数据拷贝降至 2 次硬件 DMA 拷贝。',
    category: 'SystemDesign',
    sourceNote: 'notes/SystemDesign/SystemDesignWiki Kafka.md',
  },
  {
    id: 'quiz-dt-impurity',
    title: '决策树切分不纯度准则辨析',
    question: '为什么 CART / C4.5 / ID3 决策树节点切分准则绝不采用分类错误率 (Classification Error = 1 - max p_i)，而使用信息熵或基尼系数？',
    options: [
      { id: 'A', label: 'A', text: '分类错误率计算时间复杂度太高' },
      { id: 'B', label: 'B', text: '分类错误率在概率空间是严格分段线性的，对概率分布变化不敏感，常常导致切分前后错误率无增益 (导数平坦)' },
      { id: 'C', label: 'C', text: '分类错误率只能处理二分类问题' },
      { id: 'D', label: 'D', text: '分类错误率会导致过拟合风险显著加剧' },
    ],
    answer: 'B',
    explanation: '基尼系数和信息熵是严格凸函数（或严格凹的不纯度），在任意非极端切分下均能产生正的不纯度降低；而 Classification Error 是分段线性的，例如两节点在切分前后多数类占比均不变时，信息增益为 0，无法指导贪心切分。',
    category: 'MLCoding',
    sourceNote: 'notes/MLCoding/MLCoding09 Data Science Statistical Testing Distribution Drift C2ST.md',
  },
  {
    id: 'quiz-flashattention-memory-wall',
    title: 'FlashAttention 优化第一性原理',
    question: '标准 Attention 算子在 GPU 运行时的主要物理性能瓶颈是什么？FlashAttention 是如何突破它的？',
    options: [
      { id: 'A', label: 'A', text: '计算单元 Tensor Core FLOPs 算力不足；通过剪枝减少乘法运算' },
      { id: 'B', label: 'B', text: 'GPU HBM (显存) 带宽瓶颈；通过 SRAM Tiling 分块计算与 Online Softmax 融合，消除 N x N 中间注意力矩阵在 HBM 的反复读写' },
      { id: 'C', label: 'C', text: 'Host 与 Device 之间的 PCIe 通信瓶颈；通过异步传输隐藏延迟' },
      { id: 'D', label: 'D', text: '反向传播显存占用过大；通过将权重转为 BF16 精度' },
    ],
    answer: 'B',
    explanation: '标准 Attention 是 Memory-bound 访存受限算子，N x N 的 S 与 P 矩阵频繁读写高延迟 HBM。FlashAttention 在片上极速 SRAM 中利用 Online Softmax 分块累加中间统计量 (m, l)，将 HBM 访问复杂度从 O(N^2) 降至 O(N^2 d / M)。',
    category: 'Mlsys',
    sourceNote: 'notes/Mlsys/MLSYS04 Efficient Attention and Long Context.en.md',
  },
];

/**
 * Extract multiple-choice practice and quiz blocks from notes and combine with curated quizzes.
 */
export function getPracticeQuizzes() {
  if (cachedQuizzes) return cachedQuizzes;

  const index = getNotesIndex();
  const quizzes = [...CURATED_QUIZZES];
  let globalQuizId = 100;

  for (const note of index.all) {
    const rawContent = note.fullContent;
    const quizBlockRegex = /```(?:practice|quiz|mcq)([\s\S]*?)```/g;
    let match;

    while ((match = quizBlockRegex.exec(rawContent)) !== null) {
      const blockContent = match[1];
      const lines = blockContent.split('\n').map((l) => l.trim()).filter(Boolean);

      let title = 'Practice';
      let question = '';
      let answer = '';
      let explanation = '';
      const options = [];

      for (const line of lines) {
        const fieldMatch = line.match(/^(title|question|answer|correct|explanation|解析|答案)\s*[:：]\s*(.+)$/i);
        if (fieldMatch) {
          const [, rawKey, rawValue] = fieldMatch;
          const key = rawKey.toLowerCase();
          if (key === 'correct' || key === 'answer' || rawKey === '答案') {
            answer = rawValue.trim().toUpperCase();
          } else if (key === 'explanation' || rawKey === '解析') {
            explanation = rawValue.trim();
          } else if (key === 'title') {
            title = rawValue.trim();
          } else if (key === 'question') {
            question = rawValue.trim();
          }
          continue;
        }

        const optionMatch = line.match(/^(?:[-*]\s*)?([A-Ha-h]|\d+)[).、:：]\s+(.+)$/);
        if (optionMatch) {
          const [, rawKey, text] = optionMatch;
          const isMarked = /^\*/.test(text);
          const cleanText = text.replace(/^\*\s*/, '').trim();
          const label = /^[A-Ha-h]$/i.test(rawKey) ? rawKey.toUpperCase() : String(options.length + 1);

          if (isMarked && !answer) {
            answer = label;
          }

          options.push({
            id: label,
            label,
            text: cleanText,
          });
          continue;
        }

        if (!question && !line.startsWith('```')) {
          question = line;
        }
      }

      if (question && options.length >= 2) {
        quizzes.push({
          id: `quiz-${globalQuizId++}`,
          title,
          question,
          options,
          answer,
          explanation,
          sourceNote: note.relativePath,
          category: note.category,
        });
      }
    }
  }

  cachedQuizzes = quizzes;
  return quizzes;
}

/**
 * Curated list of high-impact System Design and MLSys case interview questions.
 */
const SYSTEM_DESIGN_AND_MLSYS_QUESTIONS = [
  {
    id: 'case-chatgpt-sse',
    titleZh: '设计 ChatGPT 流式生成接入层与 Server-Sent Events (SSE) 传输架构',
    titleEn: 'Design ChatGPT Streaming Ingestion Layer & Server-Sent Events (SSE) Architecture',
    category: 'SystemDesign',
    complexity: '系统架构 · Hard',
    keyPoints: 'SSE vs WebSocket 对比 · L7 代理缓冲穿透 (X-Accel-Buffering) · 空闲心跳保活 · 客户端取消断开与 GPU 算力释放 · Last-Event-ID 重连补发',
    spec: '设计支持百万级长连接的 LLM 流式对话网关。保证低首字延迟 (TTFT <= 500ms) 与平滑打字机动画，解决移动端网络重连与用户主动取消导致 GPU 算力空耗的难题。',
    codePath: 'notes/SystemDesign/SystemDesignWiki SSE.md',
  },
  {
    id: 'case-notification-system',
    titleZh: '设计高吞吐千万级移动推送与通知系统 (Notification System)',
    titleEn: 'Design Million-QPS Mobile Push Notification System',
    category: 'SystemDesign',
    complexity: '分布式系统 · Hard',
    keyPoints: '两级扇出 · APNs / FCM 批量聚合 · 频控与去重 · 优先级队列',
    spec: '设计支持每分钟 1000 万条通知推送的高可用系统，包含突发热点推送、用户休眠期聚合与多通道降级保障。',
    codePath: 'notes/SystemDesign/SystemDesign11 Notification System.md',
  },
  {
    id: 'case-flash-sale',
    titleZh: '设计百万 QPS 极速秒杀库存扣减系统 (Flash Sale)',
    titleEn: 'Design Million-QPS High-Concurrency Flash Sale Inventory System',
    category: 'SystemDesign',
    complexity: '高并发高可用 · Hard',
    keyPoints: '多级缓存 · Redis Lua 脚本原子扣减 · 异步消息削峰 · 幂等防重与超卖防御',
    spec: '在库存量 1000 件、瞬间涌入 50 万并发请求的极端场景下，保证绝对零超卖、毫秒级响应并安全持久化至数据库。',
    codePath: 'notes/SystemDesign/SystemDesign10 Flash Sale.md',
  },
  {
    id: 'case-llm-async-rl',
    titleZh: '设计大规模异步强化学习训练平台 (Async LLM RL Platform)',
    titleEn: 'Design Large-Scale Distributed Asynchronous LLM RL Platform',
    category: 'SystemDesign',
    complexity: '分布式训练系统 · Hard',
    keyPoints: 'Actor-Learner 解耦 · 连续组批 (Continuous Batching) · 样本队列流控 · 显存 KV Cache 共享',
    spec: '支持数千张 GPU 并行 Rollout 采样与多机多卡 PPO/GRPO 策略更新，消除长尾样本等待并保证数据吞吐均衡。',
    codePath: 'notes/SystemDesign/SystemDesign08 LLM Async RL Platform.md',
  },
  {
    id: 'case-paged-attention',
    titleZh: '大模型推理引擎 PagedAttention 显存物理块管理与连续组批',
    titleEn: 'LLM Inference Engine PagedAttention Memory Management & Continuous Batching',
    category: 'Mlsys',
    complexity: '机器学习系统 · Hard',
    keyPoints: '显存碎片消除 · 虚拟内存映射与物理 Block · Prefix Caching 共享 · 动态 Iteration-level 调度',
    spec: '推导为什么传统连续 KV Cache 存在 60%~80% 的内存浪费；设计并实现类似 OS 页表的 Block Table 调度机制。',
    codePath: 'notes/Mlsys/MLSYS06 LLM Inference Engine Architecture nano-vllm.md',
  },
  {
    id: 'case-flash-attention',
    titleZh: 'FlashAttention 算子优化原理与 GPU SRAM 分块 I/O 复杂度推导',
    titleEn: 'FlashAttention Kernel Optimization & GPU SRAM Tiling I/O Complexity',
    category: 'Mlsys',
    complexity: '算子工程 · Hard',
    keyPoints: 'HBM 到 SRAM 显存带宽墙 · Online Softmax 数值递推 · Tiling 矩阵分块 · 重计算反向传播',
    spec: '从第一性原理推导标准 Attention 的 O(N^2) HBM 访问开销，并证明 FlashAttention 如何将 HBM 访问降低至 O(N^2 d / M)。',
    codePath: 'notes/Mlsys/MLSYS04 Efficient Attention and Long Context.en.md',
  },
  {
    id: 'case-absent-minded-passenger',
    titleZh: 'Absent-Minded Passenger 丢三落四乘客登机问题与鞅论证明',
    titleEn: 'Absent-Minded Passenger Problem via Martingales & Symmetry',
    category: 'quant',
    complexity: '概率鞅论 · Medium',
    keyPoints: '等可能对称性 · 示性变量分解 · 递归状态转移 · 鞅与停时定理',
    spec: '100 位乘客登机，第 1 位乘客随机乱坐。其后每位乘客若原座位空着则坐原位，否则随机坐。求第 100 位乘客坐到自己座位的精确概率。',
    codePath: 'notes/quant/Quant01 Expectation Counting Multinomial.md',
  },
];

/**
 * Retrieve question bank summary statistics.
 */
export function getQuestionBankStats() {
  const bytedance = getByteDanceQuestions();
  const quizzes = getPracticeQuizzes();
  return {
    bytedanceQuestionsCount: bytedance.length,
    quizzesCount: quizzes.length,
    systemDesignAndMLSysCasesCount: SYSTEM_DESIGN_AND_MLSYS_QUESTIONS.length,
    totalQuestions: bytedance.length + quizzes.length + SYSTEM_DESIGN_AND_MLSYS_QUESTIONS.length,
  };
}

/**
 * Query questions with filtering by category, keyword, or ID.
 */
export function queryQuestions(options = {}) {
  const { category = 'all', keyword = '', limit = 10 } = options;
  const kw = keyword.toLowerCase().trim();

  const bytedance = getByteDanceQuestions();
  const combined = [
    ...bytedance.map((q) => ({
      ...q,
      source: 'ByteDance 65 High-Frequency',
    })),
    ...SYSTEM_DESIGN_AND_MLSYS_QUESTIONS.map((q) => ({
      id: q.id,
      titleZh: q.titleZh,
      titleEn: q.titleEn,
      category: q.category,
      complexity: q.complexity,
      keyPoints: q.keyPoints,
      spec: q.spec,
      codePath: q.codePath,
      source: 'System Design & MLSys Core Cases',
    })),
  ];

  let filtered = combined;
  if (category && category !== 'all') {
    filtered = filtered.filter((q) => {
      if (category === 'bytedance_65') return q.source.includes('ByteDance');
      return q.category?.toLowerCase() === category.toLowerCase();
    });
  }

  if (kw) {
    filtered = filtered.filter((q) =>
      q.titleZh?.toLowerCase().includes(kw) ||
      q.titleEn?.toLowerCase().includes(kw) ||
      q.keyPoints?.toLowerCase().includes(kw) ||
      q.spec?.toLowerCase().includes(kw) ||
      q.id?.toLowerCase().includes(kw)
    );
  }

  return filtered.slice(0, Math.min(limit, 50));
}

/**
 * Fetch a single question by ID or random pick, optionally with reference solution.
 */
export function getQuestion(idOrRandom, options = {}) {
  const { includeSolution = false, category = 'all' } = options;
  const bytedance = getByteDanceQuestions();
  const allQuestions = [
    ...bytedance.map((q) => ({ ...q, type: 'bytedance' })),
    ...SYSTEM_DESIGN_AND_MLSYS_QUESTIONS.map((q) => ({ ...q, type: 'case' })),
  ];

  let selected = null;

  if (idOrRandom === 'random' || options.random) {
    let pool = allQuestions;
    if (category && category !== 'all') {
      pool = pool.filter((q) => q.category?.toLowerCase() === category.toLowerCase());
    }
    selected = pool[Math.floor(Math.random() * pool.length)];
  } else if (typeof idOrRandom === 'string' || typeof idOrRandom === 'number') {
    const rawId = String(idOrRandom).trim().toLowerCase();
    selected = allQuestions.find((q) =>
      q.id.toLowerCase() === rawId ||
      q.id.toLowerCase() === `bytedance-${rawId}` ||
      q.id.toLowerCase() === `bytedance-${rawId.padStart(2, '0')}` ||
      String(q.num) === rawId
    );
  }

  if (!selected) {
    return { error: `Question not found for: "${idOrRandom}".` };
  }

  const result = {
    id: selected.id,
    titleZh: selected.titleZh,
    titleEn: selected.titleEn,
    category: selected.category,
    complexity: selected.complexity,
    keyPoints: selected.keyPoints,
    specification: selected.spec,
    codePath: selected.codePath,
  };

  if (includeSolution && selected.codePath) {
    const noteData = getNoteContent(selected.codePath, { maxLength: 6000 });
    result.referenceSolution = noteData.content;
  } else {
    result.hint = 'Solution omitted for interview simulation. Set include_solution: true or call evaluate_answer with this question ID to inspect solution.';
  }

  return result;
}

/**
 * Fetch multiple choice quizzes.
 */
export function queryQuizzes(options = {}) {
  const { topic = '', category = 'all', count = 3, revealAnswer = false } = options;
  const quizzes = getPracticeQuizzes();

  let pool = quizzes;
  if (category && category !== 'all') {
    pool = pool.filter((q) => q.category.toLowerCase() === category.toLowerCase());
  }
  if (topic) {
    const t = topic.toLowerCase().trim();
    pool = pool.filter((q) =>
      q.question.toLowerCase().includes(t) ||
      q.title.toLowerCase().includes(t) ||
      q.sourceNote.toLowerCase().includes(t)
    );
  }

  // Shuffle and pick
  const shuffled = [...pool].sort(() => 0.5 - Math.random());
  const selected = shuffled.slice(0, Math.min(count, 10));

  return selected.map((q) => {
    const item = {
      id: q.id,
      title: q.title,
      question: q.question,
      options: q.options,
      sourceNote: q.sourceNote,
      category: q.category,
    };
    if (revealAnswer) {
      item.answer = q.answer;
      item.explanation = q.explanation;
    }
    return item;
  });
}
