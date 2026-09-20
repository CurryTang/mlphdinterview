# ML PhD Interview Notes

Live site: [https://currytang.github.io/mlphdinterview/](https://currytang.github.io/mlphdinterview/)

## 中文

这是一个面向 ML / LLM 方向面试复习的笔记站。内容会持续整理成几个并行板块：

- **MLSYS**：CUDA、GPU kernel、分布式训练、推理系统、KV cache、MoE、post-training infra
- **LLM 八股**：world model、agent、RL / RLVR、alignment、data、evaluation 等主题
- **Quant**：概率、期望、Markov chain、常见数学面试题
- **ML Coding**：从 tokenizer、tensor module、attention 到 training loop 的实现练习
- **System Design**：后端系统设计、LLM serving、Kubernetes 训练控制面、feature store、agent infra

Kubernetes 章节的配套实验在 [`project/LLMTrainLab/`](project/LLMTrainLab/)（独立仓库 [CurryTang/LLMTrainLab](https://github.com/CurryTang/LLMTrainLab)）。README 里有逐步手打路径：GPU landscape recipe → 调度约束 → gang 排队 → 杀 rank 恢复。

配套实验：[LLMTrainLab](https://github.com/CurryTang/LLMTrainLab)（本仓库 `project/LLMTrainLab/`）。
- **业务算法八股**：推荐、搜索、广告、排序、实验设计等，正在补充
- **ML 八股**：机器学习基础，正在补充
- **LeetCode Core Skills**：数据结构、DP、图、贪心、数学、区间等核心题型
- **找工**：AI Neolabs 地图（2026 年 9 月版）与投递方向

如果你发现内容有错误、表达不清楚、公式渲染问题，或者想补充更好的例题 / 面试题，欢迎提 issue 或 PR。纠错和贡献都很欢迎。

## English

This is a personal interview-notes site for ML / LLM roles. The notes are organized into parallel sections:

- **MLSYS**: CUDA, GPU kernels, distributed training, inference systems, KV cache, MoE, post-training infra
- **LLM Interview**: world models, agents, RL / RLVR, alignment, data, evaluation
- **Quant**: probability, expectation, Markov chains, common math interview problems
- **ML Coding**: implementation exercises from tokenizers and tensor modules to attention and training loops
- **System Design**: backend design, LLM serving, Kubernetes training control planes, feature stores, agent infrastructure

The Kubernetes lab lives in [`project/LLMTrainLab/`](project/LLMTrainLab/) (standalone repo [CurryTang/LLMTrainLab](https://github.com/CurryTang/LLMTrainLab)). The README is a step-by-step path: GPU landscape recipes, placement filters, gang admission, then restart-all after a rank dies.
- **Business Algorithms**: recommendation, search, ads, ranking, experimentation, still in progress
- **ML Fundamentals**: core machine learning interview notes, still in progress
- **LeetCode Core Skills**: data structures, DP, graphs, greedy, math, interval problems
- **Jobs**: AI Neolabs map (September 2026) and where to apply

Corrections, issue reports, and contributions are welcome. If something is wrong, unclear, outdated, or missing a useful example, feel free to open an issue or PR.

## Local Development

```bash
npm install
npm run dev
```

## Checks

```bash
npm test
npm run lint
npm run build
```
