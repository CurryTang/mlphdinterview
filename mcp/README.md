# mlphdinterview MCP Server (题库与知识库接入指南)

> 本 MCP (Model Context Protocol) 服务基于 [CurryTang/mlphdinterview](https://github.com/CurryTang/mlphdinterview) 构建，将全库 **290+ 篇高质量技术笔记** 与 **字节跳动等大厂高频面试手撕题库** 封装为标准 MCP 工具，能够直接接入 **ChatGPT 桌面版 / Custom GPTs、Claude Desktop、Cursor、Windsurf、Antigravity** 等大模型客户端，作为技术复习的**第一手知识来源与自动化模拟面试官（题库）**。

---

## 1 · 核心能力与工具清单 (Tools & Features)

| 工具名称 (Tool Name) | 核心功能定位 | 典型使用场景 |
|:---|:---|:---|
| `search_knowledge_base` | **知识库全量检索**：支持对 290+ 篇专业笔记（MLSys、System Design、Quant、LeetCode、MLCoding、推荐搜索算法）进行关键词和中英文语义检索，返回得分、匹配摘要与核心章节目录。 | 遇到具体技术概念（如 "KV Cache", "Kafka zero copy", "PagedAttention", "Martingale"）时调用作为事实依据。 |
| `get_note` | **笔记原文与章节提取**：获取指定笔记全文或特定二级/三级标题内容（支持中英文双语），具备智能长文本截断与目录导航以节省上下文。 | 针对某一专项技术进行深度理论溯源与原理精读。 |
| `list_knowledge_topics` | **知识库领域全景目录**：按类别罗列所有可用篇章、文章总数及双语覆盖状态。 | 制定复习路线图或按领域梳理知识拓扑。 |
| `get_interview_question` | **面试题库调取与随机抽题**：从精选题库中提取高频真题，包括**字节跳动 65 道独家手撕/高频真题**、分布式系统设计核心实战 Case、MLSys 系统工程手撕与量化经典谜题。支持 `random: true` 用于模拟面试。 | 模拟面试出题，默认隐去标答，由 AI 扮演面试官考察候选人。 |
| `get_practice_quiz` | **核心选择题与 Flashcards 速测**：调取理论核心概念多选题，包含题干、选项 (A/B/C/D)，支持隐藏标答以供即时自测。 | 快速检验对第一性原理与工程陷阱的掌握（如 Softmax 缩放根号 dk 原因、Nginx SSE 缓冲穿透）。 |
| `evaluate_answer` | **参考解答与评测打分标准**：调取题目的复杂度标准边界（Time/Space Complexity）、必备答题核心考点与标准工程代码/数学证明。 | 面试官评估候选人回答，进行针对性追问与打分。 |

---

## 2 · 接入 ChatGPT 指引

### 方式一：接入 ChatGPT 桌面版 (Desktop MCP)

OpenAI 官方在 ChatGPT 桌面客户端（macOS / Windows）中已正式支持 MCP 协议。

1. 打开 ChatGPT 桌面端的设置或 MCP 配置文件（通常位于 `~/Library/Application Support/OpenAI/ChatGPT/chatgpt_desktop_config.json` 或在客户端设置的 `Developer / Model Context Protocol` 中添加）。
2. 在 `mcpServers` 中添加配置：

```json
{
  "mcpServers": {
    "mlphdinterview": {
      "command": "node",
      "args": [
        "/Users/czk/Documents/mlsysnotes/MLSYS_tutorial/mcp/index.js"
      ]
    }
  }
}
```
*(请将路径替换为您本地实际的项目绝对路径)*

3. 重启 ChatGPT 桌面端。在对话框输入：
   > *"请从 mlphdinterview 题库中随机抽取一道机器学习系统或字节跳动工业界手撕题，扮演面试官对我进行模拟面试。先只出题目与约束条件，不要直接给答案。"*

---

### 方式二：接入 ChatGPT 网页版 (Custom GPT Action)

如果您使用的是 ChatGPT Web 端（Plus / Team / Enterprise），可通过 **Custom GPTs 的 Actions** 接入：

1. **启动本地 HTTP 服务**：
   ```bash
   npm run mcp:http
   # 或指定端口：node mcp/index.js --http 3333
   ```
2. **通过公网穿透工具暴露（或部署至服务器）**：
   使用 Cloudflare Tunnel 或 ngrok：
   ```bash
   ngrok http 3333
   ```
   获得公网 URL（例如 `https://xxxx.ngrok-free.app`）。
3. **在 ChatGPT 网页端创建 Custom GPT**：
   - 访问 [chatgpt.com/gpts/editor](https://chatgpt.com/gpts/editor) 点击 **Create a GPT**。
   - 在 **Configure** 标签页下点击 **Create new action**。
   - 打开项目内的 `mcp/openapi.json`，将其中的 `url: "http://localhost:3333"` 替换为您的公网穿透 URL。
   - 将完整 JSON 粘贴到 GPT 的 **Schema** 输入框中。
   - 在 GPT 的 Instructions 中填入：
     > *"你是基于 CurryTang/mlphdinterview 知识库的顶级技术面试官与复习导师。回答任何技术问题均优先调用 searchKnowledgeBase 查阅笔记，并可通过 getRandomQuestion 与 getPracticeQuizzes 组织模拟面试。"*
   - 保存后即可随时在 ChatGPT 网页端、手机端 App 中进行复习与题库刷题！

---

## 3 · 接入 Claude (Claude Desktop & Claude Code)

### 3.1 接入 Claude Desktop

在 Claude Desktop 配置文件中添加（macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`，Windows: `%APPDATA%\Claude\claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "mlphdinterview": {
      "command": "node",
      "args": [
        "/Users/czk/Documents/mlsysnotes/MLSYS_tutorial/mcp/index.js"
      ]
    }
  }
}
```

保存后重启 Claude Desktop，界面右下角将出现锤子图标，即可直接调用相关工具。

### 3.2 接入 Claude Code (CLI / Project Mode)

仓库根目录下已内置 `.claude/mcp.json` 与 `.claude/skills/mlphdinterview/SKILL.md`：
- 在仓库中直接运行 `claude` 命令，Claude Code 会自动加载本项目的 MCP 服务与专用 Skill。
- 也可以通过全局添加：
  ```bash
  claude mcp add mlphdinterview node /Users/czk/Documents/mlsysnotes/MLSYS_tutorial/mcp/index.js
  ```

---

## 4 · 接入 Cursor / Windsurf / Antigravity IDE

仓库根目录下已内置 `.cursor/mcp.json`：
```json
{
  "mcpServers": {
    "mlphdinterview": {
      "command": "node",
      "args": [
        "${workspaceFolder}/mcp/index.js"
      ]
    }
  }
}
```
项目被打开时，Cursor / Windsurf / Antigravity 将自动识别并启动该 MCP 服务器。

---

## 5 · Skill 接口集成 (Agent Skills for Claude / Antigravity)

除了 MCP 协议调用，本项目还提供了标准 **Agent Skill 规范** 文件：
- **技能文件路径**：`skills/mlphdinterview/SKILL.md`（并同步至 `.claude/skills/mlphdinterview/SKILL.md`）
- **功能特性**：
  - 自动识别 MLSys、System Design、Quant、LeetCode 与 ML Coding 技术提问并精准导流。
  - 内置严苛的三阶段模拟面试协议（Phase 1 零泄题呈现 $\to$ Phase 2 渐进式引导追问 $\to$ Phase 3 四维度评分与参考答案对比）。
  - 支持直接在无 MCP 进程的环境下作为 Prompt / Instruction 执行。

---

## 5 · 命令行快速启动与测试

```bash
# 1. 默认 Stdio 模式 (用于 MCP 客户端集成)
node mcp/index.js

# 2. 启动 HTTP / SSE 模式 (默认端口 3333，支持浏览器仪表盘与 REST API)
npm run mcp:http

# 3. 访问本地控制面板与 OpenAPI 描述
打开浏览器访问: http://localhost:3333
OpenAPI 描述文件: http://localhost:3333/openapi.json
```

---

## 6 · 预设 Prompts 推荐对话模板

MCP 注册了以下两组标准 Prompt，可在支持的客户端一键拉起：

1. **`mock-interview`** (高拟真模拟面试)：
   > *"请使用 mlphdinterview 的 mock-interview 模式，抽取一道 MLSys 或系统设计 Hard 题目，分阶段向我提问并考核我的时间空间复杂度与工程边界。"*
2. **`topic-drill`** (知识点高压快问)：
   > *"请使用 topic-drill 对 KV Cache 机制进行高压快问，包括 MHA vs GQA 显存公式、PagedAttention 原理与生产级调优要点。"*
