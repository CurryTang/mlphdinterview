# -*- coding: utf-8 -*-

with open("notes/MLCoding/MLCoding08 Minimalist Agent Loop Architecture 300 Lines Pi Agent.md", "w", encoding="utf-8") as f:
    f.write("""# ML Coding 08 · 极简 Agent 系统实现：基于 Pi Agent 的 300 行核心循环、工具调用与状态机架构

在当下 AI Agent 领域，LangChain、CrewAI、AutoGen 等工业级框架为了追求通用性，层层堆叠了复杂的类继承、隐式 Prompt 注入与厚重的依赖树，往往导致**黑盒难以调试、上下文滥用膨胀、Token 消耗失控**。

2025~2026 年，由 Mario Zechner 发起的开源项目 **Pi Agent (`pi-coding-agent`)** 凭借其**“极简主义（Do More With Less）”**哲学掀起了 Agent 架构回归本真的浪潮：
> **Agent 的核心不是复杂的调度框架，而是一个清晰可控的 ReAct 状态循环（State Loop）、一套克制的精简工具集（Four-Tool Philosophy）以及严密的上下文预算管理。其全部核心引擎仅需约 300~400 行代码即可完整实现！**

本篇将从第一性原理出发，剖析极简 Agent 系统的数学状态机模型，并手把手给出一套**零三方重依赖、纯原生 Python、约 300 行且开箱即用的工业级最小 Agent 系统实现**。

---

## 模块一：Agent 系统的本质与“四工具极简哲学”

```text
现代 Coding / Reasoning Agent 的极简拓扑架构:
┌────────────────────────────────────────────────────────────────────────┐
│                        用户输入 (User Prompt)                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               上下文与预算管理器 (Context Budget Manager)               │
│ • System Prompt (< 800 Tokens) • 短期会话历史 (Sliding Window / Tree)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     LLM 核心推理引擎 (Model Inference)                 │
│ • 模型输出: 思考过程 (Thought) + 工具调用结构体 (Tool Calls: name, args) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
                     ┌──────────────┴──────────────┐
                     │ 是否产生 Tool Call 调用?    │
                     └──────┬───────────────┬──────┘
                   [No]     │               │ [Yes]
                            ▼               ▼
┌───────────────────────────────┐ ┌──────────────────────────────────────┐
│  返回最终文本答复 (Turn Done)  │ │   四工具极简执行沙箱 (The 4 Tools)    │
│  退出单轮循环，等待用户下一输入 │ │   1. read_file   2. write_file       │
│                               │ │   3. edit_file   4. bash / exec       │
└───────────────────────────────┘ └──────────────────┬───────────────────┘
                                                     ▼
                                  ┌──────────────────────────────────────┐
                                  │ 结果截断与上下文注入 (Inject Result)  │
                                  │ • 输出截断 (Max 50KB) • 拼接为 Role:tool│
                                  └──────────────────┬───────────────────┘
                                                     │ (循环迭代 while loop)
                                                     └───────► 回到 LLM 输入
```

### 1. 为什么“四工具哲学（Four-Tool Philosophy）”能解决 95% 的任务？

许多 Agent 框架试图为每个细分任务封装独立 Tool（如 `PythonInterpreterTool`、`GitCommitTool`、`SearchFileTool`、`CalculatorTool`），这不仅造成 System Prompt 中 Tool Schema 严重膨胀（吃掉数千 Token），还增加了模型挑选工具的幻觉几率。

Pi Agent 证明：只需 **4 个底座基元工具（4 Foundational Primitives）**，即可覆盖几乎所有软件工程与复杂推理任务：
1. **`read_file(path, start_line, end_line)`**：按行号精准读取文件切片，严格控制输出 Token 数量；
2. **`write_file(path, content)`**：创建新文件或全量覆写配置；
3. **`edit_file(path, old_str, new_str)`**：基于精确子字符串匹配或行号范围做局部替换（相比全量重写极度省 Token 且不易丢弃上下文代码）；
4. **`bash(command, timeout_ms, cwd)`**：利用操作系统的原生 Shell 统一执行 `git`、`pytest`、`grep`、`find`、`npm`、`curl` 以及自定义脚本。

---

## 模块二：Agent 状态机与 Markov 决策过程数学建模

Agent 的多轮执行过程在数学上可以形式化为一个**部分可观察 Markov 决策过程（POMDP）**：

### 1. 状态序列与动作生成
- **状态轨迹（Trajectory History）**：
  $$s_t = \\left( h_0, a_1, o_1, a_2, o_2, \\dots, a_t, o_t \\right)$$
  其中 $h_0$ 为系统预设 Prompt 与用户初始 Query，$a_i$ 为模型在第 $i$ 步采取的动作（工具调用或最终输出），$o_i$ 为环境执行工具后的观察结果（Observation / Tool Result）。

- **策略分布（Policy Inference）**：
  $$\\pi_\\theta(a_{t+1} \\mid s_t) = \\text{LLM}(s_t; \\Theta)$$

- **环境状态转移（Deterministic Environment Transition）**：
  $$o_{t+1} = \\mathcal{E}(a_{t+1}), \\quad s_{t+1} = s_t \\circ \\left[ a_{t+1}, o_{t+1} \\right]$$

### 2. 状态机迁移全景

```text
Agent 核心执行状态机 (Finite State Machine):
[IDLE] ────(收到用户 Prompt)────► [FORMAT_CONTEXT]
                                         │
                                         ▼
[WAIT_USER] ◄───(无 Tool Call)──── [INVOKE_LLM]
                                         │
                                  (解析出 Tool Call)
                                         ▼
                                  [EXECUTE_TOOL]
                                         │
                                  (输出截断与校验)
                                         ▼
                                  [APPEND_HISTORY]
                                         │
                               (步数 t < Max_Steps)
                                         ▼
                                   [INVOKE_LLM] (循环)
```

---

## 模块三：300 行自包含 Python 极简 Agent 完整实现

以下是一套完整的、**零外部 Heavy 依赖（仅依赖 Python 原生标准库与可选 `urllib.request` / OpenAI 兼容接口）**的工业级极简 Agent 实现。

代码涵盖：
- **`ToolRegistry`**：利用 Python 原生 `inspect` 和类型注解自动生成 OpenAI 兼容的 Function Calling JSON Schema；
- **4 大内置核心基元工具**（`read_file`, `write_file`, `edit_file`, `bash`），自带行号截断与 50KB 防爆护栏；
- **`AgentLoop`**：支持流式与单步迭代的 ReAct 执行循环、工具执行错误自我纠错、以及最大步数防死循环熔断机制；
- **交互式 CLI REPL 终端**。

```python
import os
import sys
import json
import inspect
import subprocess
import urllib.request
import urllib.error
from typing import Dict, List, Any, Callable, Tuple, Optional
from dataclasses import dataclass, field, asdict

# 1. 消息模型与上下文数据结构
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]

@dataclass
class Message:
    role: str # "system" | "user" | "assistant" | "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

# 2. 工具注册中心与 JSON Schema 自动推导
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []

    def register(self, func: Callable) -> Callable:
        name = func.__name__
        doc = inspect.getdoc(func) or "No description provided."
        sig = inspect.signature(func)
        properties: Dict[str, Any] = {}
        required: List[str] = []
        type_map = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}

        for param_name, param in sig.parameters.items():
            param_type = type_map.get(param.annotation, "string")
            properties[param_name] = {"type": param_type, "description": f"Parameter {param_name}"}
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": doc.strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        self._tools[name] = func
        self._schemas.append(schema)
        return func

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Error: Tool '{name}' is not registered."
        try:
            res = self._tools[name](**args)
            res_str = str(res)
            # 防暴护栏：输出过长时进行截断（保护 Context 窗口）
            if len(res_str) > 50000:
                res_str = res_str[:25000] + "\\n\\n... [Output Truncated: Exceeded 50KB] ...\\n\\n" + res_str[-25000:]
            return res_str
        except Exception as e:
            return f"Tool Execution Error ({name}): {type(e).__name__}: {str(e)}"

    @property
    def schemas(self) -> List[Dict[str, Any]]:
        return self._schemas

# 3. 四大核心基元工具实现 (The 4 Foundational Tools)
registry = ToolRegistry()

@registry.register
def read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
    \"\"\"Read contents of a file within a 1-indexed line range.\"\"\"
    if not os.path.exists(path):
        return f"Error: File '{path}' does not exist."
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total_lines = len(lines)
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end_line)
        sliced = lines[start_idx:end_idx]
        numbered = [f"{start_idx + i + 1}: {line}" for i, line in enumerate(sliced)]
        return f"File: {path} (Showing lines {start_idx+1}-{end_idx} of {total_lines}):\\n" + "".join(numbered)
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"

@registry.register
def write_file(path: str, content: str) -> str:
    \"\"\"Create a new file or completely overwrite an existing file with new content.\"\"\"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{path}'."
    except Exception as e:
        return f"Error writing to file '{path}': {str(e)}"

@registry.register
def edit_file(path: str, target_chunk: str, replacement_chunk: str) -> str:
    \"\"\"Replace an exact unique target text chunk with replacement text in a file.\"\"\"
    if not os.path.exists(path):
        return f"Error: File '{path}' does not exist."
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        count = content.count(target_chunk)
        if count == 0:
            return f"Error: target_chunk not found in '{path}'. Make sure exact whitespace matches."
        if count > 1:
            return f"Error: target_chunk found {count} times in '{path}'. Must be uniquely identifiable."
        new_content = content.replace(target_chunk, replacement_chunk, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Successfully replaced target chunk in '{path}'."
    except Exception as e:
        return f"Error editing file '{path}': {str(e)}"

@registry.register
def bash(command: str, timeout_seconds: int = 30) -> str:
    \"\"\"Execute a shell command in the workspace and return stdout and stderr.\"\"\"
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout_seconds, cwd=os.getcwd()
        )
        out, err = proc.stdout.strip(), proc.stderr.strip()
        result_parts = [f"Exit Code: {proc.returncode}"]
        if out: result_parts.append(f"STDOUT:\\n{out}")
        if err: result_parts.append(f"STDERR:\\n{err}")
        return "\\n\\n".join(result_parts) if (out or err) else f"Command completed with Exit Code: {proc.returncode}"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout_seconds} seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

# 4. LLM 客户端封装 (原生 HTTP / OpenAI 协议兼容)
class LLMClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat_completion(self, messages: List[Message], tools: List[Dict[str, Any]]) -> Message:
        url = f"{self.base_url}/chat/completions"
        payload = {"model": self.model, "messages": [m.to_dict() for m in messages], "temperature": 0.2}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choice = data["choices"][0]["message"]
            role = choice.get("role", "assistant")
            content = choice.get("content")
            raw_tool_calls = choice.get("tool_calls", [])
            tool_calls = [
                ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=json.loads(tc["function"]["arguments"]))
                for tc in raw_tool_calls
            ]
            return Message(role=role, content=content, tool_calls=tool_calls if tool_calls else None)
        except Exception as e:
            raise RuntimeError(f"LLM API Error: {str(e)}")

# 5. 核心 Agent 循环执行引擎 (The ReAct Agent Loop)
class PiAgent:
    def __init__(self, client: LLMClient, tool_registry: ToolRegistry, max_steps: int = 25):
        self.client = client
        self.tools = tool_registry
        self.max_steps = max_steps
        self.system_prompt = (
            "You are an expert autonomous software engineer with access to 4 tools: "
            "read_file, write_file, edit_file, and bash.\\n"
            "Guidelines:\\n"
            "1. First explore code before editing.\\n"
            "2. Prefer edit_file over write_file for surgical changes.\\n"
            "3. Verify all changes with bash commands."
        )
        self.history: List[Message] = [Message(role="system", content=self.system_prompt)]

    def run_turn(self, user_query: str) -> str:
        self.history.append(Message(role="user", content=user_query))
        print(f"\\n[User]: {user_query}")

        for step in range(1, self.max_steps + 1):
            print(f"\\n--- [Step {step}/{self.max_steps}] Querying LLM ---")
            assistant_msg = self.client.chat_completion(self.history, self.tools.schemas)
            self.history.append(assistant_msg)

            if assistant_msg.content:
                print(f"[Assistant]:\\n{assistant_msg.content}")

            if not assistant_msg.tool_calls:
                print(f"\\n[Agent Finished]: Turn complete.")
                return assistant_msg.content or ""

            for tc in assistant_msg.tool_calls:
                print(f"[Tool Call]: {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})")
                observation = self.tools.execute(tc.name, tc.arguments)
                self.history.append(Message(role="tool", content=observation, tool_call_id=tc.id))

        return "Warning: Reached maximum execution steps limit."

if __name__ == "__main__":
    client = LLMClient()
    agent = PiAgent(client=client, tool_registry=registry)
    if len(sys.argv) > 1:
        agent.run_turn(" ".join(sys.argv[1:]))
    else:
        print("PiAgent ready. Set OPENAI_API_KEY and run: python pi_agent.py '<task>'")
```

---

## 模块四：工业级 Agent 的核心工程陷阱与防御机制

```text
工业级 Agent 运行五大常见失效模式与防御策略:
┌────────────────────────────────────────────────────────────────────────┐
│ 1. 工具调用死循环 (Infinite Tool Call Loop)                            │
│ • 现象: 模型对同一错误命令反复执行 10 次以上                           │
│ • 防御: 维护近 3 步 (tool_name, args) 哈希历史，重复触发强制注入报警  │
├────────────────────────────────────────────────────────────────────────┤
│ 2. 上下文窗口击穿 (Context Explosion via Oversized Logs)                │
│ • 现象: bash 或 read_file 输出 2MB 编译日志瞬间打满 128k 窗口          │
│ • 防御: 强制双端截断 (保留 Head 25KB + Tail 25KB，中间插提示)           │
├────────────────────────────────────────────────────────────────────────┤
│ 3. 参数类型幻觉 (Schema Arguments Hallucination)                       │
│ • 现象: 模型生成无效 JSON 或缺失 required 字段                         │
│ • 防御: 工具执行层捕获 TypeError 并返回详细 JSON 修正指引               │
├────────────────────────────────────────────────────────────────────────┤
│ 4. 不可逆破坏性执行 (Destructive Command Execution)                    │
│ • 现象: 模型执行 `rm -rf /` 或覆写未备份核心代码                       │
│ • 防御: 危险命令黑名单正则拦截 + 影子工作区 (Git Worktree 隔离)         │
├────────────────────────────────────────────────────────────────────────┤
│ 5. 会话漂移与遗忘 (Instruction Drift in Long Horizon)                  │
│ • 现象: 执行 20 步后忘记了用户的最初指令要求                           │
│ • 防御: System Prompt Pinning (系统提示词在上下文修剪中保持最高优先级) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 模块五：Agent 核心架构自测单选题（Interactive Quizzes）

<details class="exercise">
<summary><span class="q-label">Q1 · 极简 Agent 工具哲学</span> <span class="q-text">在设计高效的软件工程 Agent 时，为什么以 Pi Agent 为代表的现代架构强烈推崇“四工具哲学（read/write/edit/bash）”，而不是为 Python、Git、测试分别封装数十个专门工具？</span></summary>

- [ ] **A.** 因为大模型一次只能记住最多 4 个函数的定义，超过 4 个会发生注意力溢出。
- [ ] **B.** 专门工具的执行速度显著慢于操作系统原生 Bash 执行。
- [x] **C.** 封装过多专用工具会导致 System Prompt 中 Tool Schema 严重膨胀（浪费宝贵 Context 预算），增加模型挑选工具时的路由幻觉；而通过通用的 `bash` 原生命令与精准文本编辑基元即可更优雅地覆盖所有功能。
- [ ] **D.** 四工具哲学只能用于终端 CLI 环境，无法扩展到浏览器或 GUI 场景。

> 💡 **答案解析**：
> - **正确选项：C**。
>   1. **Token 预算与注意力稀释**：每个注册到 Agent 的 Tool 都需要提供 JSON Schema 描述（包含各参数字段）。工具过多不仅直接吃掉数千 Token 预算，还会使 LLM 在决定使用哪个工具时注意力分散，引发幻觉；
>   2. **通用表达力**：操作系统的 Bash 原生就支持调用 `git`、`pytest`、`curl` 等一切命令行程序，无需人为发明冗余抽象。
</details>

<details class="exercise">
<summary><span class="q-label">Q2 · Agent 核心循环与上下文防爆</span> <span class="q-text">在 Agent 执行过程中，当模型调用 `bash` 运行测试命令输出了高达 5MB 的详细失败堆栈时，下列哪种上下文管理策略在<strong>保留关键报错信息的同时最能有效防止 Context Window 击穿</strong>？</span></summary>

- [ ] **A.** 直接将 5MB 输出原封不动全部追加到 `role: tool` 消息中，依赖现代 LLM 的 1M 长窗口自然消化。
- [ ] **B.** 直接将整个输出丢弃，只给模型返回 Exit Code。
- [x] **C.** 执行双端保留截断（Head/Tail Truncation）：保留头部前 20KB（启动与参数信息）和尾部后 20KB（最终核心报错与总结），并在中间插入明显的截断占位提示 `[Output Truncated: Exceeded Limit]`。
- [ ] **D.** 立即终止当前 Agent 运行并直接向用户报错。

> 💡 **答案解析**：
> - **正确选项：C**。
>   1. **长日志结构特征**：绝大多数编译与测试日志的关键信息集中在最开头的命令配置以及最末尾的 Traceback 报错摘要，中间往往是数万行冗余的进度条或无用调用栈；
>   2. **双端截断优势**：在 50KB 以内完美保留首尾关键信息，既能让模型精准定位 Bug 根因，又杜绝了超长 Token 挤占推理显存与产生巨额 API 账单。
</details>
""")

print("Wrote MLCoding08.md")

with open("notes/MLCoding/MLCoding08 Minimalist Agent Loop Architecture 300 Lines Pi Agent.en.md", "w", encoding="utf-8") as f:
    f.write("""# ML Coding 08 · Minimalist Agent Architecture: 300-Line Core Loop, Tool Calling & State Machine Inspired by Pi Agent

In the modern AI agent ecosystem, bloated frameworks like LangChain, AutoGen, and CrewAI introduce complex inheritance hierarchies, hidden prompt injections, and heavy dependency graphs—often leading to **untraceable bugs, context window bloat, and runaway token costs**.

Between 2025 and 2026, Mario Zechner's open-source project **Pi Agent (`pi-coding-agent`)** inspired a paradigm shift with its radical **"Do More With Less"** philosophy:
> **An effective AI agent does not require hundreds of framework abstractions. At its core, an agent is simply an explicit ReAct state machine loop, a compact set of foundational tools (The Four-Tool Philosophy), and strict context budget management. The entire core engine can be implemented in ~300 lines of standard Python!**

This guide breaks down the mathematical POMDP state machine of autonomous agents and presents a **complete, dependency-free, production-grade 300-line agent implementation**.

---

## Module 1: The Agent Essence & The Four-Tool Philosophy

```text
Minimalist Agent System Topology:
[User Prompt] ──► [Context Budget Manager] ──► [LLM Inference Engine]
                                                       │
                                            (Produces Tool Calls?)
                                            ├── [No]  ──► [Return Final Text (Turn Done)]
                                            └── [Yes] ──► [The 4-Tool Sandbox Execution]
                                                                │
                                                                ▼
                                                    [Output Truncation & Injection]
                                                                │ (ReAct Loop)
                                                                └────► [Back to LLM]
```

### Why 4 Tools Suffice for 95% of Real-World Engineering Tasks
Instead of creating specialized tools for every task (e.g. `CalculatorTool`, `GitTool`, `FileSearchTool`), Pi Agent proves that **4 foundational primitives** cover almost all coding and reasoning workflows:
1. **`read_file(path, start_line, end_line)`**: Line-bounded inspection with token budget control;
2. **`write_file(path, content)`**: File creation and complete configuration overwrites;
3. **`edit_file(path, target_chunk, replacement_chunk)`**: Exact string replacement (minimizes token usage and preserves context);
4. **`bash(command, timeout_seconds)`**: Direct access to system shell tools (`git`, `pytest`, `npm`, `grep`, `find`, `curl`).

---

## Module 2: Mathematical Formulation of the Agent State Machine

An agent's multi-step execution can be modeled as a **Partially Observable Markov Decision Process (POMDP)**:
- **Trajectory History**:
  $$s_t = \\left( h_0, a_1, o_1, a_2, o_2, \\dots, a_t, o_t \\right)$$
- **Policy Distribution**:
  $$\\pi_\\theta(a_{t+1} \\mid s_t) = \\text{LLM}(s_t; \\Theta)$$
- **Deterministic Environment Transition**:
  $$o_{t+1} = \\mathcal{E}(a_{t+1}), \\quad s_{t+1} = s_t \\circ \\left[ a_{t+1}, o_{t+1} \\right]$$

---

## Module 3: Complete 300-Line Pure Python Agent Implementation

```python
import os
import sys
import json
import inspect
import subprocess
import urllib.request
import urllib.error
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass

# 1. Message & Data Models
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]

@dataclass
class Message:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

# 2. Tool Registry & Schema Auto-Inference
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []

    def register(self, func: Callable) -> Callable:
        name = func.__name__
        doc = inspect.getdoc(func) or "No description."
        sig = inspect.signature(func)
        properties = {}
        required = []
        type_map = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}

        for p_name, p in sig.parameters.items():
            properties[p_name] = {"type": type_map.get(p.annotation, "string"), "description": f"Parameter {p_name}"}
            if p.default == inspect.Parameter.empty:
                required.append(p_name)

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": doc.strip(),
                "parameters": {"type": "object", "properties": properties, "required": required}
            }
        }
        self._tools[name] = func
        self._schemas.append(schema)
        return func

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Error: Tool '{name}' is not registered."
        try:
            res = str(self._tools[name](**args))
            if len(res) > 50000:
                res = res[:25000] + "\\n\\n... [Output Truncated: Exceeded 50KB] ...\\n\\n" + res[-25000:]
            return res
        except Exception as e:
            return f"Tool Execution Error ({name}): {type(e).__name__}: {str(e)}"

    @property
    def schemas(self) -> List[Dict[str, Any]]:
        return self._schemas

# 3. Built-in 4 Core Tools
registry = ToolRegistry()

@registry.register
def read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
    \"\"\"Read contents of a file within a 1-indexed line range.\"\"\"
    if not os.path.exists(path):
        return f"Error: File '{path}' does not exist."
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        s, e = max(0, start_line - 1), min(len(lines), end_line)
        return f"File: {path} (Lines {s+1}-{e} of {len(lines)}):\\n" + "".join([f"{s+i+1}: {l}" for i, l in enumerate(lines[s:e])])
    except Exception as err:
        return f"Error reading '{path}': {str(err)}"

@registry.register
def write_file(path: str, content: str) -> str:
    \"\"\"Create or overwrite a file with content.\"\"\"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{path}'."
    except Exception as err:
        return f"Error writing '{path}': {str(err)}"

@registry.register
def edit_file(path: str, target_chunk: str, replacement_chunk: str) -> str:
    \"\"\"Replace a unique target text chunk with replacement text.\"\"\"
    if not os.path.exists(path):
        return f"Error: File '{path}' does not exist."
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if content.count(target_chunk) != 1:
            return f"Error: target_chunk matched {content.count(target_chunk)} times. Must match exactly once."
        new_content = content.replace(target_chunk, replacement_chunk, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Successfully edited '{path}'."
    except Exception as err:
        return f"Error editing '{path}': {str(err)}"

@registry.register
def bash(command: str, timeout_seconds: int = 30) -> str:
    \"\"\"Execute a shell command and capture stdout / stderr.\"\"\"
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout_seconds, cwd=os.getcwd())
        parts = [f"Exit Code: {proc.returncode}"]
        if proc.stdout.strip(): parts.append(f"STDOUT:\\n{proc.stdout.strip()}")
        if proc.stderr.strip(): parts.append(f"STDERR:\\n{proc.stderr.strip()}")
        return "\\n\\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout_seconds} seconds."
    except Exception as err:
        return f"Error: {str(err)}"

# 4. Standard LLM Client
class LLMClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat_completion(self, messages: List[Message], tools: List[Dict[str, Any]]) -> Message:
        payload = {"model": self.model, "messages": [m.to_dict() for m in messages], "temperature": 0.2}
        if tools: payload["tools"] = tools; payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        msg = data["choices"][0]["message"]
        tool_calls = [ToolCall(tc["id"], tc["function"]["name"], json.loads(tc["function"]["arguments"])) for tc in msg.get("tool_calls", [])]
        return Message(role=msg.get("role", "assistant"), content=msg.get("content"), tool_calls=tool_calls if tool_calls else None)

# 5. Core ReAct Agent Loop
class PiAgent:
    def __init__(self, client: LLMClient, tool_registry: ToolRegistry, max_steps: int = 25):
        self.client, self.tools, self.max_steps = client, tool_registry, max_steps
        self.system_prompt = "You are an autonomous engineer with 4 tools: read_file, write_file, edit_file, bash. Inspect before editing. Verify with bash."
        self.history: List[Message] = [Message(role="system", content=self.system_prompt)]

    def run_turn(self, user_query: str) -> str:
        self.history.append(Message(role="user", content=user_query))
        for step in range(1, self.max_steps + 1):
            msg = self.client.chat_completion(self.history, self.tools.schemas)
            self.history.append(msg)
            if not msg.tool_calls:
                return msg.content or ""
            for tc in msg.tool_calls:
                obs = self.tools.execute(tc.name, tc.arguments)
                self.history.append(Message(role="tool", content=obs, tool_call_id=tc.id))
        return "Warning: Reached maximum execution steps."

if __name__ == "__main__":
    agent = PiAgent(LLMClient(), registry)
    agent.run_turn("Inspect workspace directory and summarize project structure.")
```

---

## Module 4: High-Yield Agent Architecture Quizzes

<details class="exercise">
<summary><span class="q-label">Q1 · Agent Tooling Minimalism</span> <span class="q-text">Why does the Pi Agent architecture strictly advocate for a minimal 4-tool primitive set (read/write/edit/bash) over registering dozens of granular domain-specific tools?</span></summary>

- [ ] **A.** LLMs have a hard constraint limiting tool definitions to at most 4.
- [ ] **B.** Specialized tools execute slower than system shell calls.
- [x] **C.** Excessive tool schemas inflate the system prompt token budget and increase model selection hallucinations; a compact tool set with universal shell execution covers complex workflows with minimal overhead.
- [ ] **D.** The 4-tool model is restricted to CLI terminals and cannot be applied elsewhere.

> 💡 **Explanation**:
> - **Correct Answer: C**. Each registered tool requires a detailed JSON Schema definition. Having too many tools consumes valuable prompt context and dilutes attention, leading to tool-selection errors.
</details>

<details class="exercise">
<summary><span class="q-label">Q2 · Context Protection in Long Executions</span> <span class="q-text">When a tool execution generates a 5MB stderr log trace, what is the best context management strategy to preserve diagnostic power without crashing the context window?</span></summary>

- [ ] **A.** Pass all 5MB into the `tool` message directly.
- [ ] **B.** Discard all output and return only the integer returncode.
- [x] **C.** Apply Head/Tail Truncation: preserve the first 20KB (command start/args) and last 20KB (final traceback/summary) with a clear truncation note in between.
- [ ] **D.** Abort the entire session.

> 💡 **Explanation**:
> - **Correct Answer: C**. High-volume logs place their most informative content at the initial invocation setup and the trailing traceback summary. Head/tail truncation maintains key error signals within a strict 50KB boundary.
</details>
""")

print("Wrote MLCoding08.en.md")
