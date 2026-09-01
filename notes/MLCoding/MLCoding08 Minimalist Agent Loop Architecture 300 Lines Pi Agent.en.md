# ML Coding 08 · Minimalist Agent Architecture: 300-Line Core Loop, Tool Calling & State Machine Inspired by Pi Agent

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
  $$s_t = \left( h_0, a_1, o_1, a_2, o_2, \dots, a_t, o_t \right)$$
- **Policy Distribution**:
  $$\pi_\theta(a_{t+1} \mid s_t) = \text{LLM}(s_t; \Theta)$$
- **Deterministic Environment Transition**:
  $$o_{t+1} = \mathcal{E}(a_{t+1}), \quad s_{t+1} = s_t \circ \left[ a_{t+1}, o_{t+1} \right]$$

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
                res = res[:25000] + "\n\n... [Output Truncated: Exceeded 50KB] ...\n\n" + res[-25000:]
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
    """Read contents of a file within a 1-indexed line range."""
    if not os.path.exists(path):
        return f"Error: File '{path}' does not exist."
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        s, e = max(0, start_line - 1), min(len(lines), end_line)
        return f"File: {path} (Lines {s+1}-{e} of {len(lines)}):\n" + "".join([f"{s+i+1}: {l}" for i, l in enumerate(lines[s:e])])
    except Exception as err:
        return f"Error reading '{path}': {str(err)}"

@registry.register
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file with content."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{path}'."
    except Exception as err:
        return f"Error writing '{path}': {str(err)}"

@registry.register
def edit_file(path: str, target_chunk: str, replacement_chunk: str) -> str:
    """Replace a unique target text chunk with replacement text."""
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
    """Execute a shell command and capture stdout / stderr."""
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout_seconds, cwd=os.getcwd())
        parts = [f"Exit Code: {proc.returncode}"]
        if proc.stdout.strip(): parts.append(f"STDOUT:\n{proc.stdout.strip()}")
        if proc.stderr.strip(): parts.append(f"STDERR:\n{proc.stderr.strip()}")
        return "\n\n".join(parts)
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
