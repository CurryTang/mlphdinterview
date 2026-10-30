---
name: mlphdinterview
description: Comprehensive technical interview preparation & knowledge retrieval skill. Accesses 290+ deep-dive notes across MLSys (including MAI-Thinking-1, GPU arch, CUDA, FlashAttention, vLLM, distributed training), System Design, Quant, LeetCode, and ML Coding, plus the ByteDance 65 high-frequency technical problem bank for automated mock interviews and concept drilling.
---

# ML PhD & Quant Technical Interview Knowledge & Mock Skill (`mlphdinterview`)

Use this skill when preparing for or conducting technical interviews in **ML Systems (MLSys)**, **Distributed Training & Inference Infra**, **Backend & LLM System Design**, **Quantitative Finance & Stochastic Processes**, **Core Algorithms (LeetCode)**, and **Industrial Machine Learning / RecSys**.

---

## 1 · When to Activate This Skill

- **Technical Concept Lookups**: Deep-dive theoretical principles, architectural trade-offs, and mathematical derivations from the 290+ notes in this repository (e.g., LatentMoE, Token-level GRPO, All-to-All communication bottlenecks, PagedAttention, KV Cache paging, Kafka zero-copy, Martingales, Ito's Lemma).
- **Automated Mock Interviews**: Conducting realistic mock interviews where the AI plays the role of a Principal Engineer / Interviewer using authentic interview problems from the ByteDance 65 question bank or system design cases.
- **Concept Quizzing & Flashcard Drilling**: Rapid-fire concept verification across hardware boundaries, operator optimizations, and distributed cluster failure recovery.

---

## 2 · Available Interfaces

### Interface A: Standard MCP Tools (Preferred when MCP server is active)

When the `mlphdinterview` MCP server is registered in your environment:
1. `search_knowledge_base(query, category?, language?, limit?)`: Semantic keyword search across all 290+ notes.
2. `get_note(note_path, section?, language?, max_length?)`: Fetch full note or specific section markdown.
3. `list_knowledge_topics(category?)`: Retrieve topic catalog and bilingual coverage.
4. `get_interview_question(question_id?, category?, type?, random?)`: Pull real interview problems from ByteDance 65 bank or system design cases.
5. `get_practice_quiz(quiz_id?, category?, random?, hide_answer?)`: Concept multiple-choice verification.
6. `evaluate_answer(question_id, candidate_answer?)`: Evaluate candidate answers against reference solutions, complexity expectations, and core criteria.

### Interface B: Direct Workspace Fallback (When MCP server is not running)

- **Search Notes**: Inspect files in `notes/Mlsys/`, `notes/SystemDesign/`, `notes/quant/`, `notes/Leetcode/`, `notes/MLCoding/`, `notes/BusinessAlgorithm/`.
- **Question Bank**: Read `interview_questions_index.txt` directly (contains all 65 high-frequency interview problems with specifications, time/space complexity expectations, and reference note mappings).
- **CLI MCP Run**: Execute `node mcp/index.js` or start local HTTP API via `npm run mcp:http` (port 3333).

---

## 3 · Mock Interview Protocol

When conducting a mock interview, strictly follow this three-phase protocol:

### Phase 1: Problem Presentation (Zero-Leakage)
- Pull a question using `get_interview_question({ random: true, category: ... })` or from `interview_questions_index.txt`.
- Present **ONLY**:
  - Problem title and scenario background
  - Precise input and output specifications
  - Constraints and boundary conditions
- **DO NOT** reveal the algorithm name, time/space complexity targets, or code solution upfront.

### Phase 2: Interactive Probing & Hints
- Ask the candidate to explain their thought process, edge-case analysis, and complexity trade-offs before writing code.
- If the candidate gets stuck, give subtle, incremental hints based on the question's `mustCoverKeyPoints` without giving away the full solution.

### Phase 3: Comprehensive Evaluation
- Call `evaluate_answer` with the question ID.
- Grade the candidate's solution across:
  1. **Algorithmic Correctness & Completeness**: Did the solution meet all edge-case requirements?
  2. **Complexity Bounds**: Did time and space complexity match optimal expectations?
  3. **Production Engineering Quality**: Cleanliness, boundary condition handling, exception safety.
  4. **Systems Scalability**: How the solution behaves under concurrent or large-scale distributed conditions.

---

## 4 · Standard Knowledge Retrieval Protocol

When answering technical questions:
1. Call `search_knowledge_base` with specific keywords (e.g., `"MAI-Thinking-1"`, `"FlashAttention-2"`, `"PagedAttention"`).
2. Use `get_note` to retrieve the primary section text.
3. Deliver technical knowledge directly, concisely, and objectively with mathematical formulas, hardware data paths, and complexity bounds. No promotional phrasing or clickbait headers.
