# A Hitchhiker's Guide to ML PhD Job Hunting

This repository hosts an interview notes site. MLSYS, LLM, Quant, ML Coding, System Design, business algorithm, ML fundamentals, and LeetCode Core Skills notes are published as parallel interview sections.

## Live Site

**GitHub Pages:** [https://currytang.github.io/mlphdinterview/](https://currytang.github.io/mlphdinterview/)

The site publishes the curated MLSYS notes from `notes/Mlsys/`, Quant notes from `notes/quant/`, ML Coding notes from `notes/MLCoding/`, System Design notes from `notes/SystemDesign/`, TODO placeholders from `notes/BusinessAlgorithm/` and `notes/MLInterview/`, and LeetCode Core Skills notes from `notes/Leetcode/`. The frontend reader supports Chinese and English variants when both exist, and falls back to the available note when only one variant is present.

## Repository Layout

- `notes/Mlsys/`: MLSYS interview note markdown files and local assets
- `notes/quant/`: Quant probability and expectation interview notes
- `notes/MLCoding/`: ML implementation and coding exercise notes
- `notes/SystemDesign/`: backend, LLM serving, and agent infra system design notes
- `notes/BusinessAlgorithm/`: TODO placeholders for business algorithm interview notes
- `notes/MLInterview/`: TODO placeholders for ML fundamentals interview notes
- `notes/Leetcode/`: LeetCode Core Skills note markdown files
- `src/`: React frontend for browsing and rendering interview sections
- `docs/plans/`: design and implementation notes for repo changes

## Practice Blocks

Markdown files can render interactive multiple-choice practice blocks with a fenced code block:

````
```quiz
title: Quick Check
question: CUDA thread blocks are scheduled onto which hardware unit?
answer: B
A. Host compiler
B. GPU SM
C. Browser runtime
explanation: Blocks are assigned to streaming multiprocessors.
```
````

Use `mcq` instead of `quiz` if you prefer that fence name. The rendered block can be hidden, and option clicks show whether the selected answer is correct.

Long-form answers can be kept collapsible with native Markdown/HTML details blocks:

```html
<details class="solution">
<summary>展开解法</summary>

Write the reference solution, pseudocode, and complexity analysis here.

</details>
```

## Local Development

```bash
npm install
npm run dev
```

## Verification

```bash
npm test
npm run lint
npm run build
```

## Deployment Notes

GitHub Pages deployment is handled by `.github/workflows/deploy-pages.yml`.

Vite infers the production base path from `GITHUB_REPOSITORY` in CI. If needed, override it with `VITE_BASE_PATH` during the build.
