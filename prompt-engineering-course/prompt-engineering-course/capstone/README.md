# Capstone project

Build a small, prompt-driven application, evaluate it honestly, and present it.

## What to build (pick one)
- A grounded RAG question-answering bot over a document set of your choice.
- A structured extractor that turns messy text into validated records.
- A small tool-using agent that completes a multi-step task.

## Requirements
1. **Prompt design** — your prompts live in `starter/prompts.py`, versioned in Git,
   with commit messages that show how they evolved.
2. **Evaluation** — an eval set (`starter/eval_set.jsonl`) run through the course
   `eval` harness, plus one LLM-as-judge check. Report the numbers.
3. **Safety** — at least one prompt-injection test case in your eval set, and a
   guardrail that blunts it.
4. **Delivery** — team repo, work integrated via reviewed pull requests, submitted
   as a tagged release (`v1.0`).

## Presentation (Session 9, ~10 min)
Build & demo (3) · prompt design + iterations (2) · evaluation & numbers (2) ·
failure modes + one safety concern + Q&A (3).

## Rubric (100)
| Area | Pts | What we look for |
|---|---|---|
| Prompt design | 25 | Clear, structured prompts; evidence of measured iteration |
| Evaluation | 30 | A real eval set + judge; reported before/after numbers |
| Safety | 15 | A demonstrated risk and a working mitigation |
| Delivery & Git | 20 | Clean history, PRs, tagged release, reproducible run |
| Presentation | 10 | Clear demo and honest reflection |
