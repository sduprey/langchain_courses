# Prompt & Context Engineering — course repo

A hands-on course that treats prompting as engineering: specify, measure, iterate.
It is deliberately original material (not vendor tutorials) so the prompt-engineering
spine stands on its own. Git is woven through every lab — you hand work in via commits,
branches, and pull requests.

> Companion course: the LangChain/LangGraph decks are a **separate** applied-frameworks
> course. This repo is the prompt/context-engineering half and does not depend on it.

## Two tracks
- **Track A — local (Ollama).** Free, offline, private. Foundations through retrieval
  (Sessions 1–5). Install [Ollama](https://ollama.com), then `ollama pull llama3` and
  `ollama pull nomic-embed-text`.
- **Track B — hosted (OpenAI).** For the tool-calling and safety labs (Sessions 6, 8),
  where a stronger model matters. Needs an API key in `.env`.

Switching tracks is one line in `.env` (`MODEL_BACKEND=ollama` or `openai`) — every lab
imports the same `utils.ask` / `utils.chat`.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # edit it; .env is gitignored — never commit it
jupyter notebook
```

## Layout
```
utils/              unified model wrapper (ollama | openai) + token counting
eval/               tiny evaluation harness + a sample dataset  (introduced in S3)
01_foundations/     tokens, sampling, prompt anatomy
02_techniques/      zero/few-shot, task patterns, a real classifier
03_evaluation/      build an eval set, LLM-as-judge, prompt sensitivity   <- the pivot
04_reasoning_structure/  chain-of-thought (measured), structured output + repair
05_context_engineering/  retrieval as prompting (offline RAG), context rot
06_tools_memory/    tool calling (description = prompt), memory as context
07_cost_latency/    token economics, small vs frontier model
08_safety/          prompt injection (direct + indirect), guardrails
capstone/           brief, rubric, and a starter app
GIT_WORKFLOW.md     the exact Git flow the labs use
```

## Session → folder map
| Session | Topic | Folder |
|---|---|---|
| 1 | Foundations: LLMs & prompt anatomy | `01_foundations/` |
| 2 | Core techniques & task patterns | `02_techniques/` |
| 3 | Evaluation & iteration (the pivot) | `03_evaluation/` |
| 4 | Reasoning & structured output | `04_reasoning_structure/` |
| 5 | Context engineering I — retrieval | `05_context_engineering/` |
| 6 | Context engineering II — tools & memory | `06_tools_memory/` |
| 7 | Cost, latency & model choice | `07_cost_latency/` |
| 8 | Safety & security | `08_safety/` |
| 9 | Capstone & the future | `capstone/` |

## What runs today
`utils/` and `eval/` are complete and tested (the harness runs offline). The flagship
notebooks — tokenization, the few-shot classifier, building an eval set, LLM-as-judge,
structured output, and the injection demo — contain working code. The remaining
notebooks are deliberate scaffolds: a stated objective, starter imports, and TODOs the
students fill in during the lab.

## Handing in
See `GIT_WORKFLOW.md`. Short version: fork, branch per lab, commit with a message that
states what changed and the numbers, push, open a PR. Capstone ships as a tagged release.
