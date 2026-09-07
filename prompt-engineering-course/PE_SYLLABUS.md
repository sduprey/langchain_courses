# Prompt & Context Engineering (+ Git) — Syllabus & Agenda

**Instructor:** Stefan Duprey
**Term:** September–October 2026
**Companion repository:** your fork of `prompt-engineering-course` (the course scaffold)
**Total contact time:** 16 h 50 across 9 sessions

---

## 1. Course description

This course treats prompting as an engineering discipline: **specify, measure, iterate**.
It moves from a working mental model of LLMs, through core techniques, to the two ideas that
turn prompting into engineering — **evaluation** (introduced early, in Session 3) and
**context engineering** (deciding what fills the model's window: retrieval, tools, memory).
It closes on the production concerns that decide whether a prompt survives contact with the
real world: cost, latency, model choice, and safety.

**Git** is woven through rather than taught as a separate module: cloning and committing
before the first lab, branching and remotes as labs begin, and pull requests, review and
releases once the capstone becomes a team effort. Every lab and the capstone are handed in
through Git.

The material is **original** and built for this course. A companion course,
*Building LLM Applications with LangChain & LangGraph*, reuses these foundations to build
full applications; this course does not depend on it.

## 2. Learning outcomes

By the end of the course, students will be able to:

1. Use Git confidently: commit, branch, merge, resolve conflicts, work with remotes, and
   collaborate through pull requests and tagged releases.
2. Explain how an LLM produces text and why prompts behave the way they do.
3. Apply core prompting techniques and match prompt patterns to task types
   (classification, extraction, summarization, generation).
4. **Make prompting measurable**: build an eval set, use an LLM-as-judge, catch regressions,
   and test prompt sensitivity.
5. Use reasoning strategies (chain-of-thought, decomposition, self-consistency, ReAct) and
   produce validated structured output.
6. Practice **context engineering**: grounding via retrieval, and treating tools and memory
   as context you design.
7. Reason about cost, latency and model choice, and apply reliability patterns.
8. Recognize and mitigate prompt injection (direct and indirect), jailbreaks and data
   exfiltration.

## 3. Prerequisites

- Working Python (functions, classes, virtual environments, notebooks).
- Command-line comfort helps; no prior Git experience assumed.
- Basic ML familiarity is helpful but not required.

## 4. Tooling & setup (do before Session 1)

The course runs on **two tracks**, chosen deliberately:

- **Track A — local (Ollama), free & offline.** Foundations through retrieval
  (Sessions 1–5). Students iterate on prompts with no API key, no cost, and no data leaving
  their machine. Install [Ollama](https://ollama.com); `ollama pull llama3` and
  `ollama pull nomic-embed-text`.
- **Track B — hosted (OpenAI).** The tool-calling (Session 6) and safety (Session 8) labs,
  where a stronger model makes the exercise realistic.

Switching tracks is one line in `.env` (`MODEL_BACKEND=ollama` or `openai`); every lab
imports the same `utils.ask` / `utils.chat` wrapper.

Setup checklist:

- **Git** installed and configured, plus a **GitHub account** — everything is handed in
  through Git.
- Python 3.10+ with a virtual environment; `pip install -r requirements.txt` (dependencies
  are hard-pinned for reproducibility); Jupyter or VS Code.
- Install **Ollama** and pull the models above.
- An OpenAI API key for Track B, in a local `.env` — never committed.

> **Teaching angle:** the same prompt on a small local model and a hosted frontier model
> shows that technique is model-dependent — explicit structure and few-shot examples matter
> more on smaller models. This is made concrete in Session 7.

## 5. Assessment (proposed — adjust to your school's rules)

| Component | Weight | Notes |
|---|---|---|
| Lab participation | 30 % | Notebooks completed and handed in as **Git commits / branches** on a personal fork |
| Mid-course quiz | 20 % | Foundations + techniques + evaluation (through Session 4) |
| Capstone project | 50 % | Team build, **evaluated** (eval set + judge + one injection test), delivered via pull requests and a tagged release, presented Session 9 |

## 6. Schedule at a glance

| # | Date | Time | Dur. | Git thread | Prompt/context-engineering thread |
|---|---|---|---|---|---|
| 1 | Mar 15 sept. | 8h00–9h50 | 1h50 | init, add, commit, log, `.gitignore` | Foundations: LLMs & anatomy of a prompt |
| 2 | Mer 16 sept. | 8h00–9h40 | 1h40 | branching, merging, conflicts | Core techniques & task patterns |
| 3 | Ven 18 sept. | 8h00–10h30 | 2h30 | remotes, fork, the GitHub flow | **Evaluation & iteration — the pivot** (lab) |
| 4 | Mar 22 sept. | 10h45–12h30 | 1h45 | branch-per-task, good commits | Reasoning & structured outputs |
| 5 | Mer 23 sept. | 8h00–9h30 | 1h30 | secrets & `.gitignore` | Context engineering I — retrieval |
| 6 | Jeu 24 sept. | 11h00–13h30 | 2h30 | pull requests, review, team repos | Context engineering II — tools & memory + capstone brief |
| 7 | Mar 29 sept. | 8h00–9h45 | 1h45 | merge vs rebase, stash | Cost, latency & model choice (lab) |
| 8 | Mer 30 sept. | 8h00–9h35 | 1h35 | undo (revert/reset), tags, conflicts | Safety & security — injection & guardrails (lab) |
| 9 | Jeu 1er oct. | 11h30–13h15 | 1h45 | deliver via PRs & tags, CI/CD teaser | Capstone presentations & the future |

Total Git time ≈ 4 h 40, front-loaded and returning for the capstone; prompt and context
engineering are the spine throughout.

## 7. Session plans (woven run-of-show)

**S1 · 15 Sept (1h50).** 0:00 intro + course map · 0:15 Git foundations + hands-on
(why VCS, add/commit/log/diff, `.gitignore`) · 1:00 LLM mental model & prompt anatomy ·
1:35 Lab 0 — setup, first commit, first prompts (Track A).

**S2 · 16 Sept (1h40).** 0:00 Git — branching, merging, resolving a conflict · 0:55 core
techniques (zero/few-shot, role, structure) & task patterns · 1:25 lab: fix weak prompts,
build a few-shot classifier, committed step by step.

**S3 · 18 Sept (2h30) — the pivot.** 0:00 recap · 0:10 Git — remotes, fork, the GitHub flow ·
0:50 evaluation, LLM-as-judge, prompt sensitivity · 1:20 big lab: build an eval harness over
your classifier and score a prompt change, delivered via a pull request.

**S4 · 22 Sept (1h45).** 0:00 Git — branch per task, commit hygiene · 0:15 reasoning
(CoT, decomposition, self-consistency, ReAct) & structured output with validate/repair ·
1:00 lab: does chain-of-thought actually help? measure it on your eval set · 1:40 wrap.

**S5 · 23 Sept (1h30).** 0:00 Git — keeping secrets out of the repo · 0:15 context
engineering as retrieval (grounding, citation, refusal, context rot) · 0:40 lab: an offline,
grounded RAG that cites sources and refuses — measured.

**S6 · 24 Sept (2h30).** 0:00 Git — pull requests & code review · 0:30 tools & memory as
context (tool descriptions as prompts; memory as curated context) + lab (Track B) · 1:50
capstone brief and team-repo setup.

**S7 · 29 Sept (1h45).** 0:00 Git — merge vs rebase, stash · 0:20 cost, latency & model
choice; reliability patterns · 0:55 lab: the same task on a small vs a frontier model —
quality vs cost vs latency · 1:40 capstone check-in.

**S8 · 30 Sept (1h35).** 0:00 Git — undo safely, tags/releases, team conflicts · 0:20
prompt injection (direct & indirect via documents), jailbreaks, exfiltration, guardrails ·
0:55 lab: attack your own RAG bot, add a guardrail, add an injection eval case.

**S9 · 1 Oct (1h45).** 0:00 team presentations, delivered via merged PRs and a tagged
release · 1:20 consolidation (prompt/context-engineering cheat sheet) · 1:35 where next —
automatic prompt optimization (DSPy), context engineering, agents, and wiring evals into CI ·
1:40 closing & resources.

## 8. Repository map

`utils/` unified model wrapper · `eval/` evaluation harness (from S3) ·
`01_foundations` · `02_techniques` · `03_evaluation` · `04_reasoning_structure` ·
`05_context_engineering` · `06_tools_memory` · `07_cost_latency` · `08_safety` ·
`capstone/` (brief, rubric, starter) · `GIT_WORKFLOW.md`. Dependencies are hard-pinned in
`requirements.txt`.

## 9. Suggested further resources

- Pro Git (free online book) and GitHub's own guides for the Git thread.
- OpenAI / Anthropic prompt-engineering guides.
- DSPy (programmatic prompt optimization) for the Session 9 outlook.
- The companion course: *Building LLM Applications with LangChain & LangGraph*.
