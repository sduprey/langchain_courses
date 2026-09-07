# Git & Prompt Engineering — Course Syllabus & Agenda

**Instructor:** Stefan Duprey
**Term:** September–October 2026
**Companion repository:** https://github.com/sduprey/langchain_courses
**Total contact time:** 16 h 50 across 9 sessions

---

## 1. Course description

This course teaches prompt engineering — from a working mental model of LLMs through
reasoning, structured outputs, retrieval, tools, agents, evaluation and safety — and weaves
**Git** through it as the workflow every lab and the capstone run on. Rather than a separate
module, Git is taught where students need it: cloning and committing before the first lab,
branching and remotes as the labs begin, and pull requests, review and releases when the
capstone becomes a team effort. The hands-on labs use the notebooks in the companion
repository (four mirrored DeepLearning.AI short courses); the material on Git, fundamentals,
evaluation and security is additional content built for this course.

## 2. Learning outcomes

By the end of the course, students will be able to:

1. Use Git confidently: commit, branch, merge, resolve conflicts, work with remotes, and
   collaborate through pull requests.
2. Explain how an LLM produces text and why prompts behave the way they do.
3. Apply core prompting techniques and iterate on prompts methodically.
4. Use reasoning strategies (chain-of-thought, self-consistency, ReAct) appropriately.
5. Produce and validate structured outputs (JSON / schemas).
6. Build LLM applications with LangChain: templates, parsers, memory, chains and RAG.
7. Implement tool calling and multi-step agents with LangGraph, including memory.
8. Evaluate LLM applications and defend them against prompt injection.

## 3. Prerequisites

- Working Python (functions, classes, virtual environments, notebooks).
- Command-line comfort helps; no prior Git or LangChain experience assumed.
- Basic ML familiarity is helpful but not required.

## 4. Tooling & setup (do before Session 1)

The course runs on **two tracks**, chosen deliberately for pedagogy:

- **Track A — local (Ollama), free & offline.** Fundamentals half (Sessions 1–5). The
  `llm_application_development/` directory recodes the LangChain application-development
  material against **Ollama** instead of OpenAI, so students iterate on prompts with no API
  key, no cost, and no data leaving their machine.
- **Track B — hosted (OpenAI + Tavily).** Agent/tool/memory labs (Sessions 6–8), where the
  `01-`, `03-` and `04-` notebooks call hosted models and web search.

Setup checklist:

- **Git** installed and configured (`git config --global user.name / user.email`), plus a
  **GitHub account** — everything is handed in through Git.
- Python 3.10+ with a virtual environment; Jupyter / VS Code.
- Install **Ollama** and pull a model (`ollama pull llama3`) plus an embedding model
  (`ollama pull nomic-embed-text`).
- API keys for Track B (OpenAI, Tavily) in a local `.env` — never committed.

> **Teaching angle:** the same prompt run against a small local model and a hosted frontier
> model shows students that prompt technique is model-dependent — explicit structure and
> few-shot examples matter more on smaller models.

## 5. Assessment (proposed — adjust to your school's rules)

| Component | Weight | Notes |
|---|---|---|
| Lab participation | 30 % | Notebooks completed and handed in as **Git commits / branches** on a personal fork |
| Mid-course quiz | 20 % | Fundamentals + techniques + basic Git (Session 5) |
| Capstone project | 50 % | Team build, evaluated, **delivered via pull requests and a tagged release**, presented Session 9 |

## 6. Schedule at a glance

| # | Date | Time | Dur. | Git thread | Prompt-engineering thread |
|---|---|---|---|---|---|
| 1 | Mar 15 sept. | 8h00–9h50 | 1h50 | init, add, commit, log, `.gitignore` | LLM mental model, anatomy of a prompt |
| 2 | Mer 16 sept. | 8h00–9h40 | 1h40 | branching, merging, conflicts | Core prompting techniques |
| 3 | Ven 18 sept. | 8h00–10h30 | 2h30 | remotes, fork, the GitHub flow | Reasoning & structured outputs (lab) |
| 4 | Mar 22 sept. | 10h45–12h30 | 1h45 | branch-per-task, good commits | LangChain I — models, prompts, parsers, memory |
| 5 | Mer 23 sept. | 8h00–9h30 | 1h30 | secrets & `.gitignore` | LangChain II — chains & RAG |
| 6 | Jeu 24 sept. | 11h00–13h30 | 2h30 | pull requests, review, team repos | Tools & function calling (lab) + capstone brief |
| 7 | Mar 29 sept. | 8h00–9h45 | 1h45 | merge vs rebase, stash | Agents & LangGraph |
| 8 | Mer 30 sept. | 8h00–9h35 | 1h35 | undo (revert/reset), tags, conflicts | Memory, evaluation & safety |
| 9 | Jeu 1er oct. | 11h30–13h15 | 1h45 | deliver via PRs & tags, CI/CD teaser | Capstone presentations & wrap-up |

Total Git time ≈ 4 h 40, front-loaded and returning for the capstone; prompt engineering
remains the spine of the course.

## 7. Session plans (woven run-of-show)

**S1 · 15 Sept (1h50).** 0:00 intro to both subjects · 0:15 Git foundations + hands-on
(why VCS, add/commit/log/diff, `.gitignore`) · 1:15 LLM mental model · 1:35 Lab 0 — setup,
first commit, first prompts (Track A).

**S2 · 16 Sept (1h40).** 0:00 Git — branching, merging, resolving a conflict · 0:55 prompting
techniques (zero/few-shot, role, structure) · 1:25 lab: fix weak prompts, committed.

**S3 · 18 Sept (2h30).** 0:00 recap · 0:10 Git — remotes, fork, the GitHub flow · 0:50
reasoning & structured outputs · 1:20 big lab: structured extraction, delivered via a branch
and pull request.

**S4 · 22 Sept (1h45).** 0:00 Git — branch per task, commit hygiene · 0:15 LangChain I ·
1:00 lab (local, on a branch) · 1:40 wrap.

**S5 · 23 Sept (1h30).** 0:00 Git — keeping secrets out of the repo · 0:15 chains & RAG ·
0:40 lab: offline document QA + short quiz.

**S6 · 24 Sept (2h30).** 0:00 Git — pull requests & code review · 0:30 tools & function
calling + lab (Track B) · 1:50 capstone brief and team-repo setup.

**S7 · 29 Sept (1h45).** 0:00 Git — merge vs rebase, stash · 0:20 agents & LangGraph ·
0:55 lab: build & trace an agent · 1:40 capstone check-in.

**S8 · 30 Sept (1h35).** 0:00 Git — undo safely, tags/releases, team conflicts · 0:20
memory, evaluation & safety · 0:55 lab + final capstone logistics.

**S9 · 1 Oct (1h45).** 0:00 team presentations, delivered via merged PRs and a tagged
release · 1:20 consolidation (Git + prompt-engineering cheat sheets) · 1:35 where next
(CI/CD teaser) · 1:40 closing & resources.

## 8. Content beyond the repo (added for this course)

Git (the full thread above); the LLM mental model; the prompting-technique taxonomy and
iterate–measure–refine method; reasoning strategies as prompt design; structured outputs and
validation; evaluation (eval sets, LLM-as-judge, regression); prompt security (injection,
jailbreaks, guardrails, PII); and the capstone framework and rubric.

## 9. Suggested further resources

- Pro Git (free online book) and GitHub's own guides for the Git thread.
- OpenAI / Anthropic prompt-engineering guides.
- The four source DeepLearning.AI short courses (free, graded, video-paired).
- LangChain & LangGraph documentation.
