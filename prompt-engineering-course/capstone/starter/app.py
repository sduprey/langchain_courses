"""Minimal capstone skeleton. Fill in retrieve() and wire your own eval.

Run:  PYTHONPATH=../.. python app.py "your question"
"""
import sys
from prompts import SYSTEM, build_answer_prompt
from utils import ask  # from the course utils/ package


def retrieve(question: str) -> str:
    # TODO: replace with your retrieval (vector search over your documents).
    return "[1] Placeholder context. Replace retrieve() with real retrieval."


def answer(question: str) -> str:
    context = retrieve(question)
    return ask(build_answer_prompt(context, question), system=SYSTEM)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What is this project about?"
    print(answer(q))
