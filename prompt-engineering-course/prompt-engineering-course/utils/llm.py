"""One chat interface over a local Ollama model OR a hosted OpenAI model.

Every lab imports from here so switching backends is one line in .env.
    from utils import ask, chat, count_tokens
"""
import os
from dotenv import load_dotenv

load_dotenv()

BACKEND = os.environ.get("MODEL_BACKEND", "ollama")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def chat(messages, temperature=0.0, **kwargs):
    """messages: list of {"role": ..., "content": ...}. Returns assistant text."""
    if BACKEND == "ollama":
        import ollama
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": temperature},
        )
        return resp["message"]["content"]
    if BACKEND == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, temperature=temperature, **kwargs
        )
        return resp.choices[0].message.content
    raise ValueError(f"Unknown MODEL_BACKEND: {BACKEND!r} (use 'ollama' or 'openai')")


def ask(prompt, system=None, temperature=0.0):
    """Convenience wrapper for a single-turn prompt."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return chat(msgs, temperature=temperature)


def count_tokens(text, model="gpt-4o-mini"):
    """Token count via tiktoken; falls back to a ~chars/4 estimate offline.

    tiktoken downloads its vocabulary on first use, so the exact count needs
    network access once (it is then cached). Without it, we estimate.
    """
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)  # rough offline fallback
