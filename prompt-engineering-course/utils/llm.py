"""One chat interface over local Ollama models OR a hosted OpenAI model.

Every lab imports from here. Two model slots are configured in `.env`:

    SMALL_MODEL   a small local model  (default llama3.2:3b)   -- technique matters here
    BIG_MODEL     a strong model       (default gpt-oss:120b-cloud) -- technique matters less

    from utils import ask, chat, count_tokens, SMALL_MODEL, BIG_MODEL

Pass `model=` to target one explicitly; omit it to use the default (`OLLAMA_MODEL`
when the backend is ollama, `OPENAI_MODEL` when it is openai).
"""
import os
from dotenv import load_dotenv

load_dotenv()

BACKEND = os.environ.get("MODEL_BACKEND", "ollama")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# The two slots the labs compare. Technique (few-shot, CoT, structure, ...) has a
# large effect on SMALL_MODEL and a small one on BIG_MODEL -- that contrast is the
# point of most notebooks, so they run both.
SMALL_MODEL = os.environ.get("SMALL_MODEL", "llama3.2:3b")
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-oss:120b-cloud")

TOKENIZER_NOTE = (
    "Token counts below use tiktoken (OpenAI's o200k_base). A Llama model tokenizes "
    "differently -- expect the real count to differ by ~10-20%. Good enough for budgeting."
)


def _ollama_chat(messages, temperature, model, **kwargs):
    import time

    import ollama

    last = None
    for attempt in range(4):  # cloud tags occasionally drop a connection; retry
        try:
            resp = ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature},
                **kwargs,
            )
            return resp["message"]["content"]
        except Exception as e:  # noqa: BLE001 - narrow retry, re-raise below
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def _openai_chat(messages, temperature, model, **kwargs):
    from openai import OpenAI

    resp = OpenAI().chat.completions.create(
        model=model, messages=messages, temperature=temperature, **kwargs
    )
    return resp.choices[0].message.content


def chat(messages, temperature=0.0, model=None, **kwargs):
    """messages: list of {"role": ..., "content": ...}. Returns assistant text.

    model=None  -> the backend default (OLLAMA_MODEL / OPENAI_MODEL)
    model="..." -> that model via Ollama (or via OpenAI when MODEL_BACKEND=openai)
    """
    if BACKEND == "openai":
        return _openai_chat(messages, temperature, model or OPENAI_MODEL, **kwargs)
    if BACKEND == "ollama":
        return _ollama_chat(messages, temperature, model or OLLAMA_MODEL, **kwargs)
    raise ValueError(f"Unknown MODEL_BACKEND: {BACKEND!r} (use 'ollama' or 'openai')")


def ask(prompt, system=None, temperature=0.0, model=None):
    """Convenience wrapper for a single-turn prompt."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return chat(msgs, temperature=temperature, model=model)


def count_tokens(text, model="gpt-4o"):
    """Approximate token count via tiktoken; falls back to a ~chars/4 estimate offline.

    This is a GPT-family tokenizer. The local Llama models the course runs on use a
    different one, so treat the number as an estimate (see TOKENIZER_NOTE). tiktoken
    downloads its vocabulary on first use (then caches it); without network we estimate.
    """
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)  # rough offline fallback
