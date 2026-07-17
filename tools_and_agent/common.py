"""Shared setup for the tools_and_agent lessons.

Every lesson script does `from common import get_model`. This is the
LangChain equivalent of src/model.py: it loads `.env` once and returns a
`ChatOllama` pointed at the Ollama cloud endpoint, authenticated the same
way call_minimax.py's first_model is (Bearer token in a header).
"""

import os
import sys

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler

from tracing import langfuse  # noqa: F401 - re-exported for scripts that want the raw client too

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

# One handler, reused across every chain/model in a script so their traces
# all flow through the same Langfuse client. NOTE: `.bind_tools()` / `.bind()`
# called on a model *after* callbacks are attached via `.with_config()` drops
# them (verified empirically - bind_tools() forwards to the unwrapped model
# and returns a fresh, config-less binding). So callbacks are passed at
# invoke time instead, via `traced()`, which works regardless of how the
# runnable was built.
langfuse_handler = CallbackHandler()


def traced(run_name: str | None = None, **config) -> dict:
    """RunnableConfig with the Langfuse callback attached: `chain.invoke(x, config=traced("name"))`."""
    cfg = {"callbacks": [langfuse_handler], **config}
    if run_name:
        cfg["run_name"] = run_name
    return cfg


def get_model(temperature: float = 0.0, **kwargs) -> ChatOllama:
    """Build a ChatOllama client for the cloud model configured in `.env`.

    `OLLAMA_MODEL` / `OLLAMA_BASE_URL` / `OLLAMA_API_KEY` are the same three
    variables src/model.py uses for `first_model`.
    """
    return ChatOllama(
        model=os.environ["OLLAMA_MODEL"],
        base_url=os.environ["OLLAMA_BASE_URL"],
        temperature=temperature,
        client_kwargs={
            "headers": {"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"}
        },
        **kwargs,
    )


def strip_json_fence(text: str) -> str:
    """Strip a ```json ... ``` / ``` ... ``` wrapper some models add even with format="json"."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    return text.removesuffix("```").strip()


def configure_wikipedia() -> None:
    """Set a descriptive User-Agent on the `wikipedia` package.

    Wikimedia's API rejects requests with the library's default User-Agent
    (empty JSON body back, which then fails to parse) - it requires one that
    identifies the client per https://meta.wikimedia.org/wiki/User-Agent_policy.
    """
    import wikipedia

    wikipedia.set_user_agent(
        "ollama-playing-ground-tools_and_agent/1.0 (local course exercise; no contact url)"
    )
