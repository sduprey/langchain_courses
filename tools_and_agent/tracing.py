"""Langfuse client setup, shared by every tools_and_agent script.

Kept separate from common.py (which is LangChain-specific) so L1 - which
deliberately talks to the raw `ollama` client, no LangChain - can trace its
calls without importing langchain_ollama just for that.
"""

import atexit

from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

langfuse = get_client()
# These scripts are short-lived; without an explicit flush, the last batch of
# spans queued right before process exit can be dropped.
atexit.register(langfuse.flush)
