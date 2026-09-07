"""Langfuse client setup, shared by every llm_application_development script.

Same as tools_and_agent/tracing.py - kept framework-agnostic so any script
that doesn't need LangChain callbacks can still trace with the raw client.
"""

import atexit

from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

langfuse = get_client()
# These scripts are short-lived; without an explicit flush, the last batch of
# spans queued right before process exit can be dropped.
atexit.register(langfuse.flush)
