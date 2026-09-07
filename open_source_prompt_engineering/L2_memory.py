"""L2 - Memory, recoded from L2-Memory.ipynb.

`ConversationChain`, `ConversationBufferMemory`, `ConversationBufferWindowMemory`,
`ConversationTokenBufferMemory` and `ConversationSummaryBufferMemory` were all
removed - modern LangChain dropped the "memory object" abstraction entirely.
For an agent with tool calls, the replacement is a LangGraph checkpointer
keyed by `thread_id` (see tools_and_agent/L6_conversational_agent.py). For
plain conversation - no tools, just "what strategy keeps history under
control" - the replacement is simpler still: manage a `list[BaseMessage]`
yourself and decide what to keep before each call.

This recodes the same four strategies as small classes over that list, so
the actual behavioral differences (drop nothing / keep last k / keep what
fits a token budget / summarize the overflow) are still visible side by
side, just without a `ConversationXMemory` class name attached to each.

Token counts here are a `len(text) // 4` approximation, not `tiktoken` -
that library counts OpenAI's specific tokenizer, which has no bearing on
how the Ollama-hosted cloud model actually tokenizes text. It's good enough
to demonstrate "trim when the budget is exceeded."
"""

from common import get_model, traced
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

SYSTEM_PROMPT = "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context."


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ChatHistory:
    """Base strategy: keep everything (the `ConversationBufferMemory` equivalent)."""

    def __init__(self):
        self.messages: list[BaseMessage] = []

    def save_context(self, human_input: str, ai_output: str) -> None:
        self.messages.append(HumanMessage(content=human_input))
        self.messages.append(AIMessage(content=ai_output))

    def load_memory_variables(self) -> list[BaseMessage]:
        return list(self.messages)


class WindowChatHistory(ChatHistory):
    """Keep only the last `k` exchanges (the `ConversationBufferWindowMemory` equivalent)."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def load_memory_variables(self) -> list[BaseMessage]:
        return self.messages[-2 * self.k :]


class TokenBufferChatHistory(ChatHistory):
    """Keep as many of the most recent messages as fit under `max_tokens` (approx)."""

    def __init__(self, max_tokens: int):
        super().__init__()
        self.max_tokens = max_tokens

    def load_memory_variables(self) -> list[BaseMessage]:
        kept: list[BaseMessage] = []
        total = 0
        for message in reversed(self.messages):
            cost = approx_tokens(message.content)
            if total + cost > self.max_tokens:
                break
            kept.append(message)
            total += cost
        return list(reversed(kept))


class SummaryBufferChatHistory(ChatHistory):
    """Once the buffer overflows `max_tokens`, summarize the oldest exchanges away."""

    def __init__(self, model, max_tokens: int):
        super().__init__()
        self.model = model
        self.max_tokens = max_tokens
        self.summary = ""

    def save_context(self, human_input: str, ai_output: str) -> None:
        super().save_context(human_input, ai_output)
        self._compress_if_needed()

    def _compress_if_needed(self) -> None:
        total = sum(approx_tokens(m.content) for m in self.messages) + approx_tokens(self.summary)
        if total <= self.max_tokens or len(self.messages) <= 2:
            return
        to_summarize, self.messages = self.messages[:-2], self.messages[-2:]
        transcript = "\n".join(
            f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in to_summarize
        )
        prompt = (
            "Progressively summarize the lines of conversation provided, adding onto "
            f"the previous summary and returning a new summary.\n\n"
            f"Current summary:\n{self.summary or '(none yet)'}\n\n"
            f"New lines of conversation:\n{transcript}\n\n"
            "New summary:"
        )
        self.summary = self.model.invoke(
            prompt, config=traced("L2: summary buffer compaction")
        ).content

    def load_memory_variables(self) -> list[BaseMessage]:
        history: list[BaseMessage] = []
        if self.summary:
            history.append(SystemMessage(content=f"Summary of earlier conversation:\n{self.summary}"))
        return history + self.messages


def converse(model, history: ChatHistory, user_input: str, run_name: str) -> str:
    """The `ConversationChain.predict(input=...)` equivalent: load history, call the model, save the turn."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *history.load_memory_variables(), HumanMessage(content=user_input)]
    response = model.invoke(messages, config=traced(run_name))
    history.save_context(user_input, response.content)
    return response.content


model = get_model()


# --- ConversationBufferMemory equivalent ---

print("--- buffer: full conversation ---")
history = ChatHistory()
print(converse(model, history, "Hi, my name is Andrew", "L2: buffer turn 1"))
print(converse(model, history, "What is 1+1?", "L2: buffer turn 2"))
print(converse(model, history, "What is my name?", "L2: buffer turn 3"))
print("\nloaded history:")
for m in history.load_memory_variables():
    print(f"  {type(m).__name__}: {m.content}")


# --- save_context / load_memory_variables building blocks ---

print("\n--- raw save_context / load_memory_variables ---")
history = ChatHistory()
history.save_context("Hi", "What's up")
history.save_context("Not much, just hanging", "Cool")
for m in history.load_memory_variables():
    print(f"  {type(m).__name__}: {m.content}")


# --- ConversationBufferWindowMemory equivalent ---

print("\n--- window (k=1): only remembers the last exchange ---")
history = WindowChatHistory(k=1)
converse(model, history, "Hi, my name is Andrew", "L2: window turn 1")
converse(model, history, "What is 1+1?", "L2: window turn 2")
print(converse(model, history, "What is my name?", "L2: window turn 3"))
print("(expected: it won't know - only the 1+1 exchange survived the window)")


# --- ConversationTokenBufferMemory equivalent ---

print("\n--- token buffer (max ~15 tokens): only the most recent exchange fits ---")
history = TokenBufferChatHistory(max_tokens=15)
history.save_context("AI is what?!", "Amazing!")
history.save_context("Backpropagation is what?", "Beautiful!")
history.save_context("Chatbots are what?", "Charming!")
for m in history.load_memory_variables():
    print(f"  {type(m).__name__}: {m.content}")


# --- ConversationSummaryBufferMemory equivalent ---

schedule = (
    "There is a meeting at 8am with your product team. "
    "You will need your powerpoint presentation prepared. "
    "9am-12pm have time to work on your LangChain project which will go quickly "
    "because Langchain is such a powerful tool. At Noon, lunch at the italian "
    "restaurant with a customer who is driving from over an hour away to meet you "
    "to understand the latest in AI. Be sure to bring your laptop to show the "
    "latest LLM demo."
)

print("\n--- summary buffer (max ~100 tokens): old turns get compacted ---")
history = SummaryBufferChatHistory(model, max_tokens=100)
history.save_context("Hello", "What's up")
history.save_context("Not much, just hanging", "Cool")
history.save_context("What is on the schedule today?", schedule)
print("summary so far:", history.summary or "(not compacted yet)")
for m in history.load_memory_variables():
    print(f"  {type(m).__name__}: {m.content[:200]}")

print("\n" + converse(model, history, "What would be a good demo to show?", "L2: summary buffer turn"))
print("\nfinal summary:", history.summary)
