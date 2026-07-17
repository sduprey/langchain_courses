"""L1 - Tool calling basics, recoded from L1-openai_functions_student.ipynb.

The original notebook calls the raw `openai` SDK directly (no LangChain yet)
to show what "function calling" looks like at the wire level: you send a
JSON schema of your function(s) alongside the messages, and the model
either replies normally or replies with a `function_call` you're supposed
to execute yourself.

Ollama's `/api/chat` endpoint speaks the same shape (OpenAI-compatible
`tools=[{"type": "function", "function": {...}}]`), so this recodes the
lesson with the raw `ollama` client instead of LangChain - LangChain enters
in L2 onward, same as the original course.

Since there's no LangChain runnable here for Langfuse's `CallbackHandler` to
attach to, each call is wrapped by hand in `langfuse.start_as_current_observation(...,
as_type="generation")` - the same pattern src/model.py's `query()` uses, just
targeting the "generation" observation type so token usage shows up too.
"""

import json
import os
import sys

from dotenv import load_dotenv
from ollama import Client

from tracing import langfuse

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

client = Client(
    host=os.environ["OLLAMA_BASE_URL"],
    headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
)
MODEL = os.environ["OLLAMA_MODEL"]


def traced_chat(name, messages, tools=None):
    """`client.chat(...)`, logged to Langfuse as a `generation` observation."""
    with langfuse.start_as_current_observation(
        name=name,
        as_type="generation",
        model=MODEL,
        input=messages,
    ) as generation:
        response = client.chat(model=MODEL, messages=messages, tools=tools)
        generation.update(
            output=response.message.model_dump(exclude_none=True),
            usage_details={
                "input": response.prompt_eval_count,
                "output": response.eval_count,
            },
        )
        return response


# Example dummy function hard coded to return the same weather.
# In production, this could be your backend API or an external API.
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


# Same JSON schema shape as the OpenAI "functions" list, just wrapped in the
# {"type": "function", "function": {...}} envelope Ollama's tools API expects.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]


def show(title, response):
    print(f"\n--- {title} ---")
    print("message:", response.message)


messages = [{"role": "user", "content": "What's the weather like in Boston?"}]
response = traced_chat("L1: weather question, tools available", messages, tools=tools)
show("weather question, tools available", response)

messages = [{"role": "user", "content": "hi!"}]
response = traced_chat("L1: greeting, tools available but unused", messages, tools=tools)
show("greeting, tools available but unused", response)

# Note: unlike OpenAI's `function_call="none"` / `function_call={"name": ...}`,
# Ollama has no wire-level knob to force or forbid a tool call - the model
# decides on its own from the prompt and tool descriptions. If you need to
# force a specific tool, the usual trick is to only offer that one tool.

# --- Executing the call and completing the round trip ---
messages = [{"role": "user", "content": "What's the weather like in Boston!"}]
response = traced_chat("L1: weather question again", messages, tools=tools)
show("weather question again", response)

assistant_message = response.message
messages.append(assistant_message)

if assistant_message.tool_calls:
    call = assistant_message.tool_calls[0]
    args = call.function.arguments
    observation = get_current_weather(**args)
    print("\ntool result:", observation)

    messages.append(
        {
            "role": "tool",
            "content": observation,
            "tool_name": call.function.name,
        }
    )

    final_response = traced_chat("L1: final answer after tool result", messages)
    show("final answer after tool result", final_response)
    print("\n", final_response.message.content)
