"""L3 - Function calling in LangChain, recoded from L3-function-calling-student.ipynb.

The original notebook hand-converts Pydantic models to OpenAI function specs
via `convert_pydantic_to_openai_function`, then does `model.bind(functions=...)`.
That helper and `.bind(functions=...)` were both removed in modern LangChain:
you now pass Pydantic models (or plain callables, or BaseTool instances)
straight into `model.bind_tools([...])`, and LangChain converts them under
the hood. The Pydantic-syntax primer at the top is unchanged - it's plain
Pydantic, nothing OpenAI- or Ollama-specific about it.
"""

from typing import Optional

from common import get_model, traced
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

# --- Pydantic syntax ---


class User:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email


foo = User(name="Joe", age=32, email="joe@gmail.com")
print("plain class age (no validation):", User(name="Joe", age="bar", email="joe@gmail.com").age)


class pUser(BaseModel):
    name: str
    age: int
    email: str


foo_p = pUser(name="Jane", age=32, email="jane@gmail.com")
print("pydantic name:", foo_p.name)

print("\n--- pydantic validation error (expected) ---")
try:
    pUser(name="Jane", age="bar", email="jane@gmail.com")
except ValidationError as e:
    print(e)


class Class(BaseModel):
    students: list[pUser]


obj = Class(students=[pUser(name="Jane", age=32, email="jane@gmail.com")])
print("\nnested pydantic model:", obj)


# --- Pydantic model as a tool schema ---


class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""

    airport_code: str = Field(description="airport code to get weather for")


model = get_model()

print("\n--- bind_tools with a single Pydantic model ---")
model_with_function = model.bind_tools([WeatherSearch])
result = model_with_function.invoke("what is the weather in sf?", config=traced("L3: single Pydantic tool"))
print(result.tool_calls)


# --- Forcing it to use a tool ---
# The original notebook forces OpenAI to call a specific function via
# `function_call={"name": ...}`. `bind_tools(..., tool_choice=...)` accepts
# the same idea, but langchain-ollama documents it as a no-op: "This
# parameter is currently ignored as it is not supported by Ollama." Passing
# it does nothing here - proven below, `hi!` gets no tool call either way.
# The practical workaround, same one used in L1, is to only offer the one
# tool you want called.

print("\n--- tool_choice is accepted but ignored by Ollama (see comment above) ---")
model_with_forced_function = model.bind_tools([WeatherSearch], tool_choice="WeatherSearch")
result = model_with_forced_function.invoke("hi!", config=traced("L3: forced tool_choice (ignored)"))
print("tool_calls (expect empty - tool_choice was not honored):", result.tool_calls)


# --- Using in a chain ---

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "{input}"),
    ]
)
chain = prompt | model_with_function

print("\n--- chain: prompt | model_with_function ---")
result = chain.invoke({"input": "what is the weather in sf?"}, config=traced("L3: prompt | model_with_function"))
print(result.tool_calls)


# --- Using multiple tools ---


class ArtistSearch(BaseModel):
    """Call this to get the names of songs by a particular artist"""

    artist_name: str = Field(description="name of artist to look up")
    n: int = Field(description="number of results")


model_with_functions = model.bind_tools([WeatherSearch, ArtistSearch])

print("\n--- multiple tools: weather question ---")
print(model_with_functions.invoke("what is the weather in sf?", config=traced("L3: multi-tool weather")).tool_calls)

print("\n--- multiple tools: artist question ---")
print(
    model_with_functions.invoke(
        "what are three songs by taylor swift?", config=traced("L3: multi-tool artist")
    ).tool_calls
)

print("\n--- multiple tools: plain greeting (no tool call expected) ---")
result = model_with_functions.invoke("hi!", config=traced("L3: multi-tool greeting"))
print("content:", result.content)
print("tool_calls:", result.tool_calls)
