"""L5 - Tools and routing, recoded from L5-tools-routing-apis-student.ipynb.

The `@tool` decorator, `args_schema`, and the two real tools (Open-Meteo
weather lookup, Wikipedia search) are unchanged from the original - none of
that is OpenAI-specific. What changed:

- `format_tool_to_openai_function` was removed; `bind_tools()` now accepts
  `BaseTool` objects directly; `convert_to_openai_tool` is the closest thing
  left for just inspecting the JSON schema a tool would produce.
- The old `OpenAIFunctionsAgentOutputParser` -> `AgentAction` / `AgentFinish`
  split is gone. The modern equivalent reads straight off the `AIMessage`:
  `message.tool_calls` is empty -> the model is done and `message.content`
  is the answer; otherwise each entry in `message.tool_calls` names a tool
  and its arguments to run.

The OpenAPI-spec section from the original (`openapi_spec_to_openai_fn`,
tied to OpenAI's function format) is dropped - `routing.py`/L6 cover the
same "let the model pick a tool" idea with tools that are actually callable.
"""

import datetime

import requests
import wikipedia
from common import configure_wikipedia, get_model, traced
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

configure_wikipedia()


# --- @tool basics ---


@tool
def search(query: str) -> str:
    """Search for weather online"""
    return "42f"


print("--- basic @tool ---")
print("name:", search.name)
print("description:", search.description)
print("args:", search.args)


class SearchInput(BaseModel):
    query: str = Field(description="Thing to search for")


@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search for the weather online."""
    return "42f"


print("\n--- @tool with an explicit args_schema ---")
print("args:", search.args)
print("run:", search.run("sf"))


# --- A real tool: Open-Meteo current temperature ---


class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")


@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> str:
    """Fetch current temperature for given coordinates."""
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m",
        "forecast_days": 1,
    }
    response = requests.get(base_url, params=params, timeout=15)
    if response.status_code != 200:
        raise Exception(f"API Request failed with status code: {response.status_code}")
    results = response.json()

    current_utc_time = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    time_list = [
        datetime.datetime.fromisoformat(time_str.replace("Z", "+00:00")).replace(tzinfo=None)
        for time_str in results["hourly"]["time"]
    ]
    temperature_list = results["hourly"]["temperature_2m"]
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    return f"The current temperature is {current_temperature}°C"


print("\n--- get_current_temperature ---")
print("name:", get_current_temperature.name)
print("description:", get_current_temperature.description)
print("args:", get_current_temperature.args)
print("direct call:", get_current_temperature.invoke({"latitude": 48.85, "longitude": 2.35}))


# --- A real tool: Wikipedia search ---


@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    # Wikimedia occasionally rate-limits or hiccups on any of these calls;
    # wrap the whole lookup so the tool degrades gracefully instead of
    # crashing the chain that called it.
    try:
        page_titles = wikipedia.search(query)
    except (requests.exceptions.RequestException, ValueError):
        return "No good Wikipedia Search Result was found"

    summaries = []
    for page_title in page_titles[:3]:
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            wikipedia.exceptions.PageError,
            wikipedia.exceptions.DisambiguationError,
            requests.exceptions.RequestException,
            ValueError,  # wikipedia's own JSON decode wrapper on a bad/rate-limited response
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)


print("\n--- search_wikipedia ---")
print("name:", search_wikipedia.name)
print("description:", search_wikipedia.description)
print(search_wikipedia.invoke({"query": "langchain"})[:300], "...")

print("\n--- convert_to_openai_tool (inspection only) ---")
print(convert_to_openai_tool(get_current_temperature))


# --- Routing: bind both tools, let the model choose ---

tools = [search_wikipedia, get_current_temperature]
model = get_model().bind_tools(tools)

print("\n--- direct invoke: weather question ---")
print(model.invoke("what is the weather in sf right now", config=traced("L5: direct invoke weather")).tool_calls)

print("\n--- direct invoke: knowledge question ---")
print(model.invoke("what is langchain", config=traced("L5: direct invoke knowledge")).tool_calls)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful but sassy assistant"),
        ("user", "{input}"),
    ]
)
chain = prompt | model


def route(message):
    """Modern stand-in for the removed OpenAIFunctionsAgentOutputParser + AgentFinish/AgentAction split."""
    if not message.tool_calls:
        return message.content
    tool_map = {t.name: t for t in tools}
    call = message.tool_calls[0]
    return tool_map[call["name"]].invoke(call["args"])


routed_chain = chain | route

print("\n--- routed chain: weather ---")
print(
    routed_chain.invoke(
        {"input": "What is the weather in san francisco right now?"}, config=traced("L5: routed weather")
    )
)

print("\n--- routed chain: knowledge ---")
print(routed_chain.invoke({"input": "What is langchain?"}, config=traced("L5: routed knowledge"))[:300], "...")

print("\n--- routed chain: plain greeting ---")
print(routed_chain.invoke({"input": "hi!"}, config=traced("L5: routed greeting")))
