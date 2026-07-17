"""L6 - Conversational agent, recoded from L6-functional_conversation-student.ipynb.

The original hand-builds an agent loop: a `RunnablePassthrough.assign(...)`
chain re-invoked in a `while True`, `AgentFinish`/`AgentActionMessageLog`
checked by `isinstance`, results fed back through
`format_to_openai_functions`, then wrapped in `AgentExecutor` with a
`ConversationBufferMemory`. All of that - `AgentExecutor`,
`OpenAIFunctionsAgentOutputParser`, `format_to_openai_functions`,
`ConversationBufferMemory` - was removed from LangChain 1.x in favor of a
single call: `langchain.agents.create_agent`, built on LangGraph. It runs
the same tool-call-until-done loop internally, and a `checkpointer` gives it
cross-turn memory keyed by a `thread_id` instead of a manually threaded
`chat_history` list.

The original's Panel GUI chatbot at the end is replaced with a plain
terminal REPL (`python L6_conversational_agent.py`) - Panel is a heavyweight
dependency for what's fundamentally the same `agent.invoke(...)` call shown
right above it non-interactively.
"""

import datetime

import requests
import wikipedia
from common import configure_wikipedia, get_model, traced
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

configure_wikipedia()


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


@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
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
            ValueError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)


@tool
def create_your_own(query: str) -> str:
    """This function can do whatever you would like once you fill it in"""
    return query[::-1]


tools = [get_current_temperature, search_wikipedia, create_your_own]

agent = create_agent(
    model=get_model(),
    tools=tools,
    system_prompt="You are helpful but sassy assistant",
    checkpointer=InMemorySaver(),
)


def ask(thread_id: str, text: str) -> str:
    """One turn of conversation on a given thread; the checkpointer keeps prior turns for that thread_id."""
    result = agent.invoke(
        {"messages": [HumanMessage(content=text)]},
        config=traced(run_name=f"L6: {text[:40]}", configurable={"thread_id": thread_id}),
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Same scripted exchange as the notebook: memory persists across
        # turns, then a tool call, all on one thread_id.
        thread = "demo"
        for turn in ["my name is bob", "whats my name", "whats the weather in sf?"]:
            print(f"\n> {turn}")
            print(ask(thread, turn))
    else:
        print("Conversational agent ready. Type 'exit' to quit.\n")
        thread = "cli"
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue
            print("Bot:", ask(thread, user_input))
